"""
[파일] Camera_v2.py
[목적] Camera_only.py 기반 + 적응형 코너 검출로 중심점 조향 정확도 향상
       SolvePnP 로 색지까지의 실제 거리(Z mm)와 수평 오프셋(X mm)을 추정,
       '바퀴 축이 색지 안에 들어갔을 때' 정확히 멈춤

[Camera_only.py 대비 변경점]
  _extract_quad 개선:
    - 볼록 껍질(convexHull) 전처리로 오목부·노이즈 제거
    - 적응형 Douglas-Peucker (epsilon 0.01→0.40): 4점 발견 즉시 반환
    - 병합 폴백: 4점 초과 시 인접 최단 쌍 반복 병합
    - 최후 폴백: minAreaRect
  중심점 시각화: 검출된 4코너 / 컨투어 무게중심을 화면에 표시

[탐색(SEARCH) 전략 — Camera_only.py 동일]
  ① 강탐지 (area > 1000px) : SolvePnP 기반 정밀 접근
  ② 약탐지 (area > 200px)  : 해당 방향으로 천천히 유도
  ③ 미탐지                  : 호회전(arc) 스위프

[정지 조건 — Camera_only.py 동일]
  Path A : PnP z_mm < WHEEL_AXLE_DIST_MM AND abs(x_mm) < ALIGN_THRES_MM
  Path B : area_peak_seen 후 색지 완전 소멸 (바퀴 위)

[현장에서 반드시 측정·조정해야 하는 값]
  PAPER_W_MM / PAPER_H_MM  : 실제 색지 크기 (자로 측정)
  WHEEL_AXLE_DIST_MM        : 카메라 렌즈 ~ 앞 바퀴 축 전방 거리 (mm)

[아두이노 명령 프로토콜]
  F {steer:.2f} {speed:.2f}\n  → 전진
  T {dir:.2f}\n                → 제자리 피벗 (+우 / -좌)
  S\n                          → 즉시 정지
"""

import os
import atexit
import signal
import sys
import serial
import time
import cv2
import numpy as np

from color_v2 import (ColorDetector, load_calibration,
                       get_red_mask, get_yellow_mask, get_blue_mask)


# ─────────────────────────────────────────────────────────────────────────────
#  하드웨어 설정
# ─────────────────────────────────────────────────────────────────────────────
PORT_ARDU  = "/dev/ttyS0"
CAM_INDEX  = 0
CAM_W      = 640
CAM_H      = 480
CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "camera_calibration.pkl")

# ─────────────────────────────────────────────────────────────────────────────
#  현장 측정값 ← 반드시 직접 측정 후 수정!
# ─────────────────────────────────────────────────────────────────────────────
PAPER_W_MM      = 300.0
PAPER_H_MM      = 300.0

# ─────────────────────────────────────────────────────────────────────────────
#  주행 파라미터
# ─────────────────────────────────────────────────────────────────────────────
MAX_STEER       = 1.0
SPEED_FAR       = 0.55
SPEED_NEAR      = 0.35
DIST_SLOW_MM    = 100.0       # 이 거리 미만이면 감속 (PnP 조향 참고용)
AREA_PEAK_THRES = 0.18        # 색지 면적 피크 감지 임계값 → 정지 로직의 시작점
STEER_GAIN      = 0.015
CONFIRM_FRAMES  = 4
STOP_DURATION   = 1.0

# ─────────────────────────────────────────────────────────────────────────────
#  탐색(SEARCH) 파라미터
# ─────────────────────────────────────────────────────────────────────────────
WEAK_MIN_AREA   = 200
WEAK_SPEED      = 0.35
WEAK_STEER_GAIN = 0.60

SEARCH_TIMEOUT  = 1.5
SEARCH_ARC_STEER = 0.55
SEARCH_ARC_SPEED = 0.28
SEARCH_ARC_DUR  = 2.5

# ─────────────────────────────────────────────────────────────────────────────
#  BLUE 바닥 매트 ↔ 수직 벽 구분
# ─────────────────────────────────────────────────────────────────────────────
BLUE_ASPECT_MIN = 0.45
BLUE_BOTTOM_MIN = 0.35

TARGETS = ['red', 'yellow', 'blue']


# ─────────────────────────────────────────────────────────────────────────────
#  SolvePnP 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

_OBJ_PTS = np.array([
    [0,          0,          0],
    [PAPER_W_MM, 0,          0],
    [PAPER_W_MM, PAPER_H_MM, 0],
    [0,          PAPER_H_MM, 0],
], dtype=np.float32)


def _order_points(pts: np.ndarray) -> np.ndarray:
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                     pts[np.argmax(s)], pts[np.argmax(diff)]], dtype=np.float32)


def _extract_quad(contour: np.ndarray) -> np.ndarray:
    """
    컨투어 → 4 꼭짓점 (Camera_only 대비 개선).

    1) 볼록 껍질: 오목부·노이즈 제거
    2) 적응형 Douglas-Peucker (epsilon 0.01→0.40): 4점 즉시 반환
    3) 병합 폴백: 인접 최단 쌍 반복 병합 → 4점
    4) 최후 폴백: minAreaRect
    """
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    best = hull.reshape(-1, 2).astype(np.float32)

    for eps_ratio in np.arange(0.01, 0.40, 0.01):
        approx = cv2.approxPolyDP(hull, float(eps_ratio) * peri, True)
        pts = approx.reshape(-1, 2).astype(np.float32)
        if len(pts) == 4:
            return _order_points(pts)
        if len(pts) < 4:
            break
        best = pts

    while len(best) > 4:
        dists = [np.linalg.norm(best[i] - best[(i + 1) % len(best)])
                 for i in range(len(best))]
        idx = int(np.argmin(dists))
        nxt = (idx + 1) % len(best)
        best[idx] = (best[idx] + best[nxt]) / 2
        best = np.delete(best, nxt, axis=0)

    if len(best) < 4:
        best = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)

    return _order_points(best)


def solve_paper_pose(contour, cam_mat, dist_coeffs):
    if cam_mat is None:
        return None
    quad_pts = _extract_quad(contour)
    try:
        ok, _, tvec = cv2.solvePnP(_OBJ_PTS, quad_pts, cam_mat, dist_coeffs,
                                    flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None
    if not ok:
        return None
    z_mm = float(tvec[2][0])
    x_mm = float(tvec[0][0])
    if z_mm <= 0:
        return None
    angle = np.degrees(np.arctan2(x_mm, z_mm))
    steer = float(np.clip(angle * STEER_GAIN, -MAX_STEER, MAX_STEER))
    return z_mm, x_mm, steer, quad_pts


# ─────────────────────────────────────────────────────────────────────────────
#  탐색 보조 함수
# ─────────────────────────────────────────────────────────────────────────────

def _get_mask(hsv, color: str):
    if color == 'red':    return get_red_mask(hsv)
    if color == 'yellow': return get_yellow_mask(hsv)
    return get_blue_mask(hsv)


def get_weak_contour(hsv, color: str, frame_h: int):
    mask = _get_mask(hsv, color)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > WEAK_MIN_AREA]
    if not cnts:
        return None
    if color == 'blue':
        cnts = [c for c in cnts if is_floor_contour(c, frame_h)]
    return max(cnts, key=cv2.contourArea) if cnts else None


def is_floor_contour(cnt, frame_h: int) -> bool:
    _, y, w, h_box = cv2.boundingRect(cnt)
    if h_box == 0:
        return False
    aspect   = w / h_box
    bottom_y = (y + h_box) / frame_h
    return aspect >= BLUE_ASPECT_MIN and bottom_y >= BLUE_BOTTOM_MIN


def _contour_offset(cnt, frame_w: int) -> float:
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return 0.0
    return (M['m10'] / M['m00'] - frame_w / 2) / (frame_w / 2)


def _draw_center(vis, cx: int, cy: int, color):
    cv2.circle(vis, (cx, cy), 6, color, -1)
    cv2.line(vis, (cx - 15, cy), (cx + 15, cy), color, 1)
    cv2.line(vis, (cx, cy - 15), (cx, cy + 15), color, 1)


def search_arc_steer(elapsed_since_timeout: float, base_dir: float) -> float:
    phase = int(elapsed_since_timeout / SEARCH_ARC_DUR) % 2
    direction = base_dir if phase == 0 else -base_dir
    return direction * SEARCH_ARC_STEER


# ─────────────────────────────────────────────────────────────────────────────
#  메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ser = serial.Serial(PORT_ARDU, 460800, timeout=1)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cam_mat, dist_coeffs, calib_res = load_calibration(CALIB_FILE)
    if calib_res and calib_res != (fw, fh):
        print(f"[경고] 캘리브 해상도 불일치({calib_res} vs {fw}×{fh}) → PnP 비활성화")
        cam_mat = dist_coeffs = None

    detector = ColorDetector(frame_w=fw, frame_h=fh,
                             camera_matrix=cam_mat, dist_coeffs=dist_coeffs)

    if detector._map1 is not None:
        pnp_mat  = detector._new_mtx
        pnp_dist = np.zeros((4, 1), dtype=np.float64)
    else:
        pnp_mat  = cam_mat
        pnp_dist = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1))

    def _cleanup():
        try:
            ser.write(b"S\n"); time.sleep(0.1)
            cap.release(); cv2.destroyAllWindows(); ser.close()
        except Exception:
            pass
    atexit.register(_cleanup)

    def _sig_handler(_sig, _frame):
        _cleanup()
        sys.exit(0)
    signal.signal(signal.SIGINT,  _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGTSTP, _sig_handler)

    target_idx     = 0
    state          = 'SEEK'
    on_zone_count  = 0
    stop_start     = None
    last_seen      = time.time()
    last_steer     = 0.0
    area_peak_seen = False
    peak_area_r    = 0.0

    print("=" * 60)
    print("  Camera_v2  |  SolvePnP + 적응형 코너 검출")
    print(f"  목표: RED → YELLOW → BLUE")
    print(f"  색지 {PAPER_W_MM:.0f}×{PAPER_H_MM:.0f} mm")
    print(f"  PnP: {'ON' if pnp_mat is not None else 'OFF(fallback)'}")
    print("=" * 60)

    while True:
        ret, raw = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        if state == 'DONE':
            ser.write(b"S\n")
            vis = raw.copy()
            cv2.putText(vis, "MISSION COMPLETE", (fw // 2 - 120, fh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.imshow('Robot View', vis)
            cv2.waitKey(1)
            time.sleep(0.1)
            continue

        result = detector.detect(raw)
        color  = TARGETS[target_idx]
        vis    = detector.draw_debug(raw, result)

        # ── STOP (1초 대기) ───────────────────────────────────────────────
        if state == 'STOP':
            ser.write(b"S\n")
            elapsed = time.time() - stop_start
            remain  = max(0.0, STOP_DURATION - elapsed)
            cv2.putText(vis, f"STOP {color.upper()}  {remain:.1f}s",
                        (fw // 2 - 90, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)
            cv2.imshow('Robot View', vis)
            cv2.waitKey(1)
            if elapsed >= STOP_DURATION:
                if target_idx < len(TARGETS) - 1:
                    target_idx     += 1
                    state           = 'SEEK'
                    on_zone_count   = 0
                    area_peak_seen  = False
                    peak_area_r     = 0.0
                    last_seen       = time.time()
                    print(f"  ✅ {color.upper()} 완료 → {TARGETS[target_idx].upper()}")
                else:
                    state = 'DONE'
                    print("  ✅ 전체 미션 완료!")
            continue

        # ── SEEK ─────────────────────────────────────────────────────────
        det = result.get(color, {})
        if color == 'blue' and det.get('found'):
            if not is_floor_contour(det['contour'], fh):
                det = {'found': False}

        # ① 강탐지 ────────────────────────────────────────────────────────
        if det.get('found'):
            last_seen = time.time()
            cnt       = det['contour']
            pose      = solve_paper_pose(cnt, pnp_mat, pnp_dist)
            area_r    = det['area'] / (fw * fh)

            if area_r > AREA_PEAK_THRES:
                area_peak_seen = True
                peak_area_r    = max(peak_area_r, area_r)

            if pose is not None:
                z_mm, x_mm, steer, quad_pts = pose
                speed   = SPEED_NEAR if z_mm < DIST_SLOW_MM else SPEED_FAR
                log_msg = f"PnP z={z_mm:.0f}mm x={x_mm:+.0f}mm area={area_r:.2f}"

                cv2.putText(vis, f"Z={z_mm:.0f}mm X={x_mm:+.0f}mm A={area_r:.3f}",
                            (fw // 2 - 160, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
                cv2.polylines(vis, [quad_pts.astype(np.int32)], True, (0, 220, 255), 2)
                ctr = quad_pts.mean(axis=0).astype(int)
                _draw_center(vis, int(ctr[0]), int(ctr[1]), (0, 220, 255))

            else:
                offset  = det['offset']
                steer   = float(np.clip(offset * 0.80, -MAX_STEER, MAX_STEER))
                speed   = SPEED_NEAR if area_peak_seen else SPEED_FAR
                log_msg = f"fallback offset={offset:+.2f} area={area_r:.2f}"
                cv2.putText(vis, f"A={area_r:.3f} pk={peak_area_r:.3f}",
                            (fw // 2 - 80, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
                M_c = cv2.moments(cnt)
                if M_c['m00'] > 0:
                    cx = int(M_c['m10'] / M_c['m00'])
                    cy = int(M_c['m01'] / M_c['m00'])
                    _draw_center(vis, cx, cy, (200, 200, 200))

            last_steer = steer
            ser.write(f"F {steer:.2f} {speed:.2f}\n".encode())
            print(f"  [SEEK] {color.upper()} {log_msg} steer={steer:+.2f}")

        # ② 피크 후 미탐지 ────────────────────────────────────────────────
        elif area_peak_seen:
            hsv_u    = cv2.cvtColor(result['undistorted'], cv2.COLOR_BGR2HSV)
            weak_cnt = get_weak_contour(hsv_u, color, fh)

            if weak_cnt is not None:
                weak_offset = _contour_offset(weak_cnt, fw)
                steer       = float(np.clip(weak_offset * WEAK_STEER_GAIN,
                                            -MAX_STEER, MAX_STEER))
                last_steer  = steer
                last_seen   = time.time()
                ser.write(f"F {steer:.2f} {WEAK_SPEED:.2f}\n".encode())
                cv2.drawContours(vis, [weak_cnt], -1, (180, 180, 0), 1)
                M_w = cv2.moments(weak_cnt)
                if M_w['m00'] > 0:
                    _draw_center(vis,
                                 int(M_w['m10'] / M_w['m00']),
                                 int(M_w['m01'] / M_w['m00']),
                                 (180, 255, 0))
                cv2.putText(vis, f"ENTERING {color.upper()} off={weak_offset:+.2f}",
                            (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 0), 2)
                print(f"  [ENTER] {color.upper()} offset={weak_offset:+.2f} steer={steer:+.2f}")
            else:
                on_zone_count += 1
                ser.write(b"S\n")
                log_msg = f"invisible after pk={peak_area_r:.2f}"
                cv2.putText(vis, f"ON PAPER  cnt:{on_zone_count}/{CONFIRM_FRAMES}",
                            (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                print(f"  [ON] {color.upper()} {log_msg} cnt={on_zone_count}")
                if on_zone_count >= CONFIRM_FRAMES:
                    state = 'STOP'; stop_start = time.time()
                    ser.write(b"S\n")
                    print(f"  🎯 {color.upper()} 도달(invisible)! {log_msg}")
                    cv2.imshow('Robot View', vis); cv2.waitKey(1)
                    continue

        # ③ 미탐지 → 호회전 탐색 ─────────────────────────────────────────
        else:
            on_zone_count = max(0, on_zone_count - 1)
            hsv_u    = cv2.cvtColor(result['undistorted'], cv2.COLOR_BGR2HSV)
            weak_cnt = get_weak_contour(hsv_u, color, fh)

            if weak_cnt is not None:
                weak_offset = _contour_offset(weak_cnt, fw)
                steer       = float(np.clip(weak_offset * WEAK_STEER_GAIN,
                                            -MAX_STEER, MAX_STEER))
                last_steer  = steer
                last_seen   = time.time()
                ser.write(f"F {steer:.2f} {WEAK_SPEED:.2f}\n".encode())
                cv2.drawContours(vis, [weak_cnt], -1, (180, 180, 0), 1)
                M_w = cv2.moments(weak_cnt)
                if M_w['m00'] > 0:
                    _draw_center(vis,
                                 int(M_w['m10'] / M_w['m00']),
                                 int(M_w['m01'] / M_w['m00']),
                                 (180, 180, 0))
                cv2.putText(vis, f"WEAK {color.upper()} off={weak_offset:+.2f}",
                            (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 0), 2)
                print(f"  [WEAK] {color.upper()} offset={weak_offset:+.2f} steer={steer:+.2f}")
            else:
                elapsed = time.time() - last_seen
                if elapsed < SEARCH_TIMEOUT:
                    ser.write(b"S\n")
                    cv2.putText(vis, f"WAIT {elapsed:.1f}s",
                                (5, 58), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (100, 100, 255), 2)
                else:
                    t_search  = elapsed - SEARCH_TIMEOUT
                    base_dir  = 1.0 if last_steer >= 0 else -1.0
                    arc_steer = search_arc_steer(t_search, base_dir)
                    ser.write(f"F {arc_steer:.2f} {SEARCH_ARC_SPEED:.2f}\n".encode())
                    phase_lbl = "→우호전" if arc_steer > 0 else "←좌호전"
                    cv2.putText(vis, f"SEARCH {phase_lbl} {t_search:.1f}s",
                                (5, 58), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (100, 100, 255), 2)
                    print(f"  [SEARCH] {color.upper()} {elapsed:.1f}s 미탐지 arc={arc_steer:+.2f}")

        # ── 공통 HUD ─────────────────────────────────────────────────────
        cv2.putText(vis, f"{state} | {color.upper()} | cnt:{on_zone_count}/{CONFIRM_FRAMES}",
                    (5, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        cv2.imshow('Robot View', vis)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            print("  [종료] q 입력")
            break

    ser.write(b"S\n")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
