"""
[파일] Camera_only.py
[목적] 카메라만으로 RED → YELLOW → BLUE 색지를 순서대로 찾아가 1초 정지
       SolvePnP 로 색지까지의 실제 거리(Z mm)와 수평 오프셋(X mm)을 추정,
       '바퀴 축이 색지 안에 들어갔을 때' 정확히 멈춤

[탐색(SEARCH) 전략]
  ① 강탐지 (area > 1000px) : SolvePnP 기반 정밀 접근
  ② 약탐지 (area > 200px)  : 해당 방향으로 천천히 유도 (탐색 타이머 리셋)
  ③ 미탐지                  : 호회전(arc) 스위프 — 빠른 피벗 대신 전진+조향으로
                              카메라 시야를 천천히 쓸어가며 재탐색

[BLUE 바닥 vs 벽 구분]
  색지 = 2D 수평면 → 이미지에서 가로가 세로보다 넓고(W/H > 0.45)
                      화면 상단 35% 에만 있지 않음
  벽/배너 = 3D 수직면 → 세로가 가로보다 길고 화면 상단에 치우침

[현장에서 반드시 측정·조정해야 하는 값]
  PAPER_W_MM / PAPER_H_MM  : 실제 색지 크기 (자로 측정)
  WHEEL_AXLE_DIST_MM        : 카메라 렌즈 ~ 앞 바퀴 축 전방 거리 (mm)

[아두이노 명령 프로토콜]
  F {steer:.2f} {speed:.2f}\n  → 전진
  T {dir:.2f}\n                → 제자리 피벗 (+우 / -좌)  ← 탐색에는 미사용
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
PAPER_W_MM         = 300.0   # 색지 가로 길이 (mm)
PAPER_H_MM         = 300.0   # 색지 세로 길이 (mm)
WHEEL_AXLE_DIST_MM = 60.0  # 카메라 정사영 ~ 뒷바퀴 축 전방 거리 (mm)

# ─────────────────────────────────────────────────────────────────────────────
#  주행 파라미터
# ─────────────────────────────────────────────────────────────────────────────
MAX_STEER       = 1.0
SPEED_FAR       = 0.55        # 정상 접근 속도 (먼 거리)
SPEED_NEAR      = 0.35        # 감속 속도 (가까운 거리)
DIST_SLOW_MM    = 100.0       # 감속 시작 거리 (mm)
ALIGN_THRES_MM  = 50.0        # 도달 판정 X 허용 오차 (mm)
AREA_PEAK_THRES  = 0.18       # 색지 면적이 이 이상 → 피크 확인 플래그 ON (회전 색지도 포함)
STEER_GAIN      = 0.015       # angle(deg) → steer 변환 게인
CONFIRM_FRAMES  = 4           # 도달 연속 N 프레임 충족 시 확정
STOP_DURATION   = 1.0         # 정지 유지 시간 (초)

LOCK_MIN_AREA   = 0.12        # 잠금 허용 최소 면적비 (이 이상일 때 PnP 신뢰)
LOCK_AVG_FRAMES = 5           # 연속 N 프레임 평균으로 조향 잠금 (노이즈 감소)
SPEED_CRAWL     = 0.22        # 잠금 후 최종 접근 속도 (오버슈팅 방지)

# ─────────────────────────────────────────────────────────────────────────────
#  탐색(SEARCH) 파라미터
# ─────────────────────────────────────────────────────────────────────────────
WEAK_MIN_AREA      = 200      # 약탐지 최소 면적 (px) — 이 이상이면 방향 유도
WEAK_SPEED         = 0.35     # 약탐지 시 전진 속도
WEAK_STEER_GAIN    = 0.60     # 약탐지 시 offset → steer 게인

SEARCH_TIMEOUT     = 1.5      # 미탐지 후 호회전 탐색 시작까지 대기 (초)
SEARCH_ARC_STEER   = 0.55     # 호회전 탐색 조향 세기
SEARCH_ARC_SPEED   = 0.28     # 호회전 탐색 전진 속도
SEARCH_ARC_DUR     = 2.5      # 탐색 방향 전환 주기 (초)

# ─────────────────────────────────────────────────────────────────────────────
#  파란 바닥 매트 ↔ 수직 벽 구분 파라미터
# ─────────────────────────────────────────────────────────────────────────────
BLUE_ASPECT_MIN    = 0.45     # bounding box W/H 최솟값 (이 미만 → 세워진 벽)
BLUE_BOTTOM_MIN    = 0.35     # 컨투어 하단 y / frame_h 최솟값 (이 미만 → 화면 상단 벽)

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
    """4개 점 → [TL, TR, BR, BL] 시계방향 정렬."""
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                     pts[np.argmax(s)], pts[np.argmax(diff)]], dtype=np.float32)


def _extract_quad(contour: np.ndarray) -> np.ndarray:
    """
    컨투어 → 4 꼭짓점.

    근접 시 색지 일부가 프레임 밖으로 잘려도 안정적으로 4점 추출:
      1) 볼록 껍질: 잘린 윤곽의 오목부·노이즈 제거
      2) 적응형 Douglas-Peucker: epsilon 0.01→0.40 탐색, 4점 발견 즉시 반환
      3) 병합 폴백: 4점 초과 시 인접 최단 쌍을 반복 병합 → 4점
      4) 최후 폴백: 점이 4개 미만이면 minAreaRect
    """
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)

    best = hull.reshape(-1, 2).astype(np.float32)
    for eps_ratio in np.arange(0.01, 0.40, 0.01):
        approx = cv2.approxPolyDP(hull, float(eps_ratio) * peri, True)
        pts = approx.reshape(-1, 2).astype(np.float32)
        if len(pts) == 4:
            return _order_points(pts), True   # 클린: 병합 없이 4점 자연 검출
        if len(pts) < 4:
            break       # 이 이상 단순화하면 4점 불가 → best 병합
        best = pts      # 아직 4점 초과이지만 이전보다 단순화됨

    # 인접한 꼭짓점 중 가장 가까운 쌍을 병합해 4점으로 줄임
    while len(best) > 4:
        dists = [np.linalg.norm(best[i] - best[(i + 1) % len(best)])
                 for i in range(len(best))]
        idx = int(np.argmin(dists))
        nxt = (idx + 1) % len(best)
        best[idx] = (best[idx] + best[nxt]) / 2
        best = np.delete(best, nxt, axis=0)

    if len(best) < 4:
        best = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)

    return _order_points(best), False  # 추정: 병합·폴백 사용 → PnP 신뢰 낮음


def solve_paper_pose(contour, cam_mat, dist_coeffs):
    """
    SolvePnP 로 색지 pose 추정.
    Returns (z_mm, x_mm, steer, quad_pts) or None.
      z_mm     : 전방 거리 (mm)
      x_mm     : 수평 오프셋 (mm, 양수=우)
      steer    : 조향값 (-1 ~ +1)
      quad_pts : 추출된 4 꼭짓점 (시각화 재사용 — 이중 호출 방지)
    """
    if cam_mat is None:
        return None
    quad_pts, is_clean = _extract_quad(contour)
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
    return z_mm, x_mm, steer, quad_pts, is_clean


# ─────────────────────────────────────────────────────────────────────────────
#  탐색 보조 함수
# ─────────────────────────────────────────────────────────────────────────────

def _get_mask(hsv, color: str):
    """색상 이름으로 해당 마스크 반환."""
    if color == 'red':    return get_red_mask(hsv)
    if color == 'yellow': return get_yellow_mask(hsv)
    return get_blue_mask(hsv)


def get_weak_contour(hsv, color: str, frame_h: int):
    """
    약탐지: WEAK_MIN_AREA 이상이면 반환 (ColorDetector 기준 min_area=1000보다 낮음).
    blue 는 바닥 필터 추가 적용.
    Returns largest contour or None.
    """
    mask = _get_mask(hsv, color)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > WEAK_MIN_AREA]
    if not cnts:
        return None
    if color == 'blue':
        cnts = [c for c in cnts if is_floor_contour(c, frame_h)]
    return max(cnts, key=cv2.contourArea) if cnts else None


def is_floor_contour(cnt, frame_h: int) -> bool:
    """
    2D 바닥 매트(수평)와 3D 수직 벽/배너를 구분.

    바닥 매트 특징:
      - 카메라가 앞/아래를 향할 때 가로가 세로보다 넓거나 비슷 (W/H >= 0.45)
      - 화면 상단 35% 에만 존재하지 않음 (하단으로 이어져야 함)

    세워진 벽/배너 특징:
      - 세로가 가로보다 훨씬 길고 (W/H < 0.45)
      - 화면 상단에 국한됨
    """
    _, y, w, h_box = cv2.boundingRect(cnt)
    if h_box == 0:
        return False
    aspect   = w / h_box
    bottom_y = (y + h_box) / frame_h   # 컨투어 하단 y 비율

    # 세로로 너무 긴 객체 (세워진 벽) 제거
    if aspect < BLUE_ASPECT_MIN:
        return False
    # 화면 최상단 35% 에만 있는 객체 제거 (원거리 수직 물체)
    if bottom_y < BLUE_BOTTOM_MIN:
        return False
    return True


def _contour_offset(cnt, frame_w: int) -> float:
    """컨투어 무게중심의 좌우 오프셋 (-1.0 좌 ~ +1.0 우)."""
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return 0.0
    cx = M['m10'] / M['m00']
    return (cx - frame_w / 2) / (frame_w / 2)


def search_arc_steer(elapsed_since_timeout: float, base_dir: float) -> float:
    """
    호회전 탐색 조향값 계산.
    base_dir: 마지막으로 타겟을 본 방향 (+1.0 우 / -1.0 좌)
    SEARCH_ARC_DUR 초마다 방향 교대. 첫 번째 방향은 base_dir.
    """
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

    signal.signal(signal.SIGINT,  _sig_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, _sig_handler)   # kill
    signal.signal(signal.SIGTSTP, _sig_handler)   # Ctrl+Z

    target_idx    = 0
    state         = 'SEEK'
    on_zone_count = 0
    stop_start    = None
    last_seen     = time.time()
    last_steer      = 0.0         # 마지막 조향 방향 (탐색 시 초기 방향 결정)
    area_peak_seen  = False       # AREA_PEAK_THRES 이상을 한 번이라도 봤는지
    peak_area_r     = 0.0         # 현재 타겟에서 관측된 최대 면적비
    lock_state      = False       # True = 중심점 잠금 완료, 잠긴 방향으로 주행 중
    locked_steer    = 0.0         # 잠긴 조향값 (N프레임 평균)
    clean_steer_buf = []          # 클린 PnP 조향값 누적 버퍼

    print("=" * 60)
    print("  카메라 색상 추적 주행  |  SolvePnP + Arc Search")
    print(f"  목표: RED → YELLOW → BLUE")
    print(f"  색지 {PAPER_W_MM:.0f}×{PAPER_H_MM:.0f} mm  |  바퀴축 {WHEEL_AXLE_DIST_MM:.0f} mm")
    print(f"  PnP: {'ON' if pnp_mat is not None else 'OFF(fallback)'}")
    print(f"  약탐지 >{WEAK_MIN_AREA}px → 방향 유도 | 탐색: {SEARCH_ARC_DUR}s 호회전 교대")
    print("=" * 60)

    while True:
        ret, raw = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # DONE 상태는 색 탐지 불필요 — 연산 절감
        if state == 'DONE':
            ser.write(b"S\n")
            vis = raw.copy()
            cv2.putText(vis, "MISSION COMPLETE", (fw // 2 - 120, fh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.imshow('Robot View', vis)
            cv2.waitKey(1)
            time.sleep(0.1)
            continue

        # ColorDetector: undistort + HSV + 마스크 일괄 처리
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
                    target_idx      += 1
                    state            = 'SEEK'
                    on_zone_count    = 0
                    area_peak_seen   = False
                    peak_area_r      = 0.0
                    lock_state       = False
                    locked_steer     = 0.0
                    clean_steer_buf  = []
                    last_seen        = time.time()
                    print(f"  ✅ {color.upper()} 완료 → {TARGETS[target_idx].upper()}")
                else:
                    state = 'DONE'
                    print("  ✅ 전체 미션 완료!")
            continue

        # ── SEEK (탐색 · 접근) ────────────────────────────────────────────

        # ── ① 강탐지 (ColorDetector 기준 min_area=1000 충족) ─────────────
        det = result.get(color, {})

        # blue: 바닥 매트 필터 적용 (수직 벽/배너 제거)
        if color == 'blue' and det.get('found'):
            if not is_floor_contour(det['contour'], fh):
                det = {'found': False}   # 바닥 매트 아닌 것으로 판단 → 무시

        if det.get('found'):
            # ── ① 강탐지: 색지 보임 → 접근 주행 ─────────────────────────
            last_seen = time.time()
            cnt = det['contour']
            pose = solve_paper_pose(cnt, pnp_mat, pnp_dist)

            area_r = det['area'] / (fw * fh)
            if area_r > AREA_PEAK_THRES and lock_state:
                area_peak_seen = True
                peak_area_r = max(peak_area_r, area_r)

            if pose is not None:
                z_mm, x_mm, steer, quad_pts, is_clean = pose

                # 클린 PnP + 충분한 면적 → 잠금 버퍼 누적 및 갱신
                if is_clean and area_r >= LOCK_MIN_AREA:
                    clean_steer_buf.append(steer)
                    if len(clean_steer_buf) > LOCK_AVG_FRAMES:
                        clean_steer_buf.pop(0)
                    if len(clean_steer_buf) >= LOCK_AVG_FRAMES:
                        locked_steer = sum(clean_steer_buf) / len(clean_steer_buf)
                        lock_state   = True
                elif not is_clean and not lock_state:
                    # 불완전 코너 + 아직 잠금 전 → 버퍼 초기화 (신뢰 불가)
                    clean_steer_buf.clear()

                steer_cmd = locked_steer if lock_state else steer
                speed     = SPEED_CRAWL  if lock_state else (SPEED_NEAR if z_mm < DIST_SLOW_MM else SPEED_FAR)

                # pnp_reached: 잠금 완료 + 클린 PnP일 때만 신뢰
                pnp_reached = (lock_state and is_clean
                               and z_mm < WHEEL_AXLE_DIST_MM
                               and abs(x_mm) < ALIGN_THRES_MM)
                reached     = pnp_reached

                buf_lbl = "LOCK✓" if lock_state else f"buf:{len(clean_steer_buf)}/{LOCK_AVG_FRAMES}"
                pnp_col = ((0, 255, 0)   if reached   else
                           (0, 200, 255) if lock_state else
                           (180, 180, 0) if not is_clean else (0, 220, 255))
                log_msg = f"PnP z={z_mm:.0f}mm x={x_mm:+.0f}mm {'CLN' if is_clean else 'EST'} {buf_lbl}"
                cv2.putText(vis, f"Z={z_mm:.0f} X={x_mm:+.0f} A={area_r:.3f} {buf_lbl}",
                            (fw // 2 - 170, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, pnp_col, 2)
                cv2.polylines(vis, [quad_pts.astype(np.int32)], True, pnp_col, 2)
                ctr = quad_pts.mean(axis=0).astype(int)
                cv2.circle(vis, tuple(ctr), 6, pnp_col, -1)
                cv2.line(vis, (ctr[0]-15, ctr[1]), (ctr[0]+15, ctr[1]), pnp_col, 1)
                cv2.line(vis, (ctr[0], ctr[1]-15), (ctr[0], ctr[1]+15), pnp_col, 1)

            else:
                # PnP 실패 → 컨투어 중심으로 "4코너 보이도록" 조향
                offset    = det['offset']
                steer_cmd = locked_steer if lock_state else float(np.clip(offset * 0.80, -MAX_STEER, MAX_STEER))
                speed     = SPEED_CRAWL  if lock_state else (SPEED_NEAR if area_peak_seen else SPEED_FAR)
                reached   = False
                log_msg   = f"{'LOCK-fallback' if lock_state else 'seek-4corner'} off={offset:+.2f} A={area_r:.2f}"
                cv2.putText(vis, f"{'LOCK✓' if lock_state else 'seek-4corner'} A={area_r:.3f}",
                            (fw // 2 - 100, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
                M_c = cv2.moments(cnt)
                if M_c['m00'] > 0:
                    ctr = (int(M_c['m10'] / M_c['m00']), int(M_c['m01'] / M_c['m00']))
                    cv2.circle(vis, ctr, 6, (200, 200, 200), -1)
                    cv2.line(vis, (ctr[0]-15, ctr[1]), (ctr[0]+15, ctr[1]), (200, 200, 200), 1)
                    cv2.line(vis, (ctr[0], ctr[1]-15), (ctr[0], ctr[1]+15), (200, 200, 200), 1)

            last_steer = steer_cmd
            if area_peak_seen:
                # 최종 진입 단계: 브랜치 ②가 count 관리, 여기선 감소 금지
                if reached:
                    on_zone_count += 1
            else:
                on_zone_count = on_zone_count + 1 if reached else max(0, on_zone_count - 1)

            if on_zone_count >= CONFIRM_FRAMES:
                state = 'STOP'; stop_start = time.time()
                ser.write(b"S\n")
                print(f"  🎯 {color.upper()} 도달(PnP)! {log_msg}")
                cv2.imshow('Robot View', vis); cv2.waitKey(1)
                continue

            ser.write(f"F {steer_cmd:.2f} {speed:.2f}\n".encode())
            print(f"  [SEEK] {color.upper()} {log_msg} steer={steer_cmd:+.2f} cnt={on_zone_count}")

        elif area_peak_seen:
            # ── ② 피크 후 미탐지: 색지가 카메라 아래로 들어간 상황 ────────
            hsv_u    = cv2.cvtColor(result['undistorted'], cv2.COLOR_BGR2HSV)
            weak_cnt = get_weak_contour(hsv_u, color, fh)

            if weak_cnt is not None:
                # 아직 약하게 보임 → 마저 진입
                if lock_state:
                    steer      = locked_steer   # 잠긴 방향 유지
                    enter_lbl  = f"ENTERING(LOCK) s={steer:+.2f}"
                else:
                    weak_offset = _contour_offset(weak_cnt, fw)
                    steer       = float(np.clip(weak_offset * WEAK_STEER_GAIN,
                                                -MAX_STEER, MAX_STEER))
                    enter_lbl   = f"ENTERING {color.upper()} off={weak_offset:+.2f}"
                last_steer  = steer
                last_seen   = time.time()
                ser.write(f"F {steer:.2f} {WEAK_SPEED:.2f}\n".encode())
                cv2.drawContours(vis, [weak_cnt], -1, (180, 180, 0), 1)
                M_w = cv2.moments(weak_cnt)
                if M_w['m00'] > 0:
                    ctr = (int(M_w['m10'] / M_w['m00']), int(M_w['m01'] / M_w['m00']))
                    cv2.circle(vis, ctr, 6, (180, 255, 0), -1)
                    cv2.line(vis, (ctr[0]-15, ctr[1]), (ctr[0]+15, ctr[1]), (180, 255, 0), 1)
                    cv2.line(vis, (ctr[0], ctr[1]-15), (ctr[0], ctr[1]+15), (180, 255, 0), 1)
                cv2.putText(vis, enter_lbl,
                            (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 0), 2)
                print(f"  [ENTER] {color.upper()} {enter_lbl}")
            else:
                # 완전히 사라짐 → 양쪽 바퀴 모두 색지 위
                on_zone_count += 1
                ser.write(b"S\n")
                log_msg = f"invisible pk={peak_area_r:.2f} lock={'Y' if lock_state else 'N'}"
                cv2.putText(vis, f"ON PAPER  cnt:{on_zone_count}/{CONFIRM_FRAMES}",
                            (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                print(f"  [ON] {color.upper()} {log_msg} cnt={on_zone_count}")
                if on_zone_count >= CONFIRM_FRAMES:
                    state = 'STOP'; stop_start = time.time()
                    ser.write(b"S\n")
                    print(f"  🎯 {color.upper()} 도달(invisible)! {log_msg}")
                    cv2.imshow('Robot View', vis); cv2.waitKey(1)
                    continue

        else:
            # ── ③ 미탐지 (아직 색지 미발견) → 약탐지 후 호회전 탐색 ───────
            on_zone_count = max(0, on_zone_count - 1)
            hsv_u    = cv2.cvtColor(result['undistorted'], cv2.COLOR_BGR2HSV)
            weak_cnt = get_weak_contour(hsv_u, color, fh)

            if weak_cnt is not None:
                # 약탐지 성공 → 해당 방향으로 천천히 유도
                weak_offset = _contour_offset(weak_cnt, fw)
                steer       = float(np.clip(weak_offset * WEAK_STEER_GAIN,
                                            -MAX_STEER, MAX_STEER))
                last_steer  = steer
                last_seen   = time.time()
                ser.write(f"F {steer:.2f} {WEAK_SPEED:.2f}\n".encode())
                cv2.drawContours(vis, [weak_cnt], -1, (180, 180, 0), 1)
                M_w = cv2.moments(weak_cnt)
                if M_w['m00'] > 0:
                    ctr = (int(M_w['m10'] / M_w['m00']), int(M_w['m01'] / M_w['m00']))
                    cv2.circle(vis, ctr, 6, (180, 180, 0), -1)
                    cv2.line(vis, (ctr[0]-15, ctr[1]), (ctr[0]+15, ctr[1]), (180, 180, 0), 1)
                    cv2.line(vis, (ctr[0], ctr[1]-15), (ctr[0], ctr[1]+15), (180, 180, 0), 1)
                cv2.putText(vis, f"WEAK {color.upper()} off={weak_offset:+.2f}",
                            (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 0), 2)
                print(f"  [WEAK] {color.upper()} offset={weak_offset:+.2f} → "
                      f"gentle steer={steer:+.2f}")

            else:
                # ── 완전 미탐지 → 호회전(arc) 탐색 ─────────────────────
                elapsed = time.time() - last_seen

                if elapsed < SEARCH_TIMEOUT:
                    ser.write(b"S\n")
                    cv2.putText(vis, f"WAIT {elapsed:.1f}s",
                                (5, 58), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (100, 100, 255), 2)
                else:
                    t_search = elapsed - SEARCH_TIMEOUT
                    base_dir = 1.0 if last_steer >= 0 else -1.0
                    arc_steer = search_arc_steer(t_search, base_dir)

                    ser.write(f"F {arc_steer:.2f} {SEARCH_ARC_SPEED:.2f}\n".encode())
                    phase_lbl = "→우호전" if arc_steer > 0 else "←좌호전"
                    cv2.putText(vis, f"SEARCH {phase_lbl} {t_search:.1f}s",
                                (5, 58), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (100, 100, 255), 2)
                    print(f"  [SEARCH] {color.upper()} {elapsed:.1f}s 미탐지 "
                          f"arc={arc_steer:+.2f}")

        # ── 공통 HUD ─────────────────────────────────────────────────────
        cnt_bar = f"cnt:{on_zone_count}/{CONFIRM_FRAMES}"
        cv2.putText(vis, f"{state} | {color.upper()} | {cnt_bar}",
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
