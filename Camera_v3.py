"""
[파일] Camera_v3.py
[목적] Camera_v2 + Ki+Lidar 장점 통합

[Camera_v2에서 가져온 것]
  · color_v2 모듈 — Kobuki 박스 오탐 방지 마스크 포함 최신 색상 감지
  · SolvePnP 정밀 접근 (_extract_quad NaN 가드 포함)
  · area_peak_seen → invisible 정지 메커니즘
  · 깔끔한 SEEK / STOP / DONE 상태 구조

[Ki+Lidar에서 가져온 것]
  · VFH 갭 탐색 (_find_gaps / _best_gap / _compute_vfh)
      - 노이즈 제거 + 로봇 반경 팽창 + 갭 너비·깊이·통과가능성 판단
      - 갭 방향 + 좌우벽 반발 + 코너 반발을 하나의 목표각으로 통합
  · 긴급 후진 (전방 EMERGENCY 이내 → B cmd, 정지 아님)
  · ROT 피벗 모드 (갭이 멀거나 좁으면 제자리 회전)
  · LiDAR 기반 접근 감속 (_speed_limit)
  · 후방 카메라 마운트 마스크 (120°~240° 제외)

[아두이노 명령 프로토콜]
  F {steer:.2f} {speed:.2f}\n  → 전진
  T {dir:.2f}\n                → 제자리 피벗 (+우 / -좌)
  B {speed:.2f}\n              → 후진  ← Arduino 펌웨어 지원 필수
  S\n                          → 즉시 정지
"""

import os
import atexit
import signal
import sys
import math
import threading
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
PORT_LIDAR = "/dev/ttyUSB0"
CAM_INDEX  = 0
CAM_W, CAM_H = 640, 480
CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "camera_calibration.pkl")

# ─────────────────────────────────────────────────────────────────────────────
#  현장 측정값
# ─────────────────────────────────────────────────────────────────────────────
PAPER_W_MM = 300.0
PAPER_H_MM = 300.0

# ─────────────────────────────────────────────────────────────────────────────
#  카메라 주행 파라미터
# ─────────────────────────────────────────────────────────────────────────────
MAX_STEER       = 1.0
SPEED_FAR       = 0.55
SPEED_NEAR      = 0.35
DIST_SLOW_MM    = 100.0
AREA_PEAK_THRES = 0.04
STEER_GAIN      = 0.015
CONFIRM_FRAMES  = 4
STOP_DURATION   = 1.0

WEAK_MIN_AREA   = 200
WEAK_SPEED      = 0.35
WEAK_STEER_GAIN = 0.60
SEARCH_TIMEOUT  = 1.5

TARGETS = ['red', 'yellow', 'blue']

# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR VFH 파라미터
# ─────────────────────────────────────────────────────────────────────────────
BIN_DEG       = 4.0
N_BINS        = int(360 / BIN_DEG)
GAP_MIN_PASS  = 90.0    # 통과 가능 최소 갭 너비 (mm)
DETECT        = 560.0   # 장애물 감지 거리 (mm)
VELO_DOWN     = 400.0   # 감속 시작 거리 (mm)
EMERGENCY     = 150.0   # 긴급 후진 거리 (mm)
P4_DIST       = 170.0   # 전진 불가, ROT 피벗 전환 거리 (mm)
LID_MAX_STEER = 1.2     # VFH 최대 조향값
ROT_THRESH    = 110.0   # 갭이 이 각도 이상이면 ROT/BACK 전환 (deg)
ROBOT_RADIUS  = 35.0    # 장애물 팽창 반경 (mm)

# 카메라 마운트 마스크: 후방 180° ± 60° 제외
MOUNT_MASK_LOW  = 120.0
MOUNT_MASK_HIGH = 240.0


# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def _build_hist(scan_buf):
    hist   = [9999.0] * N_BINS
    has_pt = [False]  * N_BINS
    for a, d in scan_buf:
        if MOUNT_MASK_LOW <= a <= MOUNT_MASK_HIGH:
            continue
        idx = int(a / BIN_DEG) % N_BINS
        if d < hist[idx]:
            hist[idx] = d
            has_pt[idx] = True
    return hist, has_pt


def _nearest(hist, has_pt, center_cw, arc_half=25):
    cb = int(center_cw / BIN_DEG) % N_BINS
    nc = max(1, int(arc_half / BIN_DEG))
    md = 9999.0
    for k in range(-nc, nc + 1):
        idx = (cb + k) % N_BINS
        if has_pt[idx] and hist[idx] < md:
            md = hist[idx]
    return md


def _find_gaps(hist, has_pt):
    # ① 이진 장애물 맵
    blocked = [has_pt[i] and hist[i] <= DETECT for i in range(N_BINS)]
    # ② 노이즈 제거: 고립된 1칸 블록 제거
    smoothed = blocked[:]
    for i in range(N_BINS):
        if blocked[i] and not blocked[(i-1) % N_BINS] and not blocked[(i+1) % N_BINS]:
            smoothed[i] = False
    blocked = smoothed
    # ③ 로봇 반경 팽창: 실제 통과 가능 너비 보수적 계산
    inflated = blocked[:]
    for i in range(N_BINS):
        if blocked[i] and hist[i] < 9999.0:
            ar = math.asin(min(1.0, ROBOT_RADIUS / max(hist[i], ROBOT_RADIUS)))
            ab = int(math.degrees(ar) / BIN_DEG) + 1
            for k in range(-ab, ab + 1):
                inflated[(i + k) % N_BINS] = True
    blocked = inflated
    # ④ 갭(열린 구간) 추출
    gaps = []; seen = set(); i = 0
    while i < 2 * N_BINS:
        if not blocked[i % N_BINS]:
            j = i + 1
            while j < i + N_BINS and not blocked[j % N_BINS]:
                j += 1
            span = j - i
            if span < N_BINS:
                ccw = ((i + j) / 2.0 * BIN_DEG) % 360.0
                ck  = round(ccw)
                if ck not in seen:
                    seen.add(ck)
                    dg = span * BIN_DEG
                    dL = min(hist[(i-1) % N_BINS] if has_pt[(i-1) % N_BINS] else DETECT, DETECT)
                    dR = min(hist[j % N_BINS]     if has_pt[j % N_BINS]     else DETECT, DETECT)
                    gw = (dL + dR) * math.sin(math.radians(dg / 2.0))
                    dp = min(hist[k % N_BINS] for k in range(i, j))
                    cs = ccw if ccw <= 180.0 else ccw - 360.0
                    gaps.append({'center': cs, 'center_cw': ccw, 'width': gw,
                                 'passable': gw >= GAP_MIN_PASS, 'delta_deg': dg,
                                 'd_L': dL, 'd_R': dR, 'depth': dp})
            i = j
        else:
            i += 1
    return gaps


def _best_gap(gaps):
    if not gaps:
        return None
    pool = [g for g in gaps if g['passable']] or gaps
    return max(pool, key=lambda g: g['width'] * 0.3
                                   - abs(g['center']) * 1.6
                                   + min(g['depth'], DETECT) / DETECT * 25.0)


def _compute_vfh(hist, has_pt):
    """VFH 분석 → (action, steer, speed, rot_dir, emg_near, front_near)."""
    emg   = _nearest(hist, has_pt, 0.0, arc_half=80)
    front = _nearest(hist, has_pt, 0.0, arc_half=35)

    if not any(has_pt):
        return 'FWD', 0.0, 0.70, 1.0, emg, front

    # 감지 범위(DETECT) 내 장애물 없음 → 직진
    if not any(has_pt[i] and hist[i] <= DETECT for i in range(N_BINS)):
        return 'FWD', 0.0, 0.45, 1.0, emg, front

    gaps = _find_gaps(hist, has_pt)
    best = _best_gap(gaps)

    # 전방이 막히고 탈출 불가 → 후진
    if emg <= EMERGENCY and (best is None or not best['passable']
                              or abs(best['center']) > ROT_THRESH):
        return 'BACK', 0.0, 0.80, 1.0, emg, front

    # 통과 가능한 갭, 정면 방향 → 전진
    if best is not None and best['passable'] and abs(best['center']) <= ROT_THRESH:
        imb  = (best['d_R'] - best['d_L']) / (best['d_L'] + best['d_R'] + 1e-9)
        bias = imb * (best['delta_deg'] / 2.9)
        # 좌우 벽 반발력 (± 조향 보정)
        WR  = 150.0
        lL  = _nearest(hist, has_pt, 270.0, arc_half=45)
        lR  = _nearest(hist, has_pt,  90.0, arc_half=45)
        rep = (max(0.0, WR - lL) / WR - max(0.0, WR - lR) / WR) * 20.0
        # 전방 코너 반발력
        CR  = 350.0
        cL  = _nearest(hist, has_pt, 320.0, arc_half=25)
        cR  = _nearest(hist, has_pt,  40.0, arc_half=25)
        crn = (max(0.0, CR - cL) / CR - max(0.0, CR - cR) / CR) * 45.0
        tgt = best['center'] + bias + rep + crn
        nd  = _nearest(hist, has_pt, best['center_cw'], arc_half=35)
        rt  = min(max((VELO_DOWN - nd) / (VELO_DOWN - EMERGENCY), 0.0), 1.0)
        st  = max(-LID_MAX_STEER, min(LID_MAX_STEER,
                  tgt * (1.0 + rt * 0.5) / 90.0 * LID_MAX_STEER))
        spd = 0.85 * (1.0 - rt * 0.55)
        return 'FWD', float(st), float(spd), 1.0, emg, front

    # 통과 가능한 갭이지만 너무 가까움 → 제자리 피벗
    if best is not None and best['passable'] and emg <= P4_DIST:
        rd = 1.0 if best['center'] > 0 else -1.0
        return 'ROT', 0.0, 0.0, rd, emg, front

    # 통과 불가 갭 → 전방 ±60° 내 최선 방향으로 저속 전진
    FARC = 60.0
    if gaps:
        fg = [g for g in gaps if abs(g['center']) <= FARC]
        td = max(fg, key=lambda g: g['width'])['center'] if fg \
             else max(-FARC, min(FARC, max(gaps, key=lambda g: g['width'])['center']))
    else:
        td = 0.0
    st = max(-LID_MAX_STEER, min(LID_MAX_STEER, td / 90.0 * LID_MAX_STEER * 0.5))
    return 'FWD', float(st), 0.40, 1.0, emg, front


def _vfh_drive(ser, hist, has_pt):
    """VFH 결과로 Arduino 명령 전송. 로그 문자열 반환."""
    act, st, spd, rd, emg, _ = _compute_vfh(hist, has_pt)
    if act == 'BACK':
        ser.write(b"B 0.80\n")
        return f"BACK emg={emg:.0f}mm"
    if act == 'ROT':
        ser.write(f"T {rd:.2f}\n".encode())
        return f"ROT dir={rd:+.0f}"
    ser.write(f"F {st:.2f} {spd:.2f}\n".encode())
    return f"FWD st={st:+.2f} spd={spd:.2f}"


def _speed_limit(cam_speed, hist, has_pt):
    """전방 LiDAR 거리에 따라 카메라 속도 상한 제한."""
    front = _nearest(hist, has_pt, 0.0, arc_half=35)
    if front <= EMERGENCY:
        return 0.0
    if front < VELO_DOWN:
        rt = max(0.0, min(1.0, (VELO_DOWN - front) / (VELO_DOWN - EMERGENCY)))
        return cam_speed * (1.0 - rt * 0.45)
    return cam_speed


# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR 백그라운드 스레드
# ─────────────────────────────────────────────────────────────────────────────

class LidarThread(threading.Thread):
    def __init__(self, port: str):
        super().__init__(daemon=True)
        self._port   = port
        self._lock   = threading.Lock()
        self._hist   = [9999.0] * N_BINS
        self._has_pt = [False]  * N_BINS
        self._ready  = False
        self._ser    = None

    def run(self):
        try:
            self._ser = serial.Serial(self._port, 460800, timeout=1)
            self._ser.write(bytes([0xA5, 0x40]))
            time.sleep(0.3)
            self._ser.write(bytes([0xA5, 0x20]))
            scan_buf = []
            while True:
                data = self._ser.read(5)
                if len(data) != 5:
                    continue
                s_flag = data[0] & 0x01
                s_inv  = (data[0] & 0x02) >> 1
                if s_inv != (1 - s_flag):
                    continue
                if (data[1] & 0x01) != 1:
                    continue
                quality  = data[0] >> 2
                angle    = ((data[1] >> 1) | (data[2] << 7)) / 64.0
                distance = (data[3] | (data[4] << 8)) / 4.0
                if quality == 0 or distance < 80:
                    continue
                scan_buf.append((angle, distance))
                if s_flag == 1 and scan_buf:
                    h, hp = _build_hist(scan_buf)
                    with self._lock:
                        self._hist   = h
                        self._has_pt = hp
                        self._ready  = True
                    scan_buf = []
        except Exception as e:
            print(f"[LIDAR ERROR] {e}")

    def get_state(self):
        with self._lock:
            return list(self._hist), list(self._has_pt), self._ready

    def stop(self):
        try:
            if self._ser and self._ser.is_open:
                self._ser.write(bytes([0xA5, 0x25]))
                time.sleep(0.1)
                self._ser.close()
        except Exception:
            pass


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
    if not (np.isfinite(z_mm) and np.isfinite(x_mm) and z_mm > 0):
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


def get_weak_contour(hsv, color: str):
    mask = _get_mask(hsv, color)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > WEAK_MIN_AREA]
    return max(cnts, key=cv2.contourArea) if cnts else None


def _contour_offset(cnt, frame_w: int) -> float:
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return 0.0
    return (M['m10'] / M['m00'] - frame_w / 2) / (frame_w / 2)


def _draw_center(vis, cx: int, cy: int, color):
    cv2.circle(vis, (cx, cy), 6, color, -1)
    cv2.line(vis, (cx - 15, cy), (cx + 15, cy), color, 1)
    cv2.line(vis, (cx, cy - 15), (cx, cy + 15), color, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ser   = serial.Serial(PORT_ARDU, 460800, timeout=1)
    lidar = LidarThread(PORT_LIDAR)
    lidar.start()

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
            lidar.stop()
            cap.release(); cv2.destroyAllWindows(); ser.close()
        except Exception:
            pass
    atexit.register(_cleanup)

    def _sig(_s, _f):
        _cleanup(); sys.exit(0)
    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGTSTP, _sig)

    target_idx     = 0
    state          = 'SEEK'
    on_zone_count  = 0
    stop_start     = None
    last_seen      = time.time()
    last_steer     = 0.0
    area_peak_seen = False
    peak_area_r    = 0.0

    print("=" * 65)
    print("  Camera_v3  |  color_v2 최신 색상 + VFH LiDAR 통합")
    print(f"  목표: RED → YELLOW → BLUE   색지:{PAPER_W_MM:.0f}×{PAPER_H_MM:.0f}mm")
    print(f"  PnP: {'ON' if pnp_mat is not None else 'OFF(fallback)'}")
    print("=" * 65)

    while True:
        ret, raw = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # ── DONE ──────────────────────────────────────────────────────────
        if state == 'DONE':
            ser.write(b"S\n")
            vis = raw.copy()
            cv2.putText(vis, "MISSION COMPLETE", (fw // 2 - 120, fh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.imshow('Robot View', vis)
            cv2.waitKey(1)
            time.sleep(0.1)
            continue

        # ── LiDAR 긴급 후진 (SEEK 중 최우선) ─────────────────────────────
        hist, has_pt, lidar_ready = lidar.get_state()
        if state == 'SEEK' and lidar_ready:
            emg_d = _nearest(hist, has_pt, 0.0, arc_half=80)
            if emg_d <= EMERGENCY:
                ser.write(b"B 0.80\n")
                vis = raw.copy()
                cv2.putText(vis, f"EMG BACK {emg_d:.0f}mm",
                            (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.imshow('Robot View', vis)
                cv2.waitKey(1)
                print(f"  [EMG] 전방={emg_d:.0f}mm → 후진")
                continue

        result = detector.detect(raw)
        color  = TARGETS[target_idx]
        vis    = detector.draw_debug(raw, result)

        # ── STOP (정지 대기) ───────────────────────────────────────────────
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
                    target_idx    += 1
                    state          = 'SEEK'
                    on_zone_count  = 0
                    area_peak_seen = False
                    peak_area_r    = 0.0
                    last_seen      = time.time()
                    print(f"  ✅ {color.upper()} 완료 → {TARGETS[target_idx].upper()}")
                else:
                    state = 'DONE'
                    print("  ✅ 전체 미션 완료!")
            continue

        # ── SEEK ──────────────────────────────────────────────────────────
        det = result.get(color, {})

        # ① 강탐지 ─────────────────────────────────────────────────────────
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
                cam_spd = SPEED_NEAR if z_mm < DIST_SLOW_MM else SPEED_FAR
                speed   = _speed_limit(cam_spd, hist, has_pt) if lidar_ready else cam_spd
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
                cam_spd = SPEED_NEAR if area_peak_seen else SPEED_FAR
                speed   = _speed_limit(cam_spd, hist, has_pt) if lidar_ready else cam_spd
                log_msg = f"fallback offset={offset:+.2f} area={area_r:.2f}"
                cv2.putText(vis, f"A={area_r:.3f} pk={peak_area_r:.3f}",
                            (fw // 2 - 80, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
                M_c = cv2.moments(cnt)
                if M_c['m00'] > 0:
                    _draw_center(vis, int(M_c['m10'] / M_c['m00']),
                                 int(M_c['m01'] / M_c['m00']), (200, 200, 200))

            last_steer = steer
            if speed > 0:
                ser.write(f"F {steer:.2f} {speed:.2f}\n".encode())
            else:
                ser.write(b"S\n")
            print(f"  [SEEK] {color.upper()} {log_msg} steer={steer:+.2f} spd={speed:.2f}")

        # ② 피크 후 미탐지 (종이 위 진입 중) ───────────────────────────────
        elif area_peak_seen:
            hsv_u    = cv2.cvtColor(result['undistorted'], cv2.COLOR_BGR2HSV)
            weak_cnt = get_weak_contour(hsv_u, color)

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
                    _draw_center(vis, int(M_w['m10'] / M_w['m00']),
                                 int(M_w['m01'] / M_w['m00']), (180, 255, 0))
                cv2.putText(vis, f"ENTERING {color.upper()} off={weak_offset:+.2f}",
                            (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 0), 2)
                print(f"  [ENTER] {color.upper()} off={weak_offset:+.2f} steer={steer:+.2f}")
            else:
                on_zone_count += 1
                ser.write(b"S\n")
                cv2.putText(vis, f"ON PAPER  cnt:{on_zone_count}/{CONFIRM_FRAMES}",
                            (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                print(f"  [ON] {color.upper()} pk={peak_area_r:.2f} cnt={on_zone_count}")
                if on_zone_count >= CONFIRM_FRAMES:
                    state = 'STOP'; stop_start = time.time()
                    ser.write(b"S\n")
                    print(f"  🎯 {color.upper()} 도달!")
                    cv2.imshow('Robot View', vis); cv2.waitKey(1)
                    continue

        # ③ 미탐지 → VFH 탐색 ─────────────────────────────────────────────
        else:
            on_zone_count = max(0, on_zone_count - 1)
            hsv_u    = cv2.cvtColor(result['undistorted'], cv2.COLOR_BGR2HSV)
            weak_cnt = get_weak_contour(hsv_u, color)

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
                    _draw_center(vis, int(M_w['m10'] / M_w['m00']),
                                 int(M_w['m01'] / M_w['m00']), (180, 180, 0))
                cv2.putText(vis, f"WEAK {color.upper()} off={weak_offset:+.2f}",
                            (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 0), 2)
                print(f"  [WEAK] {color.upper()} off={weak_offset:+.2f} steer={steer:+.2f}")
            else:
                elapsed = time.time() - last_seen
                if elapsed < SEARCH_TIMEOUT:
                    ser.write(b"S\n")
                    cv2.putText(vis, f"WAIT {elapsed:.1f}s",
                                (5, 58), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (100, 100, 255), 2)
                else:
                    if lidar_ready:
                        log = _vfh_drive(ser, hist, has_pt)
                    else:
                        ser.write(b"S\n")
                        log = "NO_LIDAR"
                    cv2.putText(vis, f"VFH {log}",
                                (5, 58), cv2.FONT_HERSHEY_SIMPLEX,
                                0.50, (100, 200, 255), 2)
                    print(f"  [VFH] {color.upper()} {elapsed:.1f}s → {log}")

        # ── 공통 HUD ──────────────────────────────────────────────────────
        emg_txt = f" EMG:{_nearest(hist, has_pt, 0.0, 80):.0f}mm" if lidar_ready else ""
        cv2.putText(vis,
                    f"{state}|{color.upper()}|cnt:{on_zone_count}/{CONFIRM_FRAMES}{emg_txt}",
                    (5, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 2)
        cv2.imshow('Robot View', vis)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            print("  [종료] q 입력")
            break

    ser.write(b"S\n")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
