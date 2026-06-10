"""
[파일] Camera_v3.py
[목적] 카메라 색지 추적(RED → YELLOW → BLUE) + LiDAR VFH 장애물 회피 통합 최종본

[동작 우선순위]
  1. EMERGENCY  : 전방 장애물 < EMERGENCY mm → 후진 override (모든 상태 무시)
  2. STOP/DONE  : 색지 도달 정지 / 미션 완료
  3. CAM-LOCK   : 4코너 클린 PnP 잠금 완료 → 잠긴 방향 전진 + LiDAR 안전 보정
  4. CAM-SEEK   : 색지 강탐지 중 → PnP 접근 + LiDAR 안전 보정
  5. ENTERING   : 피크 후 색지 소멸 진입 중 → 잠긴 조향 또는 약탐지 유도
  6. ON_PAPER   : 색지 완전 소멸(바퀴 위) → CONFIRM_FRAMES 확인 후 정지
  7. WEAK       : 약탐지 신호 → 색지 방향 유도 + LiDAR 안전 보정
  8. VFH        : 색지 미탐지 → LiDAR VFH 장애물 회피 탐색

[잠금(LOCK) 로직]
  - 클린 4코너(approxPolyDP 자연 4점) + area >= LOCK_MIN_AREA 연속 LOCK_AVG_FRAMES 프레임
  - N프레임 평균 steer → locked_steer 확정
  - 이후 색지가 흔들리거나 부분 가려도 잠긴 방향 유지
  - 클린 PnP 재확인 시 locked_steer 갱신(재잠금)

[LiDAR 안전 보정]
  - 전방 장애물 거리 기반 감속
  - 측면 벽 반발력 → steer 보정
  - EMERGENCY 미만: 후진 override

[아두이노 명령 프로토콜]
  F {steer:.2f} {speed:.2f}\n  전진 (steer: -1.0 좌 ~ +1.0 우)
  B {speed:.2f}\n              후진
  T {dir:.2f}\n                제자리 피벗 (+우 / -좌)
  S\n                          즉시 정지

[현장 조정 필수값]
  PAPER_W_MM / PAPER_H_MM  : 실제 색지 크기 (mm)
  WHEEL_AXLE_DIST_MM        : 카메라 ~ 앞 바퀴 축 거리 (mm)
"""

import os
import atexit
import signal
import sys
import time
import threading
import math

import serial
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
#  현장 측정값 ← 반드시 직접 측정 후 수정!
# ─────────────────────────────────────────────────────────────────────────────
PAPER_W_MM         = 300.0
PAPER_H_MM         = 300.0
WHEEL_AXLE_DIST_MM = 60.0

# ─────────────────────────────────────────────────────────────────────────────
#  카메라 주행 파라미터
# ─────────────────────────────────────────────────────────────────────────────
MAX_STEER_CAM   = 1.0
SPEED_FAR       = 0.55
SPEED_NEAR      = 0.35
SPEED_CRAWL     = 0.22
DIST_SLOW_MM    = 100.0
ALIGN_THRES_MM  = 50.0
AREA_PEAK_THRES = 0.18
STEER_GAIN      = 0.015
CONFIRM_FRAMES  = 4
STOP_DURATION   = 1.0

LOCK_MIN_AREA   = 0.12
LOCK_AVG_FRAMES = 5

# ─────────────────────────────────────────────────────────────────────────────
#  약탐지 파라미터
# ─────────────────────────────────────────────────────────────────────────────
WEAK_MIN_AREA   = 200
WEAK_SPEED      = 0.35
WEAK_STEER_GAIN = 0.60

# ─────────────────────────────────────────────────────────────────────────────
#  BLUE 바닥 매트 ↔ 수직 벽 구분
# ─────────────────────────────────────────────────────────────────────────────
BLUE_ASPECT_MIN = 0.45
BLUE_BOTTOM_MIN = 0.35

TARGETS = ['red', 'yellow', 'blue']

# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR / VFH 파라미터
# ─────────────────────────────────────────────────────────────────────────────
BIN_DEG       = 4.0
N_BINS        = int(360 / BIN_DEG)
GAP_MIN       = 80.0
GAP_MARGIN    = 10.0
GAP_MIN_PASS  = GAP_MIN + GAP_MARGIN
DETECT        = 560.0
VELO_DOWN     = 400.0
EMERGENCY     = 150.0
P4_DIST       = 170.0
MAX_STEER_VFH = 1.2
ROT_THRESH    = 110.0
ROBOT_RADIUS  = 35.0

LIDAR_WALL_REP   = 150.0
LIDAR_CORNER_REP = 350.0


# ─────────────────────────────────────────────────────────────────────────────
#  VFH 유틸리티 함수 (Li_final.py 기반)
# ─────────────────────────────────────────────────────────────────────────────

def build_polar_hist(scan_buf):
    hist   = [9999.0] * N_BINS
    has_pt = [False]  * N_BINS
    for a, d in scan_buf:
        idx = int(a / BIN_DEG) % N_BINS
        if d < hist[idx]:
            hist[idx] = d
            has_pt[idx] = True
    return hist, has_pt


def find_vfh_gaps(hist, has_pt, detect_dist, min_pass_mm):
    blocked = [has_pt[i] and hist[i] <= detect_dist for i in range(N_BINS)]

    smoothed = blocked[:]
    for i in range(N_BINS):
        if blocked[i] and not blocked[(i-1) % N_BINS] and not blocked[(i+1) % N_BINS]:
            smoothed[i] = False
    blocked = smoothed

    inflated = blocked[:]
    for i in range(N_BINS):
        if blocked[i] and hist[i] < 9999.0:
            alpha_rad  = math.asin(min(1.0, ROBOT_RADIUS / max(hist[i], ROBOT_RADIUS)))
            alpha_bins = int(math.degrees(alpha_rad) / BIN_DEG) + 1
            for k in range(-alpha_bins, alpha_bins + 1):
                inflated[(i + k) % N_BINS] = True
    blocked = inflated

    gaps = []
    seen = set()
    i = 0
    while i < 2 * N_BINS:
        bi = i % N_BINS
        if not blocked[bi]:
            j = i + 1
            while j < i + N_BINS and not blocked[j % N_BINS]:
                j += 1
            span = j - i
            if span < N_BINS:
                center_cw = ((i + j) / 2.0 * BIN_DEG) % 360.0
                ck = round(center_cw)
                if ck not in seen:
                    seen.add(ck)
                    delta_deg = span * BIN_DEG
                    d_L = hist[(i-1) % N_BINS] if has_pt[(i-1) % N_BINS] else detect_dist
                    d_R = hist[j % N_BINS]     if has_pt[j % N_BINS]     else detect_dist
                    d_L = min(d_L, detect_dist)
                    d_R = min(d_R, detect_dist)
                    gap_w = (d_L + d_R) * math.sin(math.radians(delta_deg / 2.0))
                    depth = min(hist[k % N_BINS] for k in range(i, j))
                    center_s = center_cw if center_cw <= 180.0 else center_cw - 360.0
                    gaps.append({
                        'center': center_s, 'center_cw': center_cw,
                        'width': gap_w, 'passable': gap_w >= min_pass_mm,
                        'delta_deg': delta_deg, 'd_L': d_L, 'd_R': d_R, 'depth': depth,
                    })
            i = j
        else:
            i += 1
    return gaps


def select_best_gap(gaps, min_pass_mm=GAP_MIN_PASS):
    if not gaps:
        return None
    passable = [g for g in gaps if g['width'] >= min_pass_mm]
    pool = passable if passable else gaps
    def score(g):
        depth_norm = min(g['depth'], DETECT) / DETECT
        return g['width'] * 0.3 - abs(g['center']) * 1.6 + depth_norm * 25.0
    return max(pool, key=score)


def nearest_in_arc(hist, has_pt, center_cw, arc_half=25):
    center_bin = int(center_cw / BIN_DEG) % N_BINS
    n_check    = max(1, int(arc_half / BIN_DEG))
    min_d = 9999.0
    for k in range(-n_check, n_check + 1):
        idx = (center_bin + k) % N_BINS
        if has_pt[idx] and hist[idx] < min_d:
            min_d = hist[idx]
    return min_d


# ─────────────────────────────────────────────────────────────────────────────
#  카메라 유틸리티 함수 (Camera_v2.py 기반)
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


def _extract_quad(contour: np.ndarray):
    """
    컨투어 → (4꼭짓점, is_clean).
    is_clean=True: approxPolyDP 자연 4점 (PnP 신뢰 가능).
    is_clean=False: 병합·폴백 사용 (PnP 신뢰 낮음).
    """
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    best = hull.reshape(-1, 2).astype(np.float32)
    for eps_ratio in np.arange(0.01, 0.40, 0.01):
        approx = cv2.approxPolyDP(hull, float(eps_ratio) * peri, True)
        pts = approx.reshape(-1, 2).astype(np.float32)
        if len(pts) == 4:
            return _order_points(pts), True
        if len(pts) < 4:
            break
        best = pts
    while len(best) > 4:
        dists = [np.linalg.norm(best[i] - best[(i+1) % len(best)])
                 for i in range(len(best))]
        idx = int(np.argmin(dists))
        nxt = (idx + 1) % len(best)
        best[idx] = (best[idx] + best[nxt]) / 2
        best = np.delete(best, nxt, axis=0)
    if len(best) < 4:
        best = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    return _order_points(best), False


def solve_paper_pose(contour, cam_mat, dist_coeffs):
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
    steer = float(np.clip(angle * STEER_GAIN, -MAX_STEER_CAM, MAX_STEER_CAM))
    return z_mm, x_mm, steer, quad_pts, is_clean


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


# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR 백그라운드 스레드
# ─────────────────────────────────────────────────────────────────────────────

class LidarThread(threading.Thread):
    """
    LiDAR 5바이트 패킷을 연속 수신, 1회전 완료 시 VFH 계산.
    get_state()로 메인 루프에 최신 결과 제공 (스레드 안전).
    """

    def __init__(self, ser: serial.Serial):
        super().__init__(daemon=True)
        self.ser       = ser
        self._hist     = [9999.0] * N_BINS
        self._has_pt   = [False]  * N_BINS
        self._gaps     = []
        self._best     = None
        self._emg_near = 9999.0
        self._fwd_near = 9999.0
        self._ready    = False
        self._lock     = threading.Lock()

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready

    def get_state(self) -> dict:
        with self._lock:
            return {
                'hist':     self._hist[:],
                'has_pt':   self._has_pt[:],
                'gaps':     list(self._gaps),
                'best':     self._best,
                'emg_near': self._emg_near,
                'fwd_near': self._fwd_near,
                'ready':    self._ready,
            }

    def run(self):
        scan_buf = []
        while True:
            try:
                data = self.ser.read(5)
            except Exception:
                break
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

            if distance < 80 or quality == 0:
                continue

            scan_buf.append((angle, distance))

            if s_flag == 1 and scan_buf:
                try:
                    hist, has_pt = build_polar_hist(scan_buf)
                    emg_near = nearest_in_arc(hist, has_pt, 0.0, arc_half=80)
                    fwd_near = nearest_in_arc(hist, has_pt, 0.0, arc_half=30)
                    gaps     = find_vfh_gaps(hist, has_pt, DETECT, GAP_MIN_PASS)
                    best     = select_best_gap(gaps, GAP_MIN_PASS)
                    with self._lock:
                        self._hist     = hist
                        self._has_pt   = has_pt
                        self._gaps     = gaps
                        self._best     = best
                        self._emg_near = emg_near
                        self._fwd_near = fwd_near
                        self._ready    = True
                except Exception:
                    pass
                scan_buf = []


# ─────────────────────────────────────────────────────────────────────────────
#  통합 헬퍼 함수
# ─────────────────────────────────────────────────────────────────────────────

def apply_lidar_safety(cam_steer: float, cam_speed: float, ls: dict):
    """
    카메라 조향·속도에 LiDAR 안전 보정 적용.
    Returns: (steer, speed, is_emergency)
    """
    if ls['emg_near'] <= EMERGENCY:
        return 0.0, 0.0, True

    fwd_near = ls['fwd_near']
    if fwd_near > VELO_DOWN:
        return cam_steer, cam_speed, False

    hist, has_pt = ls['hist'], ls['has_pt']
    ratio = min(max((VELO_DOWN - fwd_near) / (VELO_DOWN - EMERGENCY), 0.0), 1.0)
    speed = max(cam_speed * (1.0 - ratio * 0.45), 0.15)

    lat_L = nearest_in_arc(hist, has_pt, 270.0, 45)
    lat_R = nearest_in_arc(hist, has_pt,  90.0, 45)
    rep_L = max(0.0, LIDAR_WALL_REP - lat_L) / LIDAR_WALL_REP
    rep_R = max(0.0, LIDAR_WALL_REP - lat_R) / LIDAR_WALL_REP
    repulsion_steer = (rep_L - rep_R) * 20.0 / 90.0 * MAX_STEER_CAM

    steer = float(np.clip(cam_steer + repulsion_steer, -MAX_STEER_CAM, MAX_STEER_CAM))
    return steer, speed, False


def compute_vfh_cmd(ls: dict):
    """
    LiDAR 상태로 VFH 주행 명령 결정 (색지 탐색 중).
    Returns: (cmd_bytes, log_str)
    """
    hist, has_pt = ls['hist'], ls['has_pt']
    gaps     = ls['gaps']
    best     = ls['best']
    emg_near = ls['emg_near']

    if not any(has_pt):
        return b"F 0.00 0.60\n", "OPEN(장애물없음)"

    # P1: VFH 전진 — 갭이 전방 반구(±ROT_THRESH) 내
    if best is not None and best['passable'] and abs(best['center']) <= ROT_THRESH:
        d_L, d_R  = best['d_L'], best['d_R']
        imbalance = (d_R - d_L) / (d_L + d_R + 1e-9)
        bias      = imbalance * (best['delta_deg'] / 2.9)

        lat_L = nearest_in_arc(hist, has_pt, 270.0, 45)
        lat_R = nearest_in_arc(hist, has_pt,  90.0, 45)
        rep_L = max(0.0, 150.0 - lat_L) / 150.0
        rep_R = max(0.0, 150.0 - lat_R) / 150.0
        repulsion = (rep_L - rep_R) * 20.0

        crn_L  = nearest_in_arc(hist, has_pt, 320.0, 25)
        crn_R  = nearest_in_arc(hist, has_pt,  40.0, 25)
        crnf_L = max(0.0, LIDAR_CORNER_REP - crn_L) / LIDAR_CORNER_REP
        crnf_R = max(0.0, LIDAR_CORNER_REP - crn_R) / LIDAR_CORNER_REP
        corner_rep = (crnf_L - crnf_R) * 45.0

        PULL_PEAK, PULL_RANGE = 300.0, 150.0
        pull_L    = max(0.0, 1.0 - abs(lat_L - PULL_PEAK) / PULL_RANGE)
        pull_R    = max(0.0, 1.0 - abs(lat_R - PULL_PEAK) / PULL_RANGE)
        side_pull = (pull_R - pull_L) * 10.0

        target     = best['center'] + bias + repulsion + corner_rep + side_pull
        near_d     = nearest_in_arc(hist, has_pt, best['center_cw'], arc_half=35)
        ratio      = min(max((VELO_DOWN - near_d) / (VELO_DOWN - EMERGENCY), 0.0), 1.0)
        steer_gain = 1.0 + ratio * 0.5
        steer = float(np.clip(target * steer_gain / 90.0 * MAX_STEER_VFH,
                               -MAX_STEER_VFH, MAX_STEER_VFH))
        speed = 0.85 * (1.0 - ratio * 0.55)
        cmd = f"F {steer:.2f} {speed:.2f}\n".encode()
        log = (f"FWD 갭={best['width']:.0f}mm@{best['center']:+.0f}도 "
               f"근접={near_d:.0f}mm steer={steer:+.2f} spd={speed:.2f}")
        return cmd, log

    # P2: 제자리 회전 — 갭이 후방 + 장애물 근접
    if best is not None and best['passable'] and emg_near <= P4_DIST:
        rot_dir = 1.0 if best['center'] > 0 else -1.0
        cmd = f"T {rot_dir:.2f}\n".encode()
        log = f"ROT 후방갭({best['center']:+.0f}도) 근접={emg_near:.0f}mm dir={rot_dir:+.0f}"
        return cmd, log

    # P3: 긴급 후진 — 전방 막힘 + 통과 가능 갭 없음
    if emg_near <= EMERGENCY and (best is None or not best['passable']
                                   or abs(best['center']) > ROT_THRESH):
        cmd = b"B 0.80\n"
        log = f"BACK 긴급={emg_near:.0f}mm"
        return cmd, log

    # P4: 통과 가능 갭 없음 → 가장 넓은 방향으로 저속 전진
    FRONT_ARC = 60.0
    if gaps:
        front  = [g for g in gaps if abs(g['center']) <= FRONT_ARC]
        open_g = max(front if front else gaps, key=lambda g: g['width'])
        tdir   = (open_g['center'] if front
                  else max(-FRONT_ARC, min(FRONT_ARC, open_g['center'])))
        widest = open_g['width']
    else:
        tdir, widest = 0.0, 0.0
    steer = float(np.clip(tdir / 90.0 * MAX_STEER_VFH * 0.5, -MAX_STEER_VFH, MAX_STEER_VFH))
    cmd = f"F {steer:.2f} 0.40\n".encode()
    log = f"SLOW 최대폭={widest:.0f}mm target={tdir:+.0f}도 steer={steer:+.2f}"
    return cmd, log


# ─────────────────────────────────────────────────────────────────────────────
#  메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ser   = serial.Serial(PORT_ARDU,  460800, timeout=1)
    ser_L = serial.Serial(PORT_LIDAR, 460800, timeout=1)

    # LiDAR 초기화
    ser_L.write(bytes([0xA5, 0x40]))  # 모터 시작
    time.sleep(1)
    ser_L.write(bytes([0xA5, 0x20]))  # 스캔 시작

    lidar = LidarThread(ser_L)
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
            ser.write(b"S\n")
            ser_L.write(bytes([0xA5, 0x25]))  # LiDAR 스캔 중지
            time.sleep(0.1)
            cap.release()
            cv2.destroyAllWindows()
            ser.close()
            ser_L.close()
        except Exception:
            pass
    atexit.register(_cleanup)

    def _sig_handler(_sig, _frame):
        _cleanup()
        sys.exit(0)
    signal.signal(signal.SIGINT,  _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGTSTP, _sig_handler)

    # ── 상태 변수 ────────────────────────────────────────────────────────────
    target_idx      = 0
    state           = 'SEEK'
    on_zone_count   = 0
    stop_start      = None
    last_steer      = 0.0
    area_peak_seen  = False
    peak_area_r     = 0.0
    lock_state      = False
    locked_steer    = 0.0
    clean_steer_buf = []

    def _reset_for_next():
        nonlocal on_zone_count, area_peak_seen, peak_area_r
        nonlocal lock_state, locked_steer, clean_steer_buf
        on_zone_count   = 0
        area_peak_seen  = False
        peak_area_r     = 0.0
        lock_state      = False
        locked_steer    = 0.0
        clean_steer_buf = []

    print("=" * 65)
    print("  Camera_v3  |  색지 추적 + LiDAR VFH 장애물 회피")
    print(f"  목표: RED → YELLOW → BLUE")
    print(f"  색지 {PAPER_W_MM:.0f}×{PAPER_H_MM:.0f}mm  바퀴축 {WHEEL_AXLE_DIST_MM:.0f}mm")
    print(f"  LiDAR: 감지={DETECT:.0f}mm  긴급={EMERGENCY:.0f}mm  최소통로={GAP_MIN_PASS:.0f}mm")
    print(f"  PnP: {'ON' if pnp_mat is not None else 'OFF(fallback)'}  "
          f"잠금: {LOCK_AVG_FRAMES}프레임 평균")
    print("=" * 65)

    # ── 메인 루프 ─────────────────────────────────────────────────────────────
    while True:
        ret, raw = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        ls = lidar.get_state()

        # ── DONE ─────────────────────────────────────────────────────────────
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

        # ── STOP (1초 정지 대기) ──────────────────────────────────────────────
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
                    target_idx += 1
                    state       = 'SEEK'
                    _reset_for_next()
                    print(f"  ✅ {color.upper()} 완료 → {TARGETS[target_idx].upper()}")
                else:
                    state = 'DONE'
                    print("  ✅ 전체 미션 완료!")
            continue

        # ── 긴급 장애물 (LiDAR 최우선 override) ──────────────────────────────
        if ls['ready'] and ls['emg_near'] <= EMERGENCY:
            ser.write(b"B 0.80\n")
            cv2.putText(vis, f"!! EMERGENCY BACK {ls['emg_near']:.0f}mm !!",
                        (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.imshow('Robot View', vis)
            cv2.waitKey(1)
            print(f"  [EMG] 전방 {ls['emg_near']:.0f}mm → 긴급 후진")
            continue

        # ── SEEK ─────────────────────────────────────────────────────────────
        weak_cnt = None  # 매 이터레이션 초기화 (HUD 잔류값 방지)
        det = result.get(color, {})
        if color == 'blue' and det.get('found'):
            if not is_floor_contour(det['contour'], fh):
                det = {'found': False}

        # ①  강탐지: PnP + 잠금 ───────────────────────────────────────────────
        if det.get('found'):
            cnt    = det['contour']
            pose   = solve_paper_pose(cnt, pnp_mat, pnp_dist)
            area_r = det['area'] / (fw * fh)

            if area_r > AREA_PEAK_THRES:
                area_peak_seen = True
                peak_area_r    = max(peak_area_r, area_r)

            if pose is not None:
                z_mm, x_mm, steer, quad_pts, is_clean = pose

                # 클린 PnP → 잠금 버퍼 누적 및 갱신
                if is_clean and area_r >= LOCK_MIN_AREA:
                    clean_steer_buf.append(steer)
                    if len(clean_steer_buf) > LOCK_AVG_FRAMES:
                        clean_steer_buf.pop(0)
                    if len(clean_steer_buf) >= LOCK_AVG_FRAMES:
                        locked_steer = sum(clean_steer_buf) / len(clean_steer_buf)
                        lock_state   = True
                elif not is_clean and not lock_state:
                    clean_steer_buf.clear()

                steer_cmd  = locked_steer if lock_state else steer
                base_speed = (SPEED_CRAWL if lock_state
                              else SPEED_NEAR if z_mm < DIST_SLOW_MM
                              else SPEED_FAR)

                if ls['ready']:
                    steer_cmd, base_speed, _ = apply_lidar_safety(steer_cmd, base_speed, ls)

                pnp_reached = (lock_state and is_clean
                               and z_mm < WHEEL_AXLE_DIST_MM
                               and abs(x_mm) < ALIGN_THRES_MM)

                buf_lbl = "LOCK✓" if lock_state else f"buf:{len(clean_steer_buf)}/{LOCK_AVG_FRAMES}"
                pnp_col = ((0, 255, 0)   if pnp_reached
                           else (0, 200, 255) if lock_state
                           else (180, 180, 0) if not is_clean
                           else (0, 220, 255))
                log_msg = (f"CAM z={z_mm:.0f}mm x={x_mm:+.0f}mm "
                           f"{'CLN' if is_clean else 'EST'} {buf_lbl}")
                cv2.putText(vis, f"Z={z_mm:.0f} X={x_mm:+.0f} A={area_r:.3f} {buf_lbl}",
                            (fw // 2 - 170, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, pnp_col, 2)
                cv2.polylines(vis, [quad_pts.astype(np.int32)], True, pnp_col, 2)

            else:
                # PnP 실패 → 4코너 보이도록 컨투어 중심 방향 조향
                offset     = det['offset']
                steer_cmd  = (locked_steer if lock_state
                              else float(np.clip(offset * 0.80, -MAX_STEER_CAM, MAX_STEER_CAM)))
                base_speed = (SPEED_CRAWL if lock_state
                              else SPEED_NEAR if area_peak_seen
                              else SPEED_FAR)
                if ls['ready']:
                    steer_cmd, base_speed, _ = apply_lidar_safety(steer_cmd, base_speed, ls)
                pnp_reached = False
                log_msg = (f"CAM {'LOCK-fb' if lock_state else 'seek-4crn'} "
                           f"off={offset:+.2f} A={area_r:.2f}")
                cv2.putText(vis, f"{'LOCK✓' if lock_state else 'seek-4crn'} A={area_r:.3f}",
                            (fw // 2 - 100, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

            last_steer    = steer_cmd
            on_zone_count = on_zone_count + 1 if pnp_reached else max(0, on_zone_count - 1)

            if on_zone_count >= CONFIRM_FRAMES:
                state = 'STOP'; stop_start = time.time()
                ser.write(b"S\n")
                print(f"  🎯 {color.upper()} 도달(PnP)! {log_msg}")
                cv2.imshow('Robot View', vis); cv2.waitKey(1)
                continue

            ser.write(f"F {steer_cmd:.2f} {base_speed:.2f}\n".encode())
            print(f"  [CAM] {color.upper()} {log_msg} steer={steer_cmd:+.2f} cnt={on_zone_count}")

        # ②  피크 후 미탐지: 색지가 카메라 아래로 진입 중 ─────────────────────
        elif area_peak_seen:
            hsv_u    = cv2.cvtColor(result['undistorted'], cv2.COLOR_BGR2HSV)
            weak_cnt = get_weak_contour(hsv_u, color, fh)

            if weak_cnt is not None:
                if lock_state:
                    steer     = locked_steer
                    enter_lbl = f"ENTERING(LOCK) s={steer:+.2f}"
                else:
                    weak_off  = _contour_offset(weak_cnt, fw)
                    steer     = float(np.clip(weak_off * WEAK_STEER_GAIN,
                                              -MAX_STEER_CAM, MAX_STEER_CAM))
                    enter_lbl = f"ENTERING {color.upper()} off={weak_off:+.2f}"
                enter_speed = WEAK_SPEED
                if ls['ready']:
                    steer, enter_speed, _ = apply_lidar_safety(steer, enter_speed, ls)
                last_steer = steer
                ser.write(f"F {steer:.2f} {enter_speed:.2f}\n".encode())
                cv2.drawContours(vis, [weak_cnt], -1, (180, 180, 0), 1)
                cv2.putText(vis, enter_lbl,
                            (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 0), 2)
                print(f"  [ENTER] {color.upper()} {enter_lbl}")
            else:
                # 완전히 사라짐 → 양쪽 바퀴 모두 색지 위
                on_zone_count += 1
                ser.write(b"S\n")
                log = f"invisible pk={peak_area_r:.2f} lock={'Y' if lock_state else 'N'}"
                cv2.putText(vis, f"ON PAPER  cnt:{on_zone_count}/{CONFIRM_FRAMES}",
                            (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                print(f"  [ON] {color.upper()} {log} cnt={on_zone_count}")
                if on_zone_count >= CONFIRM_FRAMES:
                    state = 'STOP'; stop_start = time.time()
                    ser.write(b"S\n")
                    print(f"  🎯 {color.upper()} 도달(invisible)! {log}")
                    cv2.imshow('Robot View', vis); cv2.waitKey(1)
                    continue

        # ③  미탐지: 약탐지 시도 → 없으면 VFH 탐색 ───────────────────────────
        else:
            on_zone_count = max(0, on_zone_count - 1)
            hsv_u    = cv2.cvtColor(result['undistorted'], cv2.COLOR_BGR2HSV)
            weak_cnt = get_weak_contour(hsv_u, color, fh)

            if weak_cnt is not None:
                # 약탐지: 색지 방향으로 유도 + LiDAR 안전 보정
                weak_off   = _contour_offset(weak_cnt, fw)
                steer_cmd  = float(np.clip(weak_off * WEAK_STEER_GAIN,
                                           -MAX_STEER_CAM, MAX_STEER_CAM))
                base_speed = WEAK_SPEED
                if ls['ready']:
                    steer_cmd, base_speed, _ = apply_lidar_safety(steer_cmd, base_speed, ls)
                last_steer = steer_cmd
                ser.write(f"F {steer_cmd:.2f} {base_speed:.2f}\n".encode())
                cv2.drawContours(vis, [weak_cnt], -1, (180, 180, 0), 1)
                cv2.putText(vis, f"WEAK {color.upper()} off={weak_off:+.2f}",
                            (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 0), 2)
                print(f"  [WEAK] {color.upper()} off={weak_off:+.2f} steer={steer_cmd:+.2f}")

            elif ls['ready']:
                # VFH 장애물 회피 탐색
                cmd, vfh_log = compute_vfh_cmd(ls)
                ser.write(cmd)
                cv2.putText(vis, f"VFH {vfh_log[:40]}",
                            (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 200, 255), 2)
                print(f"  [VFH] {color.upper()} {vfh_log}")

            else:
                # LiDAR 아직 초기화 중 → 천천히 전진
                ser.write(b"F 0.00 0.30\n")
                cv2.putText(vis, "LiDAR 초기화 중...",
                            (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 2)

        # ── 공통 HUD ─────────────────────────────────────────────────────────
        if det.get('found') or area_peak_seen:
            mode = "CAM-LOCK" if lock_state else "CAM"
        else:
            mode = "WEAK" if weak_cnt is not None else "VFH"

        emg_txt = (f" |EMG:{ls['emg_near']:.0f}mm"
                   if ls['ready'] and ls['emg_near'] < DETECT else "")
        cv2.putText(vis,
                    f"{state}|{color.upper()}|{mode}{emg_txt} cnt:{on_zone_count}/{CONFIRM_FRAMES}",
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
