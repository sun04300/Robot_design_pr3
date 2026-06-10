"""
[파일] resist.py
[목적] Camera_v2 기반 + LiDAR 코너 반발력(Corner Repulsion)
       로봇 4개 코너 방향의 장애물 거리를 측정,
       가까워질수록 반발력을 카메라 조향에 더해 충돌 방지.

[반발력 모델]
  로봇 4코너 각도 (라이다 기준 CW):
    전방-좌(FL): 320°   전방-우(FR):  40°
    후방-좌(RL): 220°   후방-우(RR): 140°

  반발력 = ((CORNER_REP_DIST - d) / CORNER_REP_DIST)²  × 가중치
    - 좌측 코너 → 우로 밀기(steer+)
    - 우측 코너 → 좌로 밀기(steer-)
    - 전방 코너 가중치 1.0 / 후방 코너 가중치 0.35

[LiDAR 안전 레이어]
  - 전방 VELO_DOWN_MM 이내 → 감속
  - 전방 EMERGENCY_MM 이내 → 즉시 정지

[정지 조건]
  Path B : area_peak_seen 후 색지 완전 소멸 (바퀴 위)

[아두이노 명령 프로토콜]
  F {steer:.2f} {speed:.2f}\n  → 전진
  T {dir:.2f}\n                → 제자리 피벗
  S\n                          → 즉시 정지
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
CAM_W      = 640
CAM_H      = 480
CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "camera_calibration.pkl")

# ─────────────────────────────────────────────────────────────────────────────
#  현장 측정값
# ─────────────────────────────────────────────────────────────────────────────
PAPER_W_MM      = 300.0
PAPER_H_MM      = 300.0

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

WEAK_MIN_AREA    = 200
WEAK_SPEED       = 0.35
WEAK_STEER_GAIN  = 0.60
SEARCH_TIMEOUT   = 1.5
SEARCH_ARC_STEER = 0.55
SEARCH_ARC_SPEED = 0.28
SEARCH_ARC_DUR   = 2.5
GAP_DETECT_MM    = 600.0   # 탐색 중 전방 이 거리 이내 장애물 감지 시 갭 이동 모드
GAP_STEER_GAIN   = 0.008   # 갭 방향 각도(deg) → 조향 변환 게인

TARGETS = ['red', 'yellow', 'blue']

# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR 파라미터
# ─────────────────────────────────────────────────────────────────────────────
BIN_DEG      = 4.0
N_BINS       = int(360 / BIN_DEG)
EMERGENCY_MM = 150.0
VELO_DOWN_MM = 400.0
ROBOT_RADIUS = 35.0
FRONT_ARC_HALF = 50      # 전방 긴급 감지 아크 반폭 (degree)

# ─────────────────────────────────────────────────────────────────────────────
#  코너 반발력 파라미터
# ─────────────────────────────────────────────────────────────────────────────
CORNER_FL_DEG   = 320.0   # 전방-좌 코너 각도 (CW, 라이다 기준)
CORNER_FR_DEG   =  40.0   # 전방-우 코너 각도
CORNER_RL_DEG   = 220.0   # 후방-좌 코너 각도
CORNER_RR_DEG   = 140.0   # 후방-우 코너 각도
CORNER_ARC_HALF =  25     # 코너 측정 아크 반폭 (degree)

CORNER_REP_DIST = 400.0   # 반발력 시작 거리 (mm). 이 이내부터 활성화
CORNER_REP_GAIN = 0.9     # 최대 반발력이 조향에 더해지는 최대 크기
CORNER_FRONT_W  = 1.0     # 전방 코너 가중치
CORNER_REAR_W   = 0.35    # 후방 코너 가중치 (주행 방향 외부라 영향 작음)

# 카메라 지지대 마스크: 후방 180° ± 60° 제외
# → 후방 코너(RL 220°, RR 140°)도 마스크 범위 안에 있으므로 후방 반발력은 비활성화됨
MOUNT_MASK_LOW  = 120.0
MOUNT_MASK_HIGH = 240.0


# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def build_polar_hist(scan_buf):
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


def nearest_in_arc(hist, has_pt, center_cw, arc_half=25):
    center_bin = int(center_cw / BIN_DEG) % N_BINS
    n_check    = max(1, int(arc_half / BIN_DEG))
    min_d = 9999.0
    for k in range(-n_check, n_check + 1):
        idx = (center_bin + k) % N_BINS
        if has_pt[idx] and hist[idx] < min_d:
            min_d = hist[idx]
    return min_d


def find_gap_steer(hist, has_pt):
    """전방 ±120° 중 가장 먼 방향(갭)의 steer 반환. 마스크 구간 제외."""
    best_dist = 0.0
    best_deg  = 0.0
    for angle_cw in range(0, 361, int(BIN_DEG)):
        if MOUNT_MASK_LOW <= angle_cw <= MOUNT_MASK_HIGH:
            continue
        signed = angle_cw if angle_cw <= 180 else angle_cw - 360
        if abs(signed) > 120:
            continue
        idx = int(angle_cw / BIN_DEG) % N_BINS
        d = hist[idx] if has_pt[idx] else 9999.0
        if d > best_dist:
            best_dist = d
            best_deg  = signed
    return float(np.clip(best_deg * GAP_STEER_GAIN, -MAX_STEER, MAX_STEER))


# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR 백그라운드 스레드
# ─────────────────────────────────────────────────────────────────────────────

class LidarThread(threading.Thread):
    def __init__(self, port: str):
        super().__init__(daemon=True)
        self._port   = port
        self._ser    = None
        self._lock   = threading.Lock()
        self._hist   = [9999.0] * N_BINS
        self._has_pt = [False]  * N_BINS
        self._ready  = False

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
                if distance < 80 or quality == 0:
                    continue
                scan_buf.append((angle, distance))
                if s_flag == 1 and scan_buf:
                    h, hp = build_polar_hist(scan_buf)
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
#  코너 반발력 + 안전 전진 명령 전송
# ─────────────────────────────────────────────────────────────────────────────

def send_fwd(ser, steer: float, speed: float, lidar: LidarThread) -> bool:
    """
    4코너 반발력을 카메라 조향에 더해 전진 명령 전송.
    긴급 정지 발생 시 True 반환.
    """
    hist, has_pt, ready = lidar.get_state()

    if not ready:
        ser.write(f"F {steer:.2f} {speed:.2f}\n".encode())
        return False

    # 전방 거리 확인
    front_d = nearest_in_arc(hist, has_pt, 0.0, FRONT_ARC_HALF)
    if front_d < EMERGENCY_MM:
        ser.write(b"S\n")
        print(f"  [RESIST EMG] 전방 {front_d:.0f}mm → 긴급 정지")
        return True

    # 4코너 거리 측정
    d_fl = nearest_in_arc(hist, has_pt, CORNER_FL_DEG, CORNER_ARC_HALF)
    d_fr = nearest_in_arc(hist, has_pt, CORNER_FR_DEG, CORNER_ARC_HALF)
    d_rl = nearest_in_arc(hist, has_pt, CORNER_RL_DEG, CORNER_ARC_HALF)
    d_rr = nearest_in_arc(hist, has_pt, CORNER_RR_DEG, CORNER_ARC_HALF)

    def rep(d, weight):
        # 거리가 0에 가까울수록 반발력 1, CORNER_REP_DIST 이상은 0
        f = max(0.0, (CORNER_REP_DIST - d) / CORNER_REP_DIST) ** 2
        return f * weight

    # 좌측 코너 → 우로 밀기(+steer), 우측 코너 → 좌로 밀기(-steer)
    rep_left  = rep(d_fl, CORNER_FRONT_W) + rep(d_rl, CORNER_REAR_W)
    rep_right = rep(d_fr, CORNER_FRONT_W) + rep(d_rr, CORNER_REAR_W)
    rep_steer = (rep_left - rep_right) * CORNER_REP_GAIN

    final_steer = float(np.clip(steer + rep_steer, -MAX_STEER, MAX_STEER))

    # 전방 거리 기반 감속
    ratio = max(0.0, min(1.0,
                (VELO_DOWN_MM - front_d) / (VELO_DOWN_MM - EMERGENCY_MM)))
    final_speed = speed * (1.0 - ratio * 0.5)

    ser.write(f"F {final_steer:.2f} {final_speed:.2f}\n".encode())

    if abs(rep_steer) > 0.05:
        print(f"  [RESIST] FL={d_fl:.0f} FR={d_fr:.0f} RL={d_rl:.0f} RR={d_rr:.0f}mm  "
              f"rep={rep_steer:+.3f}  steer:{steer:+.2f}→{final_steer:+.2f}  fwd={front_d:.0f}mm")
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  카메라 유틸리티 (Camera_v2.py 동일)
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


def _get_mask(hsv, color: str):
    if color == 'red':    return get_red_mask(hsv)
    if color == 'yellow': return get_yellow_mask(hsv)
    return get_blue_mask(hsv)


def get_weak_contour(hsv, color: str):
    mask = _get_mask(hsv, color)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > WEAK_MIN_AREA]
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


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
        print(f"[경고] 캘리브 해상도 불일치 → PnP 비활성화")
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

    def _sig_handler(_sig, _frame):
        _cleanup(); sys.exit(0)
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
    print("  resist.py  |  Camera_v2 + LiDAR 코너 반발력")
    print(f"  목표: RED → YELLOW → BLUE")
    print(f"  반발력 거리: {CORNER_REP_DIST:.0f}mm  긴급정지: {EMERGENCY_MM:.0f}mm")
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
            cv2.imshow('Robot View', vis); cv2.waitKey(1)
            time.sleep(0.1)
            continue

        result = detector.detect(raw)
        color  = TARGETS[target_idx]
        vis    = detector.draw_debug(raw, result)

        # LiDAR HUD
        _, _, rdy = lidar.get_state()
        lidar_txt = "LIDAR ON" if rdy else "LIDAR WAIT"
        lidar_col = (0, 200, 0) if rdy else (0, 100, 255)
        cv2.putText(vis, lidar_txt, (fw - 110, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, lidar_col, 2)

        # ── STOP ─────────────────────────────────────────────────────────
        if state == 'STOP':
            ser.write(b"S\n")
            elapsed = time.time() - stop_start
            remain  = max(0.0, STOP_DURATION - elapsed)
            cv2.putText(vis, f"STOP {color.upper()}  {remain:.1f}s",
                        (fw // 2 - 90, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)
            cv2.imshow('Robot View', vis); cv2.waitKey(1)
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

        # ── SEEK ─────────────────────────────────────────────────────────
        det = result.get(color, {})

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
            send_fwd(ser, steer, speed, lidar)
            print(f"  [SEEK] {color.upper()} {log_msg} steer={steer:+.2f}")

        # ② 피크 후 미탐지 ────────────────────────────────────────────────
        elif area_peak_seen:
            hsv_u    = cv2.cvtColor(result['undistorted'], cv2.COLOR_BGR2HSV)
            weak_cnt = get_weak_contour(hsv_u, color)

            if weak_cnt is not None:
                weak_offset = _contour_offset(weak_cnt, fw)
                steer       = float(np.clip(weak_offset * WEAK_STEER_GAIN,
                                            -MAX_STEER, MAX_STEER))
                last_steer  = steer
                last_seen   = time.time()
                send_fwd(ser, steer, WEAK_SPEED, lidar)
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
            weak_cnt = get_weak_contour(hsv_u, color)

            if weak_cnt is not None:
                weak_offset = _contour_offset(weak_cnt, fw)
                steer       = float(np.clip(weak_offset * WEAK_STEER_GAIN,
                                            -MAX_STEER, MAX_STEER))
                last_steer  = steer
                last_seen   = time.time()
                send_fwd(ser, steer, WEAK_SPEED, lidar)
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
                    hist_s, has_pt_s, _ = lidar.get_state()
                    front_d = nearest_in_arc(hist_s, has_pt_s, 0.0, 80)
                    if front_d > GAP_DETECT_MM:
                        arc_steer = search_arc_steer(t_search, base_dir)
                        send_fwd(ser, arc_steer, SEARCH_ARC_SPEED, lidar)
                        phase_lbl = "→우호전" if arc_steer > 0 else "←좌호전"
                        cv2.putText(vis, f"SEARCH {phase_lbl} {t_search:.1f}s",
                                    (5, 58), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (100, 100, 255), 2)
                        print(f"  [SEARCH] {color.upper()} {elapsed:.1f}s arc={arc_steer:+.2f}")
                    else:
                        gap_st = find_gap_steer(hist_s, has_pt_s)
                        send_fwd(ser, gap_st, SEARCH_ARC_SPEED, lidar)
                        cv2.putText(vis, f"GAP {gap_st:+.2f} front={front_d:.0f}mm",
                                    (5, 58), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (0, 180, 255), 2)
                        print(f"  [GAP] {color.upper()} front={front_d:.0f}mm gap={gap_st:+.2f}")

        # ── 공통 HUD ─────────────────────────────────────────────────────
        cv2.putText(vis,
                    f"{state} | {color.upper()} | cnt:{on_zone_count}/{CONFIRM_FRAMES}",
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
