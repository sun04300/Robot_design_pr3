"""
[파일] Camera_v3.py
[목적] Camera_v2 카메라 로직 + Ki+LiDAR_v2 라이다 로직 통합

[카메라 로직 (Camera_v2 기반)]
  · color_v2 — Kobuki 박스 오탐 방지 최신 색상 감지
  · _try_quad → None 반환으로 부분 뷰 판별 → 곡선 선회 접근
  · area_peak_seen → invisible CONFIRM → STOP
  · SEEK / STOP / DONE 상태 구조

[라이다 로직 (Ki+LiDAR_v2 전면 이식)]
  · 전역 상태 딕셔너리 + _lidar_worker 함수형 스레드
  · _vfh_drive / _speed_limit 헬퍼
  · COLOR_MEMORY_TIME 색 소실 메모리
  · smoothed_steer EMA 평활화
  · PnP 다중 알고리즘 폴백 (IPPE→SQPNP→ITERATIVE)
  · _is_valid_quad 기하 검사 + _extract_quad_by_angle 폴백
  · solve_paper_pose_with_memory (이전 PnP 결과 캐시)

[아두이노 명령 프로토콜]
  F {steer:.2f} {speed:.2f}\n  → 전진
  T {dir:.2f}\n                → 제자리 피벗 (+우 / -좌)
  B {speed:.2f}\n              → 후진
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
MAX_STEER          = 1.0
SPEED_FAR          = 0.55
SPEED_NEAR         = 0.35
DIST_SLOW_MM       = 100.0   # PnP z 기준 감속 전환 거리
AREA_SLOW_THRES    = 0.20    # 면적비 기준 감속 (PnP 없을 때)
AREA_PEAK_THRES    = 0.15
STEER_GAIN         = 0.015
CONFIRM_FRAMES     = 4       # 종이 안 보임 확인용 깜빡임 필터
PIVOT_SPEED        = 0.20    # 4꼭짓점 미확보 시 곡선 선회 최소 전진 속도
STOP_DURATION      = 1.1

WEAK_MIN_AREA      = 200
WEAK_SPEED         = 0.35
WEAK_STEER_GAIN    = 0.60
COLOR_MEMORY_TIME  = 0.40    # 색 소실 후 마지막 조향 유지 시간 (s)
STEER_SMOOTH_ALPHA = 0.45    # 조향 EMA 평활화 계수
MAX_POSE_AGE       = 8       # 이전 PnP 결과 재활용 최대 프레임 수
POSE_DECAY_MM      = 15.0    # 프레임당 전진 거리 추정치 (mm)

TARGETS = ['red', 'yellow', 'blue']

# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR VFH 파라미터  (Ki+LiDAR_v2 동일)
# ─────────────────────────────────────────────────────────────────────────────
BIN_DEG       = 4.0
N_BINS        = int(360 / BIN_DEG)
GAP_MIN_PASS  = 90.0
DETECT        = 560.0
VELO_DOWN     = 400.0
EMERGENCY     = 150.0
LID_MAX_STEER = 1.2
ROT_THRESH    = 110.0
ROBOT_RADIUS  = 35.0

# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR 공유 상태  (Ki+LiDAR_v2 전면 이식)
# ─────────────────────────────────────────────────────────────────────────────
_lidar_lock  = threading.Lock()
_lidar_state = {
    'has_data'   : False,
    'emg_near'   : 9999.0,
    'front_near' : 9999.0,
    'vfh_action' : 'FWD',
    'vfh_steer'  : 0.0,
    'vfh_speed'  : 0.65,
    'rot_dir'    : 1.0,
}


# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR VFH 헬퍼  (Ki+LiDAR_v2 전면 이식)
# ─────────────────────────────────────────────────────────────────────────────

def _build_hist(scan_buf):
    hist   = [9999.0] * N_BINS
    has_pt = [False]  * N_BINS
    for a, d in scan_buf:
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
    blocked = [has_pt[i] and hist[i] <= DETECT for i in range(N_BINS)]
    smoothed = blocked[:]
    for i in range(N_BINS):
        if blocked[i] and not blocked[(i-1) % N_BINS] and not blocked[(i+1) % N_BINS]:
            smoothed[i] = False
    blocked = smoothed
    inflated = blocked[:]
    for i in range(N_BINS):
        if blocked[i] and hist[i] < 9999.0:
            ar  = math.asin(min(1.0, ROBOT_RADIUS / max(hist[i], ROBOT_RADIUS)))
            ab  = int(math.degrees(ar) / BIN_DEG) + 1
            for k in range(-ab, ab + 1):
                inflated[(i + k) % N_BINS] = True
    blocked = inflated
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
    """VFH 분석 → (action, steer, speed, rot_dir, emg_near, front_near).  Ki+LiDAR_v2 동일."""
    emg   = _nearest(hist, has_pt, 0.0, arc_half=80)
    front = _nearest(hist, has_pt, 0.0, arc_half=35)

    if not any(has_pt):
        return 'FWD', 0.0, 0.70, 1.0, emg, front

    gaps = _find_gaps(hist, has_pt)
    best = _best_gap(gaps)

    if emg <= EMERGENCY and (best is None or not best['passable']
                              or abs(best['center']) > ROT_THRESH):
        return 'BACK', 0.0, 0.80, 1.0, emg, front

    if best is not None and best['passable'] and abs(best['center']) <= ROT_THRESH:
        imb  = (best['d_R'] - best['d_L']) / (best['d_L'] + best['d_R'] + 1e-9)
        bias = imb * (best['delta_deg'] / 2.9)
        WR   = 150.0
        lL   = _nearest(hist, has_pt, 270.0, arc_half=45)
        lR   = _nearest(hist, has_pt,  90.0, arc_half=45)
        rep  = (max(0.0, WR - lL) / WR - max(0.0, WR - lR) / WR) * 20.0
        CR   = 350.0
        cL   = _nearest(hist, has_pt, 320.0, arc_half=25)
        cR   = _nearest(hist, has_pt,  40.0, arc_half=25)
        crn  = (max(0.0, CR - cL) / CR - max(0.0, CR - cR) / CR) * 45.0
        tgt  = best['center'] + bias + rep + crn
        nd   = _nearest(hist, has_pt, best['center_cw'], arc_half=35)
        rt   = min(max((VELO_DOWN - nd) / (VELO_DOWN - EMERGENCY), 0.0), 1.0)
        st   = max(-LID_MAX_STEER, min(LID_MAX_STEER,
               tgt * (1.0 + rt * 0.5) / 90.0 * LID_MAX_STEER))
        spd  = 0.85 * (1.0 - rt * 0.55)
        return 'FWD', float(st), float(spd), 1.0, emg, front

    FARC = 60.0
    if gaps:
        fg = [g for g in gaps if abs(g['center']) <= FARC]
        td = max(fg, key=lambda g: g['width'])['center'] if fg \
             else max(-FARC, min(FARC, max(gaps, key=lambda g: g['width'])['center']))
    else:
        td = 0.0
    st = max(-LID_MAX_STEER, min(LID_MAX_STEER, td / 90.0 * LID_MAX_STEER * 0.5))
    return 'FWD', float(st), 0.40, 1.0, emg, front


# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR 백그라운드 스레드  (Ki+LiDAR_v2 전면 이식)
# ─────────────────────────────────────────────────────────────────────────────

def _lidar_worker(ser_l):
    scan_buf = []
    while True:
        try:
            data = ser_l.read(5)
            if len(data) != 5:
                continue
            s_flag     = data[0] & 0x01
            s_inv_flag = (data[0] & 0x02) >> 1
            if s_inv_flag != (1 - s_flag):
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
                hist, has_pt = _build_hist(scan_buf)
                act, st, spd, rd, emg, front = _compute_vfh(hist, has_pt)
                with _lidar_lock:
                    _lidar_state['has_data']   = True
                    _lidar_state['emg_near']   = emg
                    _lidar_state['front_near'] = front
                    _lidar_state['vfh_action'] = act
                    _lidar_state['vfh_steer']  = st
                    _lidar_state['vfh_speed']  = spd
                    _lidar_state['rot_dir']    = rd
                scan_buf = []
        except Exception as e:
            print(f"[LIDAR] {e}")
            scan_buf = []


def _lidar_read():
    """현재 LiDAR 상태 스냅샷 반환."""
    with _lidar_lock:
        return dict(_lidar_state)


def _vfh_drive(ser):
    """VFH 결과로 Arduino 주행 명령 전송 (미탐지 탐색용)."""
    ls = _lidar_read()
    if not ls['has_data']:
        ser.write(f"F 0.00 {PIVOT_SPEED:.2f}\n".encode())  # LiDAR 미준비 → 직진 유지
        return "NO_LIDAR_FWD"
    if ls['vfh_action'] == 'BACK':
        ser.write(b"B 0.80\n")
        return f"VFH_BACK emg={ls['emg_near']:.0f}mm"
    ser.write(f"F {ls['vfh_steer']:.2f} {ls['vfh_speed']:.2f}\n".encode())
    return f"VFH_FWD st={ls['vfh_steer']:+.2f} spd={ls['vfh_speed']:.2f}"


def _speed_limit(cam_speed: float) -> float:
    """전방 장애물 거리에 따라 카메라 속도 상한 제한."""
    ls = _lidar_read()
    if not ls['has_data']:
        return cam_speed
    front = ls['front_near']
    if front <= EMERGENCY:
        return 0.0
    if front < VELO_DOWN:
        rt = max(0.0, min(1.0, (VELO_DOWN - front) / (VELO_DOWN - EMERGENCY)))
        return cam_speed * (1.0 - rt * 0.45)
    return cam_speed


# ─────────────────────────────────────────────────────────────────────────────
#  SolvePnP 유틸리티  (Ki+LiDAR_v2 PnP 강화 + Camera_v3 _try_quad 유지)
# ─────────────────────────────────────────────────────────────────────────────

_PNP_FLAGS = [cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE]
try:
    _PNP_FLAGS.insert(1, cv2.SOLVEPNP_SQPNP)
except AttributeError:
    pass

_pose_cache: dict = {'pose': None, 'age': 0}

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


def _is_valid_quad(pts: np.ndarray) -> bool:
    """추출된 4점이 PnP에 사용 가능한 사각형인지 기하학적으로 검사."""
    if pts.shape != (4, 2):
        return False
    hull = cv2.convexHull(pts.astype(np.int32))
    if len(hull) != 4:
        return False
    for i in range(4):
        v1    = pts[(i - 1) % 4] - pts[i]
        v2    = pts[(i + 1) % 4] - pts[i]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        angle = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
        if not (30.0 < angle < 150.0):
            return False
    quad_area = float(cv2.contourArea(pts.astype(np.int32)))
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    if w * h == 0:
        return False
    return quad_area >= w * h * 0.30


def _extract_quad_by_angle(contour: np.ndarray) -> np.ndarray:
    """볼록 껍질에서 내각이 가장 작은 4점 직접 선택."""
    hull = cv2.convexHull(contour)
    pts  = hull.reshape(-1, 2).astype(np.float32)
    n    = len(pts)
    if n == 4:
        return _order_points(pts)
    if n < 4:
        return cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    angles = []
    for i in range(n):
        v1    = pts[(i - 1) % n] - pts[i]
        v2    = pts[(i + 1) % n] - pts[i]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        angles.append(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
    top4_idx = np.argsort(angles)[:4]
    corners  = pts[sorted(top4_idx)]
    return _order_points(corners)


def _extract_quad(contour: np.ndarray) -> np.ndarray:
    """항상 4꼭짓점 반환 (PnP용). _is_valid_quad 검사 포함."""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    best = hull.reshape(-1, 2).astype(np.float32)

    for eps_ratio in np.arange(0.01, 0.50, 0.01):
        approx = cv2.approxPolyDP(hull, float(eps_ratio) * peri, True)
        pts = approx.reshape(-1, 2).astype(np.float32)
        if len(pts) == 4:
            if _is_valid_quad(pts):
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

    if len(best) >= 4 and _is_valid_quad(best):
        return _order_points(best)

    angle_quad = _extract_quad_by_angle(contour)
    if _is_valid_quad(angle_quad):
        return angle_quad

    return _order_points(cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32))


_EDGE_MARGIN = 8

def _try_quad(contour: np.ndarray, frame_w: int = 0, frame_h: int = 0):
    """자연스럽게 4꼭짓점이 나오면 반환, 실패 시 None.
    frame_w/frame_h 지정 시 화면 경계 근처 꼭짓점은 부분 뷰로 간주 → None.
    """
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps_ratio in np.arange(0.01, 0.50, 0.01):
        approx = cv2.approxPolyDP(hull, float(eps_ratio) * peri, True)
        pts    = approx.reshape(-1, 2).astype(np.float32)
        if len(pts) == 4:
            if frame_w > 0 and frame_h > 0:
                if ((pts[:, 0] < _EDGE_MARGIN).any() or
                        (pts[:, 0] > frame_w - _EDGE_MARGIN).any() or
                        (pts[:, 1] < _EDGE_MARGIN).any() or
                        (pts[:, 1] > frame_h - _EDGE_MARGIN).any()):
                    return None
            if _is_valid_quad(pts):
                return _order_points(pts)
        if len(pts) < 4:
            break
    return None


def solve_paper_pose(contour, cam_mat, dist_coeffs, quad_pts=None):
    """PnP 다중 알고리즘 폴백 체인."""
    if cam_mat is None:
        return None
    if quad_pts is None:
        quad_pts = _extract_quad(contour)
    for flag in _PNP_FLAGS:
        try:
            ok, _, tvec = cv2.solvePnP(_OBJ_PTS, quad_pts, cam_mat, dist_coeffs,
                                        flags=flag)
        except cv2.error:
            continue
        if not ok:
            continue
        z_mm = float(tvec[2][0])
        x_mm = float(tvec[0][0])
        if np.isfinite(z_mm) and np.isfinite(x_mm) and z_mm > 0:
            angle = np.degrees(np.arctan2(x_mm, z_mm))
            steer = float(np.clip(angle * STEER_GAIN, -MAX_STEER, MAX_STEER))
            return z_mm, x_mm, steer, quad_pts
    return None


def solve_paper_pose_with_memory(contour, cam_mat, dist_coeffs, quad_pts=None):
    """PnP 실패 시 이전 유효 결과를 최대 MAX_POSE_AGE 프레임까지 재활용."""
    result = solve_paper_pose(contour, cam_mat, dist_coeffs, quad_pts=quad_pts)
    if result is not None:
        _pose_cache['pose'] = result
        _pose_cache['age']  = 0
        return result
    _pose_cache['age'] += 1
    if _pose_cache['pose'] is not None and _pose_cache['age'] <= MAX_POSE_AGE:
        z_mm, x_mm, _, qp = _pose_cache['pose']
        z_est = max(10.0, z_mm - POSE_DECAY_MM * _pose_cache['age'])
        angle = np.degrees(np.arctan2(x_mm, z_est))
        steer = float(np.clip(angle * STEER_GAIN, -MAX_STEER, MAX_STEER))
        return z_est, x_mm, steer, qp
    _pose_cache['pose'] = None
    return None


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


def _contour_offset(cnt, frame_w: int, opt_cx: float = None) -> float:
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return 0.0
    center = opt_cx if opt_cx is not None else frame_w / 2
    return (M['m10'] / M['m00'] - center) / (frame_w / 2)


def _draw_center(vis, cx: int, cy: int, color):
    cv2.circle(vis, (cx, cy), 6, color, -1)
    cv2.line(vis, (cx - 15, cy), (cx + 15, cy), color, 1)
    cv2.line(vis, (cx, cy - 15), (cx, cy + 15), color, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ser   = serial.Serial(PORT_ARDU,  460800, timeout=1)
    ser_l = serial.Serial(PORT_LIDAR, 460800, timeout=1)

    ser_l.write(bytes([0xA5, 0x40]))
    time.sleep(1.0)
    ser_l.reset_input_buffer()
    ser_l.write(bytes([0xA5, 0x20]))
    print(f"[LIDAR] 스캔 시작: {PORT_LIDAR}")

    t_lidar = threading.Thread(target=_lidar_worker, args=(ser_l,), daemon=True)
    t_lidar.start()

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

    opt_cx = fw / 2.0

    def _cleanup():
        try:
            ser.write(b"S\n"); time.sleep(0.1)
            ser_l.write(bytes([0xA5, 0x25])); time.sleep(0.1)
            cap.release(); cv2.destroyAllWindows()
            ser.close(); ser_l.close()
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
    smoothed_steer = 0.0
    area_peak_seen = False
    peak_area_r    = 0.0

    print("=" * 65)
    print("  Camera_v3  |  color_v2 + Ki+LiDAR_v2 LiDAR 전면 이식")
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

        # ── LiDAR 긴급 후진 (SEEK 최우선) ────────────────────────────────
        ls = _lidar_read()
        if state == 'SEEK' and ls['has_data'] and ls['emg_near'] <= EMERGENCY:
            ser.write(b"B 0.80\n")
            vis = raw.copy()
            cv2.putText(vis, f"LIDAR EMERGENCY {ls['emg_near']:.0f}mm",
                        (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.imshow('Robot View', vis)
            cv2.waitKey(1)
            print(f"  [EMG] {ls['emg_near']:.0f}mm → 후진")
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
                    target_idx          += 1
                    state                = 'SEEK'
                    on_zone_count        = 0
                    area_peak_seen       = False
                    peak_area_r          = 0.0
                    last_seen            = time.time()
                    _pose_cache['pose']  = None
                    _pose_cache['age']   = 0
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
            area_r    = det['area'] / (fw * fh)

            # 4꼭짓점 자연 추출 (경계 체크 포함)
            quad = _try_quad(cnt, fw, fh)
            pose = solve_paper_pose_with_memory(cnt, pnp_mat, pnp_dist, quad_pts=quad) \
                   if quad is not None else None

            # 강탐지기(min_area=1000)가 이미 노이즈 차단 → area 기준만으로 피크 기록
            if area_r > AREA_PEAK_THRES:
                area_peak_seen = True
                peak_area_r    = max(peak_area_r, area_r)

            if pose is not None:
                # ── PnP 성공: 정밀 조향 ────────────────────────────────
                z_mm, x_mm, steer, quad_pts = pose
                speed   = _speed_limit(SPEED_NEAR if z_mm < DIST_SLOW_MM else SPEED_FAR)
                log_msg = f"PnP z={z_mm:.0f}mm x={x_mm:+.0f}mm area={area_r:.2f}"
                cv2.putText(vis, f"Z={z_mm:.0f}mm X={x_mm:+.0f}mm A={area_r:.3f}",
                            (fw // 2 - 160, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
                cv2.polylines(vis, [quad_pts.astype(np.int32)], True, (0, 220, 255), 2)
                ctr = quad_pts.mean(axis=0).astype(int)
                _draw_center(vis, int(ctr[0]), int(ctr[1]), (0, 220, 255))

            elif quad is not None:
                # ── 4꼭짓점 있지만 PnP 수치 실패 → 꼭짓점 픽셀 중심 ──
                ctr_x  = float(quad[:, 0].mean())
                ctr_y  = float(quad[:, 1].mean())
                offset = (ctr_x - fw / 2) / (fw / 2)
                steer  = float(np.clip(offset * 0.45, -MAX_STEER, MAX_STEER))
                speed  = _speed_limit(SPEED_NEAR if area_r > AREA_SLOW_THRES else SPEED_FAR)
                log_msg = f"quad-ctr off={offset:+.2f} area={area_r:.2f}"
                cv2.putText(vis, f"QUAD-CTR A={area_r:.3f}",
                            (fw // 2 - 70, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 200), 2)
                cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 200, 200), 2)
                _draw_center(vis, int(ctr_x), int(ctr_y), (0, 200, 200))

            else:
                # ── 4꼭짓점 미검출 ──
                M_c   = cv2.moments(cnt)
                ctr_x = M_c['m10'] / M_c['m00'] if M_c['m00'] > 0 else fw / 2
                ctr_y = M_c['m01'] / M_c['m00'] if M_c['m00'] > 0 else fh / 2
                offset = (ctr_x - fw / 2) / (fw / 2)
                if M_c['m00'] > 0:
                    _draw_center(vis, int(ctr_x), int(ctr_y), (255, 200, 0))

                if area_r > AREA_SLOW_THRES:
                    # 매우 가까움(면적>20%) → 부드럽게 직진 진입
                    steer   = float(np.clip(offset * 0.5, -MAX_STEER, MAX_STEER))
                    speed   = _speed_limit(SPEED_NEAR)
                    log_msg = f"CLOSE-FWD off={offset:+.2f} area={area_r:.2f}"
                    cv2.putText(vis, f"CLOSE-FWD A={area_r:.3f}",
                                (fw // 2 - 80, 38),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)
                else:
                    # 부분 뷰 → 곡선 선회 접근
                    steer   = float(np.clip(offset * 3.0, -MAX_STEER, MAX_STEER))
                    speed   = PIVOT_SPEED
                    arrow   = '→' if steer > 0 else '←'
                    log_msg = f"CURVE{arrow} off={offset:+.2f} area={area_r:.2f}"
                    cv2.putText(vis, f"CURVE{arrow}  A={area_r:.3f}",
                                (fw // 2 - 80, 38),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

            last_steer     = steer
            smoothed_steer = steer  # 강탐지 시 평활화 즉시 동기화
            if speed > 0:
                ser.write(f"F {steer:.2f} {speed:.2f}\n".encode())
                print(f"  [SEEK] {color.upper()} {log_msg} steer={steer:+.2f} spd={speed:.2f}")
            else:
                ser.write(b"S\n")
                print(f"  [SEEK] {color.upper()} {log_msg} (속도제한 0)")

        # ② 피크 후 미탐지 (종이 위 진입 중) ───────────────────────────────
        elif area_peak_seen:
            hsv_u    = cv2.cvtColor(result['undistorted'], cv2.COLOR_BGR2HSV)
            weak_cnt = get_weak_contour(hsv_u, color)

            if weak_cnt is not None:
                on_zone_count  = 0          # 약한 윤곽 보이면 카운터 리셋 (조기 정지 방지)
                weak_offset    = _contour_offset(weak_cnt, fw, opt_cx)
                steer          = float(np.clip(weak_offset * WEAK_STEER_GAIN,
                                               -MAX_STEER, MAX_STEER))
                last_steer     = steer
                last_seen      = time.time()
                steer_cmd      = STEER_SMOOTH_ALPHA * steer + (1.0 - STEER_SMOOTH_ALPHA) * smoothed_steer
                smoothed_steer = steer_cmd
                ser.write(f"F {steer_cmd:.2f} {WEAK_SPEED:.2f}\n".encode())
                cv2.drawContours(vis, [weak_cnt], -1, (180, 180, 0), 1)
                M_w = cv2.moments(weak_cnt)
                if M_w['m00'] > 0:
                    _draw_center(vis, int(M_w['m10'] / M_w['m00']),
                                 int(M_w['m01'] / M_w['m00']), (180, 255, 0))
                cv2.putText(vis, f"ENTERING {color.upper()} off={weak_offset:+.2f}",
                            (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 0), 2)
                print(f"  [ENTER] {color.upper()} off={weak_offset:+.2f} st={steer_cmd:+.2f}")
            else:
                on_zone_count += 1
                ser.write(b"S\n")
                cv2.putText(vis, f"INVISIBLE cnt:{on_zone_count}/{CONFIRM_FRAMES}",
                            (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)
                print(f"  [INVIS] {color.upper()} pk={peak_area_r:.2f} cnt={on_zone_count}")
                if on_zone_count >= CONFIRM_FRAMES:
                    state = 'STOP'; stop_start = time.time()
                    ser.write(b"S\n")
                    print(f"  🎯 {color.upper()} 도달! peak={peak_area_r:.2f}")
                    cv2.imshow('Robot View', vis); cv2.waitKey(1)
                    continue

        # ③ 미탐지 → VFH 탐색 ─────────────────────────────────────────────
        else:
            on_zone_count = max(0, on_zone_count - 1)
            hsv_u    = cv2.cvtColor(result['undistorted'], cv2.COLOR_BGR2HSV)
            weak_cnt = get_weak_contour(hsv_u, color)

            if weak_cnt is not None:
                # 피크 미확인 + 약탐지 → 최대 조향 + 최소 전진 곡선 접근
                weak_offset = _contour_offset(weak_cnt, fw, opt_cx)
                w_steer = float(np.clip(weak_offset * 3.0, -MAX_STEER, MAX_STEER))
                last_seen = time.time()
                ser.write(f"F {w_steer:.2f} {PIVOT_SPEED:.2f}\n".encode())
                cv2.drawContours(vis, [weak_cnt], -1, (180, 180, 0), 1)
                x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(weak_cnt)
                _draw_center(vis, x_bb + w_bb // 2, y_bb + h_bb // 2, (180, 180, 0))
                arrow = '→' if w_steer > 0 else '←'
                cv2.putText(vis, f"WEAK-CURVE{arrow} off={weak_offset:+.2f}",
                            (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 0), 2)
                print(f"  [WEAK] {color.upper()} curve{arrow} off={weak_offset:+.2f}")
            else:
                elapsed = time.time() - last_seen
                if elapsed < COLOR_MEMORY_TIME:
                    # 색 소실 직후: 마지막 조향 방향 유지
                    steer_cmd      = STEER_SMOOTH_ALPHA * last_steer + (1.0 - STEER_SMOOTH_ALPHA) * smoothed_steer
                    smoothed_steer = steer_cmd
                    mem_speed      = _speed_limit(SPEED_NEAR * 0.7)
                    if mem_speed > 0:
                        ser.write(f"F {steer_cmd:.2f} {mem_speed:.2f}\n".encode())
                    else:
                        ser.write(b"S\n")
                    cv2.putText(vis, f"MEM {elapsed:.2f}s st={steer_cmd:+.2f}",
                                (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 100), 2)
                    print(f"  [MEM] {color.upper()} t={elapsed:.2f}s st={steer_cmd:+.2f}")
                else:
                    # VFH 탐색
                    log = _vfh_drive(ser)
                    smoothed_steer *= (1.0 - STEER_SMOOTH_ALPHA)
                    cv2.putText(vis, f"VFH {log}",
                                (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 200, 255), 2)
                    print(f"  [VFH] {color.upper()} {elapsed:.1f}s → {log}")

        # ── 공통 HUD ──────────────────────────────────────────────────────
        emg_txt = f" EMG:{ls['emg_near']:.0f}mm" if ls['has_data'] else ""
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
