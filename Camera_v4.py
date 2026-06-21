"""
[파일] Camera_v4.py
[목적] Camera_v3 경량화 — PnP/quad 제거, minAreaRect+distanceTransform 중심 추출

[v3 대비 변경점]
  · SolvePnP / _try_quad / _is_valid_quad / _extract_quad 전면 제거
  · 처리 해상도 320×240 다운스케일 → 프레임당 연산량 ¼
  · 중심 추출: minAreaRect(정상) / distanceTransform(가림·왜곡) 이중 폴백
  · 모폴로지 연산 color_v2 내부 최소한만 유지

[유지]
  · LiDAR VFH (Ki+LiDAR_v2 동일)
  · SEEK / STOP / DONE 상태 머신
  · 아두이노 F/B/S/T 시리얼 프로토콜
  · area_r 기반 속도 제어 (CLOSE-FWD / FWD-NQ / CURVE)
  · area_peak_seen → ENTERING → STOP 흐름
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

from color_v2 import (load_calibration,
                       get_red_mask, get_yellow_mask, get_blue_mask,
                       RED_LOWER1, RED_UPPER1, RED_LOWER2, RED_UPPER2,
                       YELLOW_LOWER, YELLOW_UPPER,
                       BLUE_LOWER, BLUE_UPPER)


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
#  탐지 파라미터
# ─────────────────────────────────────────────────────────────────────────────
MIN_AREA       = 1000   # 강탐지 최소 컨투어 면적 (640×480 기준 px²)
WEAK_MIN_AREA  = 200    # 약탐지 최소 컨투어 면적
AR_MIN         = 0.5    # minAreaRect 종횡비 정상 범위 하한
AR_MAX         = 2.0    # minAreaRect 종횡비 정상 범위 상한

# ─────────────────────────────────────────────────────────────────────────────
#  카메라 주행 파라미터
# ─────────────────────────────────────────────────────────────────────────────
MAX_STEER         = 1.0
SPEED_FAR         = 0.55
SPEED_NEAR        = 0.45
PIVOT_SPEED       = 0.25
WEAK_SPEED        = 0.45
WEAK_STEER_GAIN   = 0.60
AREA_SLOW_THRES   = 0.20   # 근접 판단 면적비 (PROC 기준)
AREA_PEAK_THRES   = 0.20   # area_peak_seen 세팅 면적비
CONFIRM_FRAMES    = 4      # INVISIBLE 확인용 프레임 수
STOP_DURATION     = 1.1    # 정지 대기 시간 (초)
COLOR_MEMORY_TIME = 0.40   # 색 소실 후 조향 유지 시간 (초)
STEER_SMOOTH_ALPHA = 0.45  # EMA 평활화 계수

TARGETS = ['red', 'yellow', 'blue']

# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR VFH 파라미터  (Ki+Lider_Re-Lat 기준)
# ─────────────────────────────────────────────────────────────────────────────
BIN_DEG       = 4.0
N_BINS        = int(360 / BIN_DEG)
GAP_MIN_PASS  = 100.0
DETECT        = 500.0
VELO_DOWN     = 400.0
EMERGENCY     = 200.0
LID_MAX_STEER = 1.2
ROT_THRESH    = 100.0
ROBOT_RADIUS  = 60.0

OBS_RETURN_TIME = 10.0   # 장애물 통과 후 복귀 조향 유지 시간 (s)
OBS_RETURN_GAIN = 0.70   # 복귀 조향 계수

# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR 공유 상태  (Ki+LiDAR_v2 동일)
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
    'lat_L'      : 9999.0,
    'lat_R'      : 9999.0,
}


# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR VFH 헬퍼  (Ki+LiDAR_v2 동일)
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
    emg   = _nearest(hist, has_pt, 0.0,   arc_half=80)
    front = _nearest(hist, has_pt, 0.0,   arc_half=35)
    lL    = _nearest(hist, has_pt, 270.0, arc_half=45)
    lR    = _nearest(hist, has_pt,  90.0, arc_half=45)
    if not any(has_pt):
        return 'FWD', 0.0, 0.70, 1.0, emg, front, lL, lR
    gaps = _find_gaps(hist, has_pt)
    best = _best_gap(gaps)
    if best is not None and best['passable'] and abs(best['center']) <= ROT_THRESH:
        imb  = (best['d_R'] - best['d_L']) / (best['d_L'] + best['d_R'] + 1e-9)
        bias = imb * (best['delta_deg'] / 2.9)
        WR   = 150.0
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
        return 'FWD', float(st), float(spd), 1.0, emg, front, lL, lR
    FARC = 60.0
    if gaps:
        fg = [g for g in gaps if abs(g['center']) <= FARC]
        td = max(fg, key=lambda g: g['width'])['center'] if fg \
             else max(-FARC, min(FARC, max(gaps, key=lambda g: g['width'])['center']))
    else:
        td = 0.0
    st = max(-LID_MAX_STEER, min(LID_MAX_STEER, td / 90.0 * LID_MAX_STEER * 0.5))
    return 'FWD', float(st), 0.40, 1.0, emg, front, lL, lR


# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR 백그라운드 스레드  (Ki+LiDAR_v2 동일)
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
                act, st, spd, rd, emg, front, lL, lR = _compute_vfh(hist, has_pt)
                with _lidar_lock:
                    _lidar_state['has_data']   = True
                    _lidar_state['emg_near']   = emg
                    _lidar_state['front_near'] = front
                    _lidar_state['vfh_action'] = act
                    _lidar_state['vfh_steer']  = st
                    _lidar_state['vfh_speed']  = spd
                    _lidar_state['rot_dir']    = rd
                    _lidar_state['lat_L']      = lL
                    _lidar_state['lat_R']      = lR
                scan_buf = []
        except Exception as e:
            print(f"[LIDAR] {e}")
            scan_buf = []


def _lidar_read():
    with _lidar_lock:
        return dict(_lidar_state)


def _vfh_drive(ser):
    ls = _lidar_read()
    if not ls['has_data']:
        ser.write(f"F 0.00 {PIVOT_SPEED:.2f}\n".encode())
        return "NO_LIDAR_FWD"
    ser.write(f"F {ls['vfh_steer']:.2f} {ls['vfh_speed']:.2f}\n".encode())
    return f"VFH_FWD st={ls['vfh_steer']:+.2f} spd={ls['vfh_speed']:.2f}"


def _speed_limit(cam_speed: float) -> float:
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
#  경량 색지 탐지  (minAreaRect + distanceTransform)
# ─────────────────────────────────────────────────────────────────────────────

def _get_mask(hsv, color: str):
    if color == 'red':    return get_red_mask(hsv)
    if color == 'yellow': return get_yellow_mask(hsv)
    return get_blue_mask(hsv)


def _detect_paper(frame, color: str):
    """
    640×480 프레임에서 색지 탐지.
    반환: None  또는
          {'cx','cy','area_r','contour'}
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = _get_mask(hsv, color)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > MIN_AREA]
    if not cnts:
        return None

    cnt    = max(cnts, key=cv2.contourArea)
    area_r = cv2.contourArea(cnt) / (CAM_W * CAM_H)

    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return None
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']

    return {
        'cx': cx, 'cy': cy,
        'area_r': area_r,
        'contour': cnt,
    }


def _weak_detect(frame, color: str):
    """
    약탐지: 낮은 면적 임계값으로 종이 일부만 보여도 중심 반환.
    반환: None 또는 (contour, cx, cy)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 모폴로지 없이 raw inRange — 작은/먼 blob도 방향 탐지 가능
    if color == 'red':
        mask = cv2.bitwise_or(cv2.inRange(hsv, RED_LOWER1, RED_UPPER1),
                              cv2.inRange(hsv, RED_LOWER2, RED_UPPER2))
    elif color == 'yellow':
        mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    else:
        mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > WEAK_MIN_AREA]
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    M   = cv2.moments(cnt)
    if M['m00'] == 0:
        return None
    return cnt, M['m10'] / M['m00'], M['m01'] / M['m00']


# ─────────────────────────────────────────────────────────────────────────────
#  보조 함수
# ─────────────────────────────────────────────────────────────────────────────

def _offset(cx: float) -> float:
    return (cx - CAM_W / 2) / (CAM_W / 2)


def _draw_ctr(vis, cx, cy, color):
    x, y = int(cx), int(cy)
    cv2.circle(vis, (x, y), 5, color, -1)
    cv2.line(vis, (x - 12, y), (x + 12, y), color, 1)
    cv2.line(vis, (x, y - 12), (x, y + 12), color, 1)


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

    # 캘리브레이션 로드 → undistort 맵 생성
    cam_mat, dist_coeffs, calib_res = load_calibration(CALIB_FILE)
    map1 = map2 = None
    if cam_mat is not None and calib_res == (CAM_W, CAM_H):
        new_mtx, _ = cv2.getOptimalNewCameraMatrix(
            cam_mat, dist_coeffs, (CAM_W, CAM_H), 1, (CAM_W, CAM_H))
        map1, map2 = cv2.initUndistortRectifyMap(
            cam_mat, dist_coeffs, None, new_mtx, (CAM_W, CAM_H), cv2.CV_16SC2)
        print("[CAM] 캘리브레이션 적용")
    else:
        print("[CAM] 캘리브레이션 없음 (raw 사용)")

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

    # ── 상태 변수 ────────────────────────────────────────────────────────────
    target_idx     = 0
    state          = 'SEEK'
    on_zone_count  = 0
    stop_start     = None
    last_seen      = time.time()
    last_steer     = 0.0
    smoothed_steer = 0.0
    area_peak_seen        = False
    peak_area_r           = 0.0
    approach_steer_locked = False
    locked_approach_steer = 0.0
    prev_area_r           = 0.0
    last_obs_steer        = 0.0
    obs_active            = False
    obs_clear_time        = None

    print("=" * 60)
    print("  Camera_v4  |  경량화 (minAreaRect+distanceTransform)")
    print(f"  처리해상도: {CAM_W}×{CAM_H}   목표: RED→YELLOW→BLUE")
    print("=" * 60)

    while True:
        ret, raw = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # ── undistort ────────────────────────────────────────────────────
        undist = cv2.remap(raw, map1, map2, cv2.INTER_LINEAR) if map1 is not None else raw
        vis    = undist.copy()

        # ── DONE ─────────────────────────────────────────────────────────
        if state == 'DONE':
            ser.write(b"S\n")
            cv2.putText(vis, "MISSION COMPLETE", (40, CAM_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow('Robot View', vis)
            cv2.waitKey(1); time.sleep(0.1)
            continue

        # ── LiDAR 긴급 후진 (SEEK 최우선) ────────────────────────────────
        ls = _lidar_read()
        if state == 'SEEK' and ls['has_data'] and ls['emg_near'] <= EMERGENCY:
            ser.write(b"B 0.80\n")
            cv2.putText(vis, f"EMG BACK {ls['emg_near']:.0f}mm",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.imshow('Robot View', vis)
            cv2.waitKey(1)
            print(f"  [EMG] {ls['emg_near']:.0f}mm → 후진")
            continue

        color = TARGETS[target_idx]

        # ── STOP 대기 ─────────────────────────────────────────────────────
        if state == 'STOP':
            ser.write(b"S\n")
            elapsed = time.time() - stop_start
            remain  = max(0.0, STOP_DURATION - elapsed)
            cv2.putText(vis, f"STOP {color.upper()}  {remain:.1f}s",
                        (CAM_W // 2 - 140, 56), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, (0, 255, 255), 2)
            cv2.imshow('Robot View', vis)
            cv2.waitKey(1)
            if elapsed >= STOP_DURATION:
                if target_idx < len(TARGETS) - 1:
                    target_idx            += 1
                    state                  = 'SEEK'
                    on_zone_count          = 0
                    area_peak_seen         = False
                    peak_area_r            = 0.0
                    approach_steer_locked  = False
                    locked_approach_steer  = 0.0
                    prev_area_r            = 0.0
                    last_obs_steer         = 0.0
                    obs_active             = False
                    obs_clear_time         = None
                    last_seen              = time.time()
                    print(f"  ✅ {color.upper()} 완료 → {TARGETS[target_idx].upper()}")
                else:
                    state = 'DONE'
                    print("  ✅ 전체 미션 완료!")
            continue

        # ── SEEK ─────────────────────────────────────────────────────────
        det = _detect_paper(undist, color)

        # ① 강탐지 ──────────────────────────────────────────────────────
        if det is not None:
            last_seen = time.time()
            cx, cy    = det['cx'], det['cy']
            area_r    = det['area_r']
            offset    = _offset(cx)
            dcx, dcy  = int(cx), int(cy)

            if area_r >= AREA_PEAK_THRES:
                area_peak_seen = True
                peak_area_r    = max(peak_area_r, area_r)

            if area_r >= AREA_SLOW_THRES:
                # CLOSE-FWD: 최초 진입 시 조향 고정
                if not approach_steer_locked:
                    if prev_area_r > 0.05:
                        # 멀리서 서서히 접근 → 현재 offset 기반 부드러운 보정 (누적 steer 대신)
                        locked_approach_steer = float(np.clip(offset * 0.25, -0.35, 0.35))
                    else:
                        # 회전 중 갑자기 발견 → 직진
                        locked_approach_steer = 0.0
                    approach_steer_locked = True
                steer   = locked_approach_steer
                speed   = _speed_limit(SPEED_NEAR)
                src     = "approach" if prev_area_r > 0.05 else "sudden"
                log_msg = f"CLOSE-FWD(lock/{src}) st={steer:+.2f} area={area_r:.2f}"
                cv2.putText(vis, f"CLOSE-FWD A={area_r:.2f} lock={steer:+.2f}",
                            (CAM_W // 2 - 180, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)

            else:
                # 멀리 있을 때: 정상 조향 업데이트 + lock 해제
                approach_steer_locked = False
                steer = float(np.clip(offset * 1.5, -MAX_STEER, MAX_STEER))
                if abs(steer) < 0.4:
                    speed   = _speed_limit(SPEED_NEAR)
                    log_msg = f"FWD-NQ off={offset:+.2f} area={area_r:.2f}"
                    cv2.putText(vis, f"FWD-NQ A={area_r:.2f}",
                                (CAM_W // 2 - 140, 44),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 100), 2)
                else:
                    speed   = PIVOT_SPEED
                    arrow   = '→' if steer > 0 else '←'
                    log_msg = f"CURVE{arrow} off={offset:+.2f} area={area_r:.2f}"
                    cv2.putText(vis, f"CURVE{arrow} A={area_r:.2f}",
                                (CAM_W // 2 - 140, 44),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

            x, y, w, h = cv2.boundingRect(det['contour'])
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 180, 255), 2)
            _draw_ctr(vis, dcx, dcy, (0, 180, 255))

            smoothed_steer = STEER_SMOOTH_ALPHA * steer + (1.0 - STEER_SMOOTH_ALPHA) * smoothed_steer
            last_steer     = smoothed_steer
            prev_area_r    = area_r
            if speed > 0:
                ser.write(f"F {smoothed_steer:.2f} {speed:.2f}\n".encode())
                print(f"  [SEEK] {color.upper()} {log_msg} spd={speed:.2f}")
            else:
                ser.write(b"S\n")
                print(f"  [SEEK] {color.upper()} {log_msg} (속도제한 0)")

        # ② 피크 후 미탐지 → ENTERING ────────────────────────────────────
        elif area_peak_seen:
            prev_area_r = 0.0
            w = _weak_detect(undist, color)
            if w is not None:
                cnt_w, wcx, wcy = w
                on_zone_count  = 0
                weak_offset    = _offset(wcx)
                steer          = float(np.clip(weak_offset * WEAK_STEER_GAIN,
                                               -MAX_STEER, MAX_STEER))
                last_steer     = steer
                last_seen      = time.time()
                steer_cmd      = STEER_SMOOTH_ALPHA * steer + (1.0 - STEER_SMOOTH_ALPHA) * smoothed_steer
                smoothed_steer = steer_cmd
                ser.write(f"F {steer_cmd:.2f} {WEAK_SPEED:.2f}\n".encode())
                cv2.drawContours(vis, [cnt_w], -1, (180, 255, 0), 1)
                _draw_ctr(vis, int(wcx), int(wcy), (180, 255, 0))
                cv2.putText(vis, f"ENTERING {color.upper()} off={weak_offset:+.2f}",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 255, 0), 2)
                print(f"  [ENTER] {color.upper()} off={weak_offset:+.2f} st={steer_cmd:+.2f}")
            else:
                on_zone_count += 1
                ser.write(b"S\n")
                cv2.putText(vis, f"INVISIBLE cnt:{on_zone_count}/{CONFIRM_FRAMES}",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)
                print(f"  [INVIS] {color.upper()} pk={peak_area_r:.2f} cnt={on_zone_count}")
                if on_zone_count >= CONFIRM_FRAMES:
                    state = 'STOP'; stop_start = time.time()
                    ser.write(b"S\n")
                    print(f"  🎯 {color.upper()} 도달! peak={peak_area_r:.2f}")
                    cv2.imshow('Robot View', vis)
                    cv2.waitKey(1)
                    continue

        # ③ 미탐지 → VFH 탐색 ────────────────────────────────────────────
        else:
            prev_area_r   = 0.0
            on_zone_count = max(0, on_zone_count - 1)
            w = _weak_detect(undist, color)
            if w is not None:
                cnt_w, wcx, wcy = w
                weak_offset = _offset(wcx)
                w_steer     = float(np.clip(weak_offset * 3.0, -MAX_STEER, MAX_STEER))
                last_seen   = time.time()
                ser.write(f"F {w_steer:.2f} {PIVOT_SPEED:.2f}\n".encode())
                cv2.drawContours(vis, [cnt_w], -1, (180, 180, 0), 1)
                arrow = '→' if w_steer > 0 else '←'
                cv2.putText(vis, f"WEAK-CURVE{arrow} off={weak_offset:+.2f}",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 0), 2)
                print(f"  [WEAK] {color.upper()} curve{arrow} off={weak_offset:+.2f}")
            else:
                elapsed = time.time() - last_seen
                if elapsed < COLOR_MEMORY_TIME:
                    steer_cmd      = STEER_SMOOTH_ALPHA * last_steer + (1.0 - STEER_SMOOTH_ALPHA) * smoothed_steer
                    smoothed_steer = steer_cmd
                    mem_speed      = _speed_limit(SPEED_NEAR * 0.7)
                    ser.write(f"F {steer_cmd:.2f} {mem_speed:.2f}\n".encode()
                              if mem_speed > 0 else b"S\n")
                    cv2.putText(vis, f"MEM {elapsed:.2f}s st={steer_cmd:+.2f}",
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 220, 100), 2)
                    print(f"  [MEM] {color.upper()} t={elapsed:.2f}s st={steer_cmd:+.2f}")
                else:
                    obstacle_now = ls['has_data'] and ls['emg_near'] < DETECT
                    if obstacle_now:
                        last_obs_steer = float(np.clip(
                            (ls['lat_L'] - ls['lat_R']) / (DETECT + 1e-9), -1.0, 1.0))
                        if not obs_active:
                            obs_clear_time = None
                        obs_active = True
                    elif obs_active:
                        obs_active     = False
                        obs_clear_time = time.time()

                    return_elapsed = (time.time() - obs_clear_time) \
                        if obs_clear_time is not None else OBS_RETURN_TIME + 1.0

                    if return_elapsed < OBS_RETURN_TIME:
                        ret_st  = float(np.clip(last_obs_steer * OBS_RETURN_GAIN,
                                                -MAX_STEER, MAX_STEER))
                        ret_spd = _speed_limit(SPEED_NEAR)
                        ser.write(f"F {ret_st:.2f} {ret_spd:.2f}\n".encode()
                                  if ret_spd > 0 else b"S\n")
                        cv2.putText(vis, f"OBS_RET st={ret_st:+.2f} {return_elapsed:.1f}s",
                                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 165, 255), 2)
                        print(f"  [OBS_RET] {color.upper()} st={ret_st:+.2f} t={return_elapsed:.1f}s")
                    else:
                        log = _vfh_drive(ser)
                        smoothed_steer *= (1.0 - STEER_SMOOTH_ALPHA)
                        cv2.putText(vis, f"VFH {log}",
                                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (100, 200, 255), 2)
                        print(f"  [VFH] {color.upper()} {elapsed:.1f}s → {log}")

        # ── 공통 HUD ─────────────────────────────────────────────────────
        emg_txt = f" EMG:{ls['emg_near']:.0f}mm" if ls['has_data'] else ""
        cv2.putText(vis,
                    f"{state}|{color.upper()}|cnt:{on_zone_count}/{CONFIRM_FRAMES}{emg_txt}",
                    (10, CAM_H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        cv2.imshow('Robot View', vis)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            print("  [종료] q 입력")
            break

    ser.write(b"S\n")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()