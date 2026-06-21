"""
[파일] Camera_v5_APF.py
[목적] APF(Artificial Potential Field) 기반 통합 내비게이션 예비 코드

[v4 대비 핵심 차이]
  · VFH 갭 탐색 제거
  · 카메라 인력(attractive) + LiDAR 척력(repulsive) → 단일 합력 벡터
  · 종이 탐지 중에도 장애물 척력이 동시에 반영 (v4는 분리 동작)
  · 랜덤 장애물 환경 대응: 갭 존재 여부 무관하게 합력이 자연스럽게 우회 경로 생성
  · 로컬 미니멈 탈출: 일정 시간 속도 0 지속 시 제자리 회전 실행

[유지]
  · SEEK / STOP / DONE 상태 머신
  · CLOSE-FWD 조향 잠금 (area ≥ 0.20 최종 접근)
  · ENTERING / INVISIBLE / NUDGE 흐름
  · 아두이노 F/S/T 시리얼 프로토콜
  · color_v2 탐지 함수
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
#  탐지 파라미터  (v4 동일)
# ─────────────────────────────────────────────────────────────────────────────
MIN_AREA      = 1000
WEAK_MIN_AREA = 200

# ─────────────────────────────────────────────────────────────────────────────
#  카메라 주행 파라미터  (v4 동일)
# ─────────────────────────────────────────────────────────────────────────────
MAX_STEER          = 1.0
SPEED_NEAR         = 0.45
PIVOT_SPEED        = 0.25
WEAK_SPEED         = 0.45
NUDGE_SPEED        = 0.25
WEAK_STEER_GAIN    = 0.60
AREA_SLOW_THRES    = 0.20
AREA_PEAK_THRES    = 0.20
CONFIRM_FRAMES     = 10
STOP_DURATION      = 1.1
COLOR_MEMORY_TIME  = 0.40
STEER_SMOOTH_ALPHA = 0.45

TARGETS = ['red', 'yellow', 'blue']

# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR 기본 파라미터  (v4 동일)
# ─────────────────────────────────────────────────────────────────────────────
BIN_DEG   = 4.0
N_BINS    = int(360 / BIN_DEG)
EMERGENCY = 200.0   # 전방 긴급 정지 거리 (mm)
VELO_DOWN = 400.0   # 속도 감속 시작 거리 (mm)
DETECT    = 500.0   # LiDAR 유효 감지 거리 = APF 척력 유효 반경 D0

# ─────────────────────────────────────────────────────────────────────────────
#  APF 파라미터  (튜닝 포인트)
# ─────────────────────────────────────────────────────────────────────────────
# 좌표계: 로봇 기준 +x=우측, +y=전방
# LiDAR 각도: CW(시계방향) 기준, 0°=전방, 90°=우측, 270°=좌측

K_ATTR      = 1.0      # 인력 계수 (카메라 기반)  — 기준값, 다른 파라미터는 이에 상대적으로 조정
K_REP       = 2.0e7    # 척력 계수 (LiDAR 기반)   — 단위: mm²  (Khatib 공식 기반)
D0          = DETECT   # 척력 유효 반경 (mm): D0 이내 장애물만 반발력 발생

APF_SPD_MAX  = 0.75   # APF 최대 전진 속도
APF_SPD_MIN  = 0.20   # 전방 합력이 양수일 때 최소 보장 속도
APF_STR_GAIN = 1.3    # 합력 방향각(rad) → steer 변환 게인

# 로컬 미니멈 탈출 (U자형 장애물 포위 등)
STUCK_TIMEOUT = 3.0   # 이 시간(s) 이상 속도 0 지속 시 탈출 회전 실행
STUCK_ROT_SPD = 0.30  # 탈출 제자리 회전 속도

# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR 공유 상태  — 원시 히스토그램 저장 (APF 계산은 메인 루프에서 수행)
# ─────────────────────────────────────────────────────────────────────────────
_lidar_lock  = threading.Lock()
_lidar_state = {
    'has_data' : False,
    'hist'     : [9999.0] * N_BINS,
    'has_pt'   : [False]  * N_BINS,
    'emg_near' : 9999.0,
}


# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR 헬퍼
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


# ─────────────────────────────────────────────────────────────────────────────
#  APF 핵심 함수
# ─────────────────────────────────────────────────────────────────────────────

def _compute_apf(hist, has_pt, cam_offset):
    """
    인력(카메라) + 척력(LiDAR) 합력으로 조향·속도 계산.

    cam_offset : 카메라 수평 오프셋 (-1~+1, 양수=우측).
                 None 이면 카메라 탐지 없음 → 약한 전방 바이어스만.

    반환: (steer, speed, emg_near)
      steer    : -1.0(좌) ~ +1.0(우)
      speed    : 0.0 ~ APF_SPD_MAX
      emg_near : 전방 80° 이내 최근접 거리 (mm)

    [물리 공식 — Khatib 1986]
    F_rep_i = K_REP × (1/d_i - 1/D0) / d_i²  when d_i < D0
    방향: 장애물 i에서 로봇 방향으로 (장애물 반대)
    """
    # ── ① 인력 벡터 ──────────────────────────────────────────────────────────
    if cam_offset is not None:
        fx_a = K_ATTR * float(cam_offset)   # 종이 측면 오차 → 횡방향 인력
        fy_a = K_ATTR                        # 항상 전방으로 당김
    else:
        fx_a = 0.0
        fy_a = K_ATTR * 0.3                 # 종이 소실 시: 약한 전방 바이어스

    # ── ② 척력 벡터 합산 ─────────────────────────────────────────────────────
    fx_r = 0.0
    fy_r = 0.0
    emg  = 9999.0

    for i in range(N_BINS):
        if not has_pt[i]:
            continue
        d = hist[i]
        if d < emg:
            emg = d
        if d >= D0:
            continue

        theta = math.radians(i * BIN_DEG)
        # CW 각도 → 로봇 기준 단위벡터
        # 0°=전방(+y), 90°=우측(+x), 180°=후방(-y), 270°=좌측(-x)
        ox = math.sin(theta)   # 장애물 x방향 성분
        oy = math.cos(theta)   # 장애물 y방향 성분

        mag = K_REP * (1.0 / d - 1.0 / D0) / (d * d)  # Khatib 공식
        fx_r -= mag * ox   # 장애물 반대 방향으로 밀어냄
        fy_r -= mag * oy

    # ── ③ 합력 ───────────────────────────────────────────────────────────────
    fx = fx_a + fx_r
    fy = fy_a + fy_r

    # ── ④ 합력 → 조향 / 속도 ─────────────────────────────────────────────────
    if abs(fx) < 1e-9 and abs(fy) < 1e-9:
        return 0.0, 0.0, emg

    # 합력 방향각: atan2(x, y) → 전방=0, 우측=+, 좌측=-
    angle = math.atan2(fx, max(fy, 1e-6))
    steer = float(np.clip(angle * APF_STR_GAIN, -MAX_STEER, MAX_STEER))

    # 전방 합력 비율 (인력 기준으로 정규화)
    fwd_ratio = fy / (K_ATTR + 1e-9)
    if fwd_ratio <= 0.0:
        speed = 0.0   # 전방이 막힘 → 정지 후 조향으로 탈출
    else:
        speed = float(np.clip(fwd_ratio * APF_SPD_MAX, APF_SPD_MIN, APF_SPD_MAX))

    return steer, speed, emg


# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR 백그라운드 스레드  (히스토그램만 저장)
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
                emg = _nearest(hist, has_pt, 0.0, arc_half=80)
                with _lidar_lock:
                    _lidar_state['has_data'] = True
                    _lidar_state['hist']     = hist
                    _lidar_state['has_pt']   = has_pt
                    _lidar_state['emg_near'] = emg
                scan_buf = []
        except Exception as e:
            print(f"[LIDAR] {e}")
            scan_buf = []


def _lidar_read():
    with _lidar_lock:
        return dict(_lidar_state)


def _speed_limit(cam_speed: float) -> float:
    """CLOSE-FWD 잠금 구간용 LiDAR 속도 제한 (APF 미사용 구간)."""
    ls = _lidar_read()
    if not ls['has_data']:
        return cam_speed
    emg = ls['emg_near']
    if emg <= EMERGENCY:
        return 0.0
    if emg < VELO_DOWN:
        rt = max(0.0, min(1.0, (VELO_DOWN - emg) / (VELO_DOWN - EMERGENCY)))
        return cam_speed * (1.0 - rt * 0.45)
    return cam_speed


# ─────────────────────────────────────────────────────────────────────────────
#  색지 탐지  (v4 동일)
# ─────────────────────────────────────────────────────────────────────────────

def _get_mask(hsv, color: str):
    if color == 'red':    return get_red_mask(hsv)
    if color == 'yellow': return get_yellow_mask(hsv)
    return get_blue_mask(hsv)


def _detect_paper(frame, color: str):
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
    return {
        'cx'      : M['m10'] / M['m00'],
        'cy'      : M['m01'] / M['m00'],
        'area_r'  : area_r,
        'contour' : cnt,
    }


def _weak_detect(frame, color: str):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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

    # ── 상태 변수 ─────────────────────────────────────────────────────────────
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

    stuck_since = None   # 로컬 미니멈 진입 시각

    print("=" * 60)
    print("  Camera_v5_APF  |  Artificial Potential Field")
    print(f"  K_ATTR={K_ATTR}  K_REP={K_REP:.1e}  D0={D0:.0f}mm")
    print(f"  목표: RED → YELLOW → BLUE")
    print("=" * 60)

    while True:
        ret, raw = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        undist = cv2.remap(raw, map1, map2, cv2.INTER_LINEAR) if map1 is not None else raw
        vis    = undist.copy()

        # ── DONE ──────────────────────────────────────────────────────────────
        if state == 'DONE':
            ser.write(b"S\n")
            cv2.putText(vis, "MISSION COMPLETE", (40, CAM_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow('Robot View', vis)
            cv2.waitKey(1); time.sleep(0.1)
            continue

        ls    = _lidar_read()
        color = TARGETS[target_idx]

        # ── STOP 대기 ─────────────────────────────────────────────────────────
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
                    stuck_since            = None
                    last_seen              = time.time()
                    print(f"  ✅ {color.upper()} 완료 → {TARGETS[target_idx].upper()}")
                else:
                    state = 'DONE'
                    print("  ✅ 전체 미션 완료!")
            continue

        # ── SEEK ──────────────────────────────────────────────────────────────
        det = _detect_paper(undist, color)

        # ① 강탐지 ─────────────────────────────────────────────────────────────
        if det is not None:
            last_seen = time.time()
            stuck_since = None
            cx, cy  = det['cx'], det['cy']
            area_r  = det['area_r']
            offset  = _offset(cx)

            if area_r >= AREA_PEAK_THRES:
                area_peak_seen = True
                peak_area_r    = max(peak_area_r, area_r)

            # ── CLOSE-FWD: 최종 접근 조향 잠금 (v4 동일) ──────────────────────
            if area_r >= AREA_SLOW_THRES or (approach_steer_locked and area_peak_seen):
                if not approach_steer_locked:
                    if prev_area_r > 0.05:
                        locked_approach_steer = float(np.clip(offset * 0.25, -0.35, 0.35))
                    else:
                        locked_approach_steer = 0.0
                    approach_steer_locked = True
                steer = locked_approach_steer
                speed = _speed_limit(SPEED_NEAR)
                src   = "approach" if prev_area_r > 0.05 else "sudden"
                log_msg = f"CLOSE-FWD(lock/{src}) st={steer:+.2f} area={area_r:.2f}"
                cv2.putText(vis, f"CLOSE-FWD A={area_r:.2f} lock={steer:+.2f}",
                            (CAM_W // 2 - 180, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)

            # ── 원거리: APF 통합 제어 ─────────────────────────────────────────
            else:
                approach_steer_locked = False
                if ls['has_data']:
                    steer, speed, _ = _compute_apf(ls['hist'], ls['has_pt'], offset)
                    mode = "APF"
                else:
                    # LiDAR 없으면 단순 비례 조향
                    steer = float(np.clip(offset * 1.5, -MAX_STEER, MAX_STEER))
                    speed = SPEED_NEAR
                    mode  = "CAM"
                arrow   = '→' if steer > 0.1 else ('←' if steer < -0.1 else '↑')
                log_msg = f"{mode}{arrow} off={offset:+.2f} st={steer:+.2f} area={area_r:.2f}"
                cv2.putText(vis, f"{mode} {arrow} A={area_r:.2f} st={steer:+.2f}",
                            (CAM_W // 2 - 160, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

            x, y, w, h = cv2.boundingRect(det['contour'])
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 180, 255), 2)
            _draw_ctr(vis, int(cx), int(cy), (0, 180, 255))

            smoothed_steer = STEER_SMOOTH_ALPHA * steer + (1.0 - STEER_SMOOTH_ALPHA) * smoothed_steer
            last_steer     = smoothed_steer
            prev_area_r    = area_r
            if speed > 0:
                ser.write(f"F {smoothed_steer:.2f} {speed:.2f}\n".encode())
            else:
                ser.write(b"S\n")
            print(f"  [SEEK] {color.upper()} {log_msg} spd={speed:.2f}")

        # ② 피크 후 미탐지 → ENTERING ──────────────────────────────────────────
        elif area_peak_seen:
            prev_area_r = 0.0
            stuck_since = None
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
                nudge_spd = _speed_limit(NUDGE_SPEED)
                ser.write(f"F 0.00 {nudge_spd:.2f}\n".encode() if nudge_spd > 0 else b"S\n")
                cv2.putText(vis, f"INVISIBLE cnt:{on_zone_count}/{CONFIRM_FRAMES} nudge={nudge_spd:.2f}",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)
                print(f"  [INVIS] {color.upper()} pk={peak_area_r:.2f} cnt={on_zone_count}")
                if on_zone_count >= CONFIRM_FRAMES:
                    state = 'STOP'; stop_start = time.time()
                    ser.write(b"S\n")
                    print(f"  🎯 {color.upper()} 도달! peak={peak_area_r:.2f}")
                    cv2.imshow('Robot View', vis)
                    cv2.waitKey(1)
                    continue

        # ③ 미탐지 → APF 탐색 ──────────────────────────────────────────────────
        else:
            prev_area_r   = 0.0
            on_zone_count = max(0, on_zone_count - 1)

            # 약탐지: 색상 방향 인식
            w = _weak_detect(undist, color)
            if w is not None:
                cnt_w, wcx, wcy = w
                weak_offset    = _offset(wcx)
                last_seen      = time.time()
                stuck_since    = None
                # 약탐지도 APF에 offset 반영
                if ls['has_data']:
                    w_steer, w_speed, _ = _compute_apf(
                        ls['hist'], ls['has_pt'], weak_offset * 0.6)
                else:
                    w_steer = float(np.clip(weak_offset * 1.5, -MAX_STEER, MAX_STEER))
                    w_speed = PIVOT_SPEED
                steer_cmd      = STEER_SMOOTH_ALPHA * w_steer + (1.0 - STEER_SMOOTH_ALPHA) * smoothed_steer
                smoothed_steer = steer_cmd
                ser.write(f"F {steer_cmd:.2f} {w_speed:.2f}\n".encode())
                cv2.drawContours(vis, [cnt_w], -1, (180, 180, 0), 1)
                arrow = '→' if w_steer > 0 else '←'
                cv2.putText(vis, f"WEAK-APF{arrow} off={weak_offset:+.2f}",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 0), 2)
                print(f"  [WEAK] {color.upper()} APF{arrow} off={weak_offset:+.2f}")

            else:
                elapsed = time.time() - last_seen

                # 단기 기억: 마지막 조향 방향 유지
                if elapsed < COLOR_MEMORY_TIME:
                    steer_cmd      = STEER_SMOOTH_ALPHA * last_steer + (1.0 - STEER_SMOOTH_ALPHA) * smoothed_steer
                    smoothed_steer = steer_cmd
                    mem_speed      = _speed_limit(SPEED_NEAR * 0.7)
                    ser.write(f"F {steer_cmd:.2f} {mem_speed:.2f}\n".encode()
                              if mem_speed > 0 else b"S\n")
                    cv2.putText(vis, f"MEM {elapsed:.2f}s st={steer_cmd:+.2f}",
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 220, 100), 2)
                    print(f"  [MEM] {color.upper()} t={elapsed:.2f}s st={steer_cmd:+.2f}")

                # 완전 소실: APF만으로 탐색 (인력=전방 바이어스, 척력=장애물)
                else:
                    if ls['has_data']:
                        apf_steer, apf_speed, apf_emg = _compute_apf(
                            ls['hist'], ls['has_pt'], None)
                    else:
                        apf_steer, apf_speed = 0.0, PIVOT_SPEED
                        apf_emg = 9999.0

                    # ── 로컬 미니멈 탈출 ───────────────────────────────────────
                    if apf_speed <= 0.0:
                        if stuck_since is None:
                            stuck_since = time.time()
                        elif time.time() - stuck_since > STUCK_TIMEOUT:
                            ser.write(f"T {STUCK_ROT_SPD:.2f}\n".encode())
                            cv2.putText(vis, "STUCK ESCAPE (rotate)",
                                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.65, (0, 0, 255), 2)
                            print(f"  [STUCK] {color.upper()} 로컬 미니멈 탈출 회전")
                            cv2.imshow('Robot View', vis)
                            cv2.waitKey(1)
                            continue
                    else:
                        stuck_since = None

                    smoothed_steer = STEER_SMOOTH_ALPHA * apf_steer + (1.0 - STEER_SMOOTH_ALPHA) * smoothed_steer
                    ser.write(f"F {smoothed_steer:.2f} {apf_speed:.2f}\n".encode()
                              if apf_speed > 0 else b"S\n")
                    cv2.putText(vis, f"APF-SEEK st={smoothed_steer:+.2f} spd={apf_speed:.2f} emg={apf_emg:.0f}mm",
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 2)
                    print(f"  [APF] {color.upper()} st={smoothed_steer:+.2f} spd={apf_speed:.2f} emg={apf_emg:.0f}mm")

        # ── 공통 HUD ──────────────────────────────────────────────────────────
        emg_txt = f" EMG:{ls['emg_near']:.0f}mm" if ls['has_data'] else " NO_LIDAR"
        cv2.putText(vis,
                    f"APF|{state}|{color.upper()}|cnt:{on_zone_count}/{CONFIRM_FRAMES}{emg_txt}",
                    (10, CAM_H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 2)

        cv2.imshow('Robot View', vis)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            print("  [종료] q 입력")
            break

    ser.write(b"S\n")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
