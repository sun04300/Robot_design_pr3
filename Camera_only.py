"""
[파일] camera_only_robot.py
[목적] LiDAR 없이 카메라만으로 색 인식 → 주행 → 색지 위 1초 정지 → 다음 색

[내일 목표 전용 단순 버전]
  - LiDAR / VFH / 스레딩 / 장애물 회피 전부 제거
  - 미션: RED → (1초 정지) → YELLOW → (1초 정지) → BLUE → (완전 정지)
  - 카메라 캘리브레이션은 그대로 적용 (camera_corrector.py 재사용)

[필요 파일 (같은 폴더에)]
  camera_only_robot.py   ← 이 파일
  camera_corrector.py    ← 왜곡 보정 클래스
  camera_calibration.pkl ← 캘리브레이션 결과 (없으면 보정 없이 동작)

[아두이노 명령 프로토콜] — Robo_move.ino 와 동일
  F {steer:.2f} {speed:.2f}\n  → 전진 (steer: -1~+1, speed: 0~1)
  T {dir:.2f}\n                → 제자리 피벗 회전 (+1=우, -1=좌)
  S\n                          → 즉시 정지  ← 아두이노에 추가 필요!
      loop()에  else if (cmd == 'S') { brakeMotors(); }  한 줄 추가

[조향 부호] 배선 좌우 반대를 코드로 보정한 상태이므로 그대로 사용
"""

import os
import serial
import time
import cv2
import numpy as np

from camera_corrector import CameraCorrector


# ─────────────────────────────────────────────────────────────────────────────
#  설정
# ─────────────────────────────────────────────────────────────────────────────
PORT_ARDU  = "/dev/ttyS0"     # 아두이노 포트
CAM_INDEX  = 0
CAM_W, CAM_H = 640, 480
CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_calibration.pkl")

MAX_STEER  = 1.0              # 조향 한계

# ── HSV 색상 범위 (실제 색지 사진 측정값) ──────────────────────────────────
# [수정] 빨강 V 상한 230→255: 실험실(어두운 바닥)에서 색지 반사로 V가 246까지
#        올라가 230 상한에 잘려 검출이 0%가 되는 문제 해결. 시험장도 V 175라 안전.
RED_LO1, RED_HI1 = np.array([0,   100, 80]), np.array([8,   255, 255])
RED_LO2, RED_HI2 = np.array([155,  80, 80]), np.array([179, 255, 255])

# [수정] YELLOW: 갈색 박스 오인식 방지 → V하한 100→150, H상한 30→28, H하한 18→20
#   근거: 노란종이 V=160~239 vs 갈색박스 V=93~149 → V하한 150으로 분리
#         갈색박스 모서리 주황반사(H=29)를 H상한 28로 차단
YELLOW_LO, YELLOW_HI = np.array([20, 100, 150]), np.array([28, 255, 255])

# [수정] BLUE: 파란 박스(Double A) 인쇄면 오인식 방지 → S상한 220→160, V하한 60→110
#   근거: 파란종이 S=113~141 vs 박스인쇄면 S=160~218 → S상한 160으로 분리
#         (종이=연한파랑 저채도, 박스잉크=진한파랑 고채도)
BLUE_LO, BLUE_HI     = np.array([100, 90, 110]),  np.array([120, 160, 240])

BLUE_PAPER_RATIO = 0.025      # 파란 종이 vs 박스 면적 구분 (화면의 2.5% 이상 = 종이)

_K5 = np.ones((5, 5), np.uint8)
_K9 = np.ones((9, 9), np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  색상 마스크 / 탐지
# ─────────────────────────────────────────────────────────────────────────────

def _clean(mask):
    """노이즈 제거: OPEN(잡음) → CLOSE(구멍 메우기)."""
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _K5)
    m = cv2.morphologyEx(m,    cv2.MORPH_CLOSE, _K9)
    return m


def get_color_mask(hsv, color):
    """지정 색상의 마스크 반환."""
    if color == 'red':
        m = cv2.bitwise_or(cv2.inRange(hsv, RED_LO1, RED_HI1),
                           cv2.inRange(hsv, RED_LO2, RED_HI2))
    elif color == 'yellow':
        m = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
    else:  # blue
        m = cv2.inRange(hsv, BLUE_LO, BLUE_HI)
    return _clean(m)


def detect_color(hsv, color, frame_w, frame_h, min_area=1000):
    """
    지정 색상의 가장 큰 영역을 탐지.

    Returns dict:
      found  : 탐지 여부
      cx, cy : 무게중심 좌표
      area   : 면적 (px)
      offset : 좌우 오프셋 (-1.0 좌 ~ +1.0 우, 0=중앙)
    """
    mask = get_color_mask(hsv, color)

    # 파란 종이는 면적 필터로 박스 제거
    if color == 'blue':
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_px = frame_w * frame_h
        paper = [c for c in cnts if cv2.contourArea(c) > total_px * BLUE_PAPER_RATIO]
        mask = np.zeros_like(mask)
        if paper:
            cv2.drawContours(mask, paper, -1, 255, -1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts    = [c for c in cnts if cv2.contourArea(c) > min_area]
    if not cnts:
        return {'found': False, 'cx': None, 'cy': None, 'area': 0.0, 'offset': None}

    best = max(cnts, key=cv2.contourArea)
    M    = cv2.moments(best)
    if M['m00'] == 0:
        return {'found': False, 'cx': None, 'cy': None, 'area': 0.0, 'offset': None}

    cx     = int(M['m10'] / M['m00'])
    cy     = int(M['m01'] / M['m00'])
    area   = cv2.contourArea(best)
    offset = (cx - frame_w / 2) / (frame_w / 2)
    return {'found': True, 'cx': cx, 'cy': cy, 'area': area, 'offset': offset}


def bottom_fill_ratio(hsv, color):
    """
    화면 하단 40% ROI 에서 타겟 색이 차지하는 비율.
    → 색지에 올라탔는지 판단하는 핵심 지표.
    """
    h = hsv.shape[0]
    roi = hsv[int(h * 0.60):, :]
    mask = get_color_mask(roi, color)
    return np.count_nonzero(mask) / (roi.shape[0] * roi.shape[1])


# ─────────────────────────────────────────────────────────────────────────────
#  메인 로직
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 시리얼 / 카메라 초기화 ─────────────────────────────────────────────
    ser = serial.Serial(PORT_ARDU, 460800, timeout=1)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── 캘리브레이션 보정 모듈 ─────────────────────────────────────────────
    corrector = CameraCorrector(calib_file=CALIB_FILE, resolution=(actual_w, actual_h))
    if not corrector.is_ready:
        print("[경고] 캘리브레이션 미적용. 보정 없이 진행합니다.")

    # ── 미션 / 상태 파라미터 ──────────────────────────────────────────────
    TARGETS = ['red', 'yellow', 'blue']   # 순서 고정
    target_idx = 0

    # 상태: SEEK(탐색·접근) / STOP(정지 대기) / DONE(완료)
    state = 'SEEK'

    # 조정 가능한 임계값 ─ 내일 현장에서 이 숫자들을 맞추면 됨
    ON_ZONE_FILL    = 0.50    # 하단 ROI 색 채움 비율 (이 값 이상이면 색지 위)
    ON_ZONE_OFFSET  = 0.35    # 좌우 정렬 허용 오차
    CONFIRM_FRAMES  = 10      # 연속 N 프레임 충족 시 도달 인정 (오탐 방지)
    STOP_DURATION   = 1.0     # 정지 시간 (초) — 요구사항: 1초 이상

    STEER_GAIN      = 0.80    # 조향 게인 (offset → steer)
    SPEED_FAR       = 0.55    # 멀리 있을 때 접근 속도
    SPEED_NEAR      = 0.35    # 가까이 왔을 때 감속 속도
    AREA_SLOW       = 0.08    # 화면의 8% 이상 차지하면 감속
    SEARCH_PIVOT    = 0.50    # 타겟 안 보일 때 피벗 회전 방향/세기 (+우)
    SEARCH_TIMEOUT  = 1.5     # 타겟 미탐지 N초 후 피벗 재탐색 시작

    on_zone_count = 0
    stop_start    = None
    last_seen     = time.time()
    last_offset   = 0.0

    # ── 종료 시 정지 ───────────────────────────────────────────────────────
    import atexit
    def _cleanup():
        try:
            ser.write(b"S\n")
            time.sleep(0.1)
            cap.release()
            ser.close()
        except Exception:
            pass
    atexit.register(_cleanup)

    print("=" * 60)
    print("  카메라 단독 색상 추적 주행 시작")
    print("  목표 순서: RED → YELLOW → BLUE")
    print("  종료: Ctrl+C")
    print("=" * 60)

    # ── 메인 루프 (매 카메라 프레임마다 판단) ────────────────────────────
    while True:
        ret, raw = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # (1) 왜곡 보정
        frame = corrector.update_frame(raw, crop_roi=True, debug=False)
        fh, fw = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ── 미션 완료 상태 ──────────────────────────────────────────────
        if state == 'DONE':
            ser.write(b"S\n")
            print("  ✅ 미션 완료 — 완전 정지 유지 중")
            time.sleep(0.1)
            continue

        color = TARGETS[target_idx]   # 현재 목표 색

        # ── 정지 대기 상태 (1초 카운트) ────────────────────────────────
        if state == 'STOP':
            ser.write(b"S\n")   # 매 프레임 정지 명령 재전송 (확실한 정지 유지)
            elapsed = time.time() - stop_start
            remain  = max(0.0, STOP_DURATION - elapsed)
            print(f"  [STOP_{color.upper()}] 정지 중... 남은 시간 {remain:.2f}초")

            if elapsed >= STOP_DURATION:
                # 1초 경과 → 다음 색으로
                if target_idx < len(TARGETS) - 1:
                    target_idx += 1
                    state = 'SEEK'
                    on_zone_count = 0
                    last_seen = time.time()
                    print(f"  ✅ {color.upper()} 완료! → 다음 목표: {TARGETS[target_idx].upper()}")
                else:
                    state = 'DONE'
                    print(f"  🔵 {color.upper()} 완료! 모든 미션 종료")
            continue

        # ── 탐색·접근 상태 (SEEK) ──────────────────────────────────────
        # min_area=1500: 갈색 박스 잔여 오탐(최대 689px)을 확실히 차단.
        #   실제 색종이는 4000px 이상이라 정탐에는 영향 없음.
        det = detect_color(hsv, color, fw, fh, min_area=1500)

        # (2) 색지 위 판정
        if det['found']:
            last_seen   = time.time()
            last_offset = det['offset']

            fill = bottom_fill_ratio(hsv, color)
            on_zone = (fill >= ON_ZONE_FILL) and (abs(det['offset']) < ON_ZONE_OFFSET)

            if on_zone:
                on_zone_count += 1
            else:
                on_zone_count = max(0, on_zone_count - 2)

            # 연속 프레임 충족 → 도달 인정 → 정지 시작
            if on_zone_count >= CONFIRM_FRAMES:
                state = 'STOP'
                stop_start = time.time()
                ser.write(b"S\n")
                print(f"  🎯 {color.upper()} 도달! 1초 정지 시작 "
                      f"(fill={fill:.2f} offset={det['offset']:+.2f})")
                continue

            # (3) 아직 도달 전 → 색지 향해 접근
            offset     = det['offset']
            area_ratio = det['area'] / (fw * fh)
            steer = max(-MAX_STEER, min(MAX_STEER, offset * STEER_GAIN))
            speed = SPEED_NEAR if area_ratio > AREA_SLOW else SPEED_FAR

            ser.write(f"F {steer:.2f} {speed:.2f}\n".encode())
            print(f"  [SEEK_{color.upper()}] offset={offset:+.2f} area={area_ratio*100:.1f}% "
                  f"fill={fill:.2f} cnt={on_zone_count} → F {steer:.2f} {speed:.2f}")

        else:
            # (4) 타겟 안 보임 → 잠시 후 제자리 피벗 재탐색
            on_zone_count = 0
            elapsed = time.time() - last_seen
            if elapsed > SEARCH_TIMEOUT:
                # 마지막으로 본 방향 쪽으로 회전 (놓친 방향으로 되돌아감)
                pivot = SEARCH_PIVOT if last_offset >= 0 else -SEARCH_PIVOT
                ser.write(f"T {pivot:.2f}\n".encode())
                print(f"  [SEARCH_{color.upper()}] 미탐지 {elapsed:.1f}초 → 피벗 dir={pivot:+.2f}")
            else:
                # 잠깐의 미탐지는 정지로 버팀 (떨림 방지)
                ser.write(b"S\n")
                print(f"  [SEARCH_{color.upper()}] 타겟 일시 미탐지 ({elapsed:.1f}초) → 정지 대기")


if __name__ == '__main__':
    main()