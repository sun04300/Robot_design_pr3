"""
[파일] color_detector.py
[목적] 실제 경기 환경(형광등 조명)에서 촬영한 색지 샘플을 기반으로
       RED / YELLOW / BLUE 종이를 인식하고 Double A 박스와 구별하는 모듈.

[HSV 측정값 요약 (실제 사진 기반)]
  RED    : H mean=172.0  S mean=171.6  V mean=175.6  (핑크-마젠타 계열!)
  YELLOW : H mean=24.4   S mean=149.4  V mean=214.5
  BLUE   : H mean=114.9  S mean=143.3  V mean=143.0
  BOX    : H mean=109.7  S mean=138.4  V mean=103.9  ← 종이보다 어두움

[파란 종이 vs 박스 구별 전략]
  1차: V(밝기) 차이 → 종이는 밝음, 박스는 어두움
  2차: 컨투어 면적 → 종이(전체 화면 3% 이상) vs 박스(더 작음)
"""

import cv2
import numpy as np
import time
import datetime
import pickle
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CALIB = os.path.join(_SCRIPT_DIR, 'camera_calibration.pkl')


# ────────────────────────────────────────────────────
#  HSV 범위 상수 (실제 환경 측정값 기반)
#  경기 당일 조명이 다르면 HSV_TUNER()로 재조정!
# ────────────────────────────────────────────────────

# RED: 핑크-마젠타 계열이므로 H 155~179 + H 0~8 두 범위 합산 필수
RED_LOWER1  = np.array([  0, 100, 130])  # 순수 빨강 (H 저주파 끝)
RED_UPPER1  = np.array([ 10, 255, 255])
RED_LOWER2  = np.array([150, 100,  80])  # 핑크-마젠타 (H 고주파 끝)
RED_UPPER2  = np.array([179, 255, 255])

# YELLOW: 종이 실측 H≈18-21, S≈90-150. 현재 바닥(핑크타일)은 H≈15, S≈51로
#  H_min=18 + S_min=85 조합으로 이중 차단됨.
#  이전 H_min=22 제약은 다른 환경 기준 → 현재 환경에서는 H_min=18이 안전.
YELLOW_LOWER = np.array([ 18,  85, 100])
YELLOW_UPPER = np.array([ 35, 255, 255])

# BLUE: Kobuki 박스(V≈104)와 파란 종이(V≈143) 구별 → V_min=120 핵심
#  박스: H≈110, S≈138, V≈104 → V<120으로 차단
#  종이: H≈115, S≈143, V≈143 → V>120으로 통과
#  A4 그림자·인쇄물(V<120 또는 V>200)도 이 범위에서 자연 차단됨
BLUE_LOWER   = np.array([ 95, 110, 120])
BLUE_UPPER   = np.array([135, 255, 200])

# 노이즈 제거용 커널
_K5  = np.ones(( 5,  5), np.uint8)
_K9  = np.ones(( 9,  9), np.uint8)
_K15 = np.ones((15, 15), np.uint8)  # 노란 종이 CLOSE 전용


# ────────────────────────────────────────────────────
#  캘리브레이션 로드 헬퍼
# ────────────────────────────────────────────────────

def load_calibration(pkl_path: str = _DEFAULT_CALIB):
    """
    cali.py 가 생성한 camera_calibration.pkl 을 읽어
    (camera_matrix, dist_coeffs, calib_resolution) 을 반환.

    파일이 없거나 손상된 경우 None, None, None 을 반환하고
    경고만 출력함 → 캘리브 없이도 색상 인식은 계속 동작.
    """
    if not os.path.exists(pkl_path):
        print(f"[CALIB] '{pkl_path}' 없음 → 왜곡 보정 비활성화")
        return None, None, None
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        mtx  = data['camera_matrix']
        dist = data['dist_coeffs']
        res  = data.get('resolution', None)   # (width, height) or None
        print(f"[CALIB] 로드 완료 → 해상도={res}  RMS={data.get('rms_error', '?'):.4f}px")
        return mtx, dist, res
    except Exception as e:
        print(f"[CALIB] 로드 실패({e}) → 왜곡 보정 비활성화")
        return None, None, None


# ────────────────────────────────────────────────────
#  핵심 마스크 생성 함수
# ────────────────────────────────────────────────────

def get_red_mask(hsv: np.ndarray) -> np.ndarray:
    """
    빨간(핑크-마젠타) 종이 마스크 반환.
    HSV의 H 채널 wrap-around 특성 때문에 두 범위를 OR 연산으로 합침.
    """
    m1  = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    m2  = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    raw = cv2.bitwise_or(m1, m2)
    # 모폴로지: OPEN(잡음 제거) → CLOSE(내부 구멍 메우기)
    out = cv2.morphologyEx(raw, cv2.MORPH_OPEN,  _K5)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, _K9)
    return out


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def get_yellow_mask(hsv: np.ndarray) -> np.ndarray:
    """
    노란 종이 마스크 반환.
    빛번짐 대응: V 채널에 CLAHE를 적용해 과포화된 밝기를 정규화한 뒤
    기본 범위(YELLOW_LOWER/UPPER)로 검출. 정규화 전/후 마스크를 OR해
    글레어 중심부와 외곽 모두 포착.
    """
    # CLAHE로 V채널 정규화 → 빛번짐 중심부 채도 복원
    hsv_eq = hsv.copy()
    hsv_eq[:, :, 2] = _CLAHE.apply(hsv[:, :, 2])

    raw1 = cv2.inRange(hsv,    YELLOW_LOWER, YELLOW_UPPER)  # 원본
    raw2 = cv2.inRange(hsv_eq, YELLOW_LOWER, YELLOW_UPPER)  # CLAHE 보정본
    raw  = cv2.bitwise_or(raw1, raw2)
    out  = cv2.morphologyEx(raw, cv2.MORPH_OPEN,  _K5)
    out  = cv2.morphologyEx(out, cv2.MORPH_CLOSE, _K15)  # 9→15: 더 큰 구멍 메움
    return out


def get_blue_mask(hsv: np.ndarray) -> np.ndarray:
    """파란색 전체(종이 + 박스) 마스크 반환."""
    raw = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    out = cv2.morphologyEx(raw, cv2.MORPH_OPEN,  _K5)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, _K9)
    return out


# ────────────────────────────────────────────────────
#  컨투어 분석 함수
# ────────────────────────────────────────────────────

def get_largest_contour(mask: np.ndarray, min_area: int = 1000):
    """마스크에서 가장 큰 컨투어 반환. 없으면 None."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > min_area]
    return max(cnts, key=cv2.contourArea) if cnts else None


def contour_center(contour) -> tuple:
    """컨투어 무게중심 (cx, cy) 반환."""
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return None
    return int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])




# ────────────────────────────────────────────────────
#  메인 컬러 디텍터 클래스
# ────────────────────────────────────────────────────

class ColorDetector:
    """
    프레임 1장을 받아 RED/YELLOW/BLUE 종이의 존재 여부와
    화면 내 위치(무게중심 x 오프셋)를 반환하는 클래스.

    사용 예시:
        detector = ColorDetector(frame_w=640, frame_h=480)
        result = detector.detect(frame)
        # result['red']['found'], result['red']['cx'], result['red']['area']

    캘리브레이션 적용 예시:
        mtx, dist, _ = load_calibration('camera_calibration.pkl')
        detector = ColorDetector(frame_w=640, frame_h=480,
                                 camera_matrix=mtx, dist_coeffs=dist)
    """

    def __init__(self, frame_w: int = 640, frame_h: int = 480,
                 min_area: int = 1000,
                 camera_matrix=None, dist_coeffs=None):
        self.fw       = frame_w
        self.fh       = frame_h
        self.min_area = min_area

        # ── 캘리브레이션 파라미터 저장 ──────────────────────
        self.mtx  = camera_matrix   # 3x3 카메라 내부 행렬
        self.dist = dist_coeffs     # 왜곡 계수 (1x5)

        # undistort 맵을 미리 계산해두면 매 프레임마다 재계산하지 않아도 됨
        # → 실시간 처리 속도 향상 (initUndistortRectifyMap 은 1회만 실행)
        if self.mtx is not None and self.dist is not None:
            # getOptimalNewCameraMatrix: alpha=1 → 왜곡 제거 후 유효 픽셀 최대 보존
            # alpha=0 으로 바꾸면 검은 테두리 없이 크롭됨 (필요 시 조정)
            new_mtx, roi = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist,
                (frame_w, frame_h), alpha=1,
                newImgSize=(frame_w, frame_h)
            )
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                self.mtx, self.dist, None, new_mtx,
                (frame_w, frame_h), cv2.CV_16SC2
            )
            self._roi     = roi        # (x, y, w, h) — 유효 영역 (참고용)
            self._new_mtx = new_mtx    # 보정 후 카메라 행렬 (참고용)
            print(f"[CALIB] undistort 맵 생성 완료  ROI={roi}")
        else:
            # 캘리브 없음 → 보정 맵을 None 으로 표시, 원본 프레임 그대로 사용
            self._map1 = self._map2 = None
            print("[CALIB] 카메라 행렬 없음 → 왜곡 보정 스킵")

    def detect(self, frame: np.ndarray) -> dict:
        """
        BGR 프레임을 입력받아 각 색상의 탐지 결과를 딕셔너리로 반환.

        내부 처리 순서:
          1) 캘리브 맵이 있으면 렌즈 왜곡 보정 (remap)
          2) BGR → HSV 변환
          3) 색상별 마스크 생성 → 컨투어 분석

        반환값 구조:
          {
            'red':    {'found': bool, 'cx': int|None, 'cy': int|None,
                       'area': float, 'offset': float,  # -1.0(좌) ~ +1.0(우)
                       'contour': ndarray|None},
            'yellow': { ... },
            'blue':   { ... },   # 종이만 (박스 제외)
            'undistorted': ndarray,  # 왜곡 보정된 프레임 (보정 없으면 원본)
          }
        offset = (cx - frame_center_x) / (frame_w / 2)
        → 0에 가까울수록 중앙, 음수=좌측, 양수=우측
        """
        # ── Step 1: 렌즈 왜곡 보정 ───────────────────────────
        # 캘리브 맵이 사전 계산된 경우에만 remap 수행
        # remap 은 initUndistortRectifyMap 이 만든 맵을 이용해
        # 각 출력 픽셀의 원본 좌표를 미리 알고 있으므로 매우 빠름
        if self._map1 is not None:
            undist = cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)
        else:
            undist = frame   # 보정 없이 그대로 사용

        # ── Step 2: BGR → HSV ────────────────────────────────
        hsv    = cv2.cvtColor(undist, cv2.COLOR_BGR2HSV)
        result = {'undistorted': undist}   # 보정 프레임도 반환 (draw_debug 에서 사용)

        # RED
        red_m = get_red_mask(hsv)
        red_c = get_largest_contour(red_m, self.min_area)
        result['red'] = self._make_entry(red_c)

        # YELLOW — 종이는 평평하므로 볼록껍질(convex hull)로 울퉁불퉁 제거
        yel_m = get_yellow_mask(hsv)
        yel_c = get_largest_contour(yel_m, self.min_area)
        if yel_c is not None:
            yel_c = cv2.convexHull(yel_c)
        result['yellow'] = self._make_entry(yel_c)

        # BLUE
        blu_m          = get_blue_mask(hsv)
        blu_c          = get_largest_contour(blu_m, self.min_area)
        result['blue'] = self._make_entry(blu_c)

        return result

    def _make_entry(self, contour) -> dict:
        if contour is None:
            return {'found': False, 'cx': None, 'cy': None,
                    'area': 0.0, 'offset': None, 'contour': None}
        center = contour_center(contour)
        cx, cy = center if center else (None, None)
        area   = cv2.contourArea(contour)
        offset = ((cx - self.fw / 2) / (self.fw / 2)) if cx is not None else None
        return {'found': True, 'cx': cx, 'cy': cy,
                'area': area, 'offset': offset, 'contour': contour}

    def draw_debug(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """
        탐지 결과를 프레임에 시각화하여 반환 (디버깅용).
        캘리브 보정이 적용된 경우 result['undistorted'] 를 베이스로 사용.
        """
        # 왜곡 보정된 프레임이 있으면 그 위에 오버레이 → 정확한 위치 표시
        base = result.get('undistorted', frame)
        vis  = base.copy()
        config = {
            'red'   : ((0, 0, 255),   'RED'),
            'yellow': ((0, 220, 220), 'YELLOW'),
            'blue'  : ((255, 120, 0), 'BLUE'),
        }
        for key, (color, label) in config.items():
            entry = result.get(key, {})
            if not entry.get('found'):
                continue
            c = entry['contour']
            cv2.drawContours(vis, [c], -1, color, 3)
            cx, cy   = entry['cx'], entry['cy']
            txt      = f"{label} x={cx} y={cy}"
            cv2.putText(vis, txt, (max(0, cx - 60), max(20, cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # 화면 중심선
        cv2.line(vis, (self.fw // 2, 0), (self.fw // 2, self.fh), (180, 180, 180), 1)
        # 캘리브 적용 여부를 좌상단에 표시
        calib_txt = "CALIB ON" if self._map1 is not None else "CALIB OFF"
        calib_col = (0, 255, 0) if self._map1 is not None else (0, 100, 255)
        cv2.putText(vis, calib_txt, (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, calib_col, 2)
        return vis


# ────────────────────────────────────────────────────
#  경기 당일 현장 HSV 튜닝 도구
#  실행: python3 color_detector.py --tune
# ────────────────────────────────────────────────────

def hsv_tuner(cam_index: int = 0):
    """
    트랙바로 HSV 범위를 실시간 조정하는 디버깅 도구.
    색이 잘 잡히지 않을 때 경기 현장에서 바로 실행하여 최적값을 찾을 것.

    조작법:
      q → 종료 후 현재 HSV 범위 출력
      s → 현재 값 터미널에 출력 (저장용)
    """
    cap = cv2.VideoCapture(cam_index)
    cv2.namedWindow('HSV Tuner', cv2.WINDOW_NORMAL)

    def nothing(_): pass
    # 초기값: RED 범위 2 (핑크-마젠타)
    params = [
        ('H_low',  155), ('H_high', 179),
        ('S_low',   80), ('S_high', 255),
        ('V_low',   80), ('V_high', 230),
    ]
    for name, val in params:
        cv2.createTrackbar(name, 'HSV Tuner', val, 179 if 'H' in name else 255, nothing)

    print("[HSV Tuner] 트랙바로 범위 조정 | q=종료 | s=값 출력")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lo = np.array([cv2.getTrackbarPos('H_low',  'HSV Tuner'),
                       cv2.getTrackbarPos('S_low',  'HSV Tuner'),
                       cv2.getTrackbarPos('V_low',  'HSV Tuner')])
        hi = np.array([cv2.getTrackbarPos('H_high', 'HSV Tuner'),
                       cv2.getTrackbarPos('S_high', 'HSV Tuner'),
                       cv2.getTrackbarPos('V_high', 'HSV Tuner')])

        mask   = cv2.inRange(hsv, lo, hi)
        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        result = cv2.bitwise_and(frame, frame, mask=mask)
        display = np.hstack([
            cv2.resize(frame,  (480, 360)),
            cv2.resize(result, (480, 360)),
        ])
        pct = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1]) * 100
        cv2.putText(display, f"Detected: {pct:.1f}%  lo={lo.tolist()} hi={hi.tolist()}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow('HSV Tuner', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            print(f"  LOWER = np.array({lo.tolist()})")
            print(f"  UPPER = np.array({hi.tolist()})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[최종값] LOWER={lo.tolist()}  UPPER={hi.tolist()}")


# ────────────────────────────────────────────────────
#  단독 실행 (카메라 테스트)
# ────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if '--tune' in sys.argv:
        # HSV 튜닝 모드
        cam = int(sys.argv[sys.argv.index('--tune') + 1]) if '--tune' in sys.argv and \
              sys.argv.index('--tune') + 1 < len(sys.argv) and \
              sys.argv[sys.argv.index('--tune') + 1].isdigit() else 0
        hsv_tuner(cam)
    else:
        # ── 캘리브레이션 파일 로드 ────────────────────────────
        # camera_calibration.pkl 이 없어도 동작하지만
        # 있으면 렌즈 왜곡 보정이 자동으로 활성화됨
        CALIB_PATH = _DEFAULT_CALIB
        mtx, dist, calib_res = load_calibration(CALIB_PATH)

        FRAME_W, FRAME_H = 640, 480

        # 캘리브 해상도 불일치 경고
        # → 해상도가 다르면 카메라 행렬의 fx/fy/cx/cy 가 맞지 않아 보정이 오히려 악화됨
        if calib_res is not None and calib_res != (FRAME_W, FRAME_H):
            print(f"[경고] 캘리브 해상도({calib_res}) ≠ 캡처 해상도({FRAME_W}x{FRAME_H})")
            print("       → 동일 해상도로 재촬영/재캘리브하거나 FRAME_W/H를 맞춰주세요.")
            print("       → 일단 보정을 비활성화하고 실행합니다.")
            mtx = dist = None   # 불일치 시 보정 비활성화

        # 라이브 탐지 테스트 모드
        cap = cv2.VideoCapture(0)  # 라즈베리파이: CAP_V4L2 사용 (인덱스 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

        # 캘리브 파라미터를 ColorDetector 에 전달 → __init__ 에서 undistort 맵 생성
        detector = ColorDetector(frame_w=FRAME_W, frame_h=FRAME_H,
                                 camera_matrix=mtx, dist_coeffs=dist)

        os.makedirs("./color", exist_ok=True)   # 캡처 저장 폴더 자동 생성
        print("[테스트] q=종료 | a=화면 캡처(vis) | 탐지 결과 터미널 출력")
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = detector.detect(frame)   # 내부에서 undistort → HSV → 마스크
            vis    = detector.draw_debug(frame, result)
            cv2.imshow('Color Detector', vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('a') and ret:
                filename = datetime.datetime.now().strftime("./color/capture_%Y%m%d_%H%M%S.png")
                cv2.imwrite(filename, vis)      # ← 원본 frame 대신 vis 저장 (오버레이 포함)
                count += 1
                detected = [c.upper() for c in ['red', 'yellow', 'blue'] if result[c]['found']]
                print(f"[{count}장] {filename} 저장됨 | 감지 색상: {detected if detected else '없음'}")

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()