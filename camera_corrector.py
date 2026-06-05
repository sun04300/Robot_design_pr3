"""
[파일] camera_corrector.py
[목적] 캘리브레이션 데이터를 이용한 실시간 왜곡 보정 모듈
[수정 이력]
  - alpha=1 → alpha=0 변경: 색상 추적 로봇에 최적화 (유효 픽셀 최대화)
  - 해상도 불일치 시 경고 출력
  - pkl 파일에서 저장된 해상도를 읽어 자동 검증
"""

import cv2
import numpy as np
import pickle
import os


class CameraCorrector:
    def __init__(self, calib_file='camera_calibration.pkl', resolution=(640, 480)):
        """
        카메라 왜곡 보정 모듈 초기화.

        :param calib_file:  calibrate_camera.py로 생성한 pkl 파일 경로
        :param resolution:  실제 주행 시 사용할 카메라 해상도 (width, height)
                            ※ 반드시 캘리브레이션 촬영 해상도와 동일해야 함!
        """
        self.calib_file = calib_file
        self.w, self.h  = resolution
        self.mapx       = None
        self.mapy       = None
        self.roi        = None
        self.is_ready   = False

        self._prepare_maps()

    def _prepare_maps(self):
        """
        왜곡 보정용 매핑 테이블을 1회만 미리 계산.
        실시간 루프에서 매번 계산하면 FPS가 급락하므로 초기화 시 1회만 실행.
        """
        if not os.path.exists(self.calib_file):
            print(f"[경고] 캘리브레이션 파일 '{self.calib_file}'이 없습니다. "
                  "보정 없이 원본 프레임을 반환합니다.")
            return

        try:
            with open(self.calib_file, 'rb') as f:
                data = pickle.load(f)

            mtx  = data['camera_matrix']
            dist = data['dist_coeffs']

            # [수정] 캘리브레이션 해상도와 현재 설정 해상도 불일치 검증
            if 'resolution' in data:
                calib_w, calib_h = data['resolution']
                if (calib_w, calib_h) != (self.w, self.h):
                    print(f"[경고] 해상도 불일치! "
                          f"캘리브레이션: ({calib_w}×{calib_h}), "
                          f"현재 설정: ({self.w}×{self.h})")
                    print("  → 카메라 해상도를 캘리브레이션 때와 동일하게 맞춰주세요.")
                    print("  → cap.set(cv2.CAP_PROP_FRAME_WIDTH, ...) 사용")

            # [수정] alpha=0 사용: 검은 여백 없이 유효 픽셀만 남김
            # alpha=1(원본): 가장자리 검은 영역 최대 → 색상 컨투어 탐지 시 노이즈 유발
            # alpha=0(수정): 유효 영역만 크롭 → 색상 인식 ROI로 바로 사용 가능
            newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
                mtx, dist, (self.w, self.h), 0, (self.w, self.h)
            )

            # 보정 매핑 테이블 1회 계산 (이후 remap만 호출 → 고속 처리)
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                mtx, dist, None, newcameramtx, (self.w, self.h), cv2.CV_32FC1
            )

            self.is_ready = True
            rms_info = f" (캘리브레이션 RMS: {data.get('rms_error', '?'):.4f} px)" \
                       if 'rms_error' in data else ""
            print(f"[CameraCorrector] 초기화 완료{rms_info}")

        except Exception as e:
            print(f"[에러] 캘리브레이션 파일 로드 실패: {e}")

    def update_frame(self, frame, crop_roi=True, debug=False):
        """
        원본 프레임을 왜곡 보정된 프레임으로 변환하여 반환.

        :param frame:     cv2.VideoCapture로 읽은 원본 BGR 프레임
        :param crop_roi:  True = 보정 후 검은 여백 자동 크롭 (색상 탐지에 권장)
        :param debug:     True = 원본/보정본 비교 화면 표시 (디버깅 완료 후 False로 변경)
        :return:          왜곡 보정된 프레임 (캘리브레이션 파일 없으면 원본 반환)
        """
        if not self.is_ready:
            return frame

        # 고속 왜곡 보정 (미리 계산된 맵 사용 → INTER_LINEAR가 품질/속도 균형 최적)
        dst = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)

        # 유효 영역 크롭 (alpha=0이면 roi가 전체 프레임과 거의 동일)
        if crop_roi and self.roi is not None:
            x, y, w, h = self.roi
            if all(v > 0 for v in [x, y, w, h]):
                dst = dst[y:y + h, x:x + w]

        # [디버깅용] 원본 | 보정본 비교 화면 (확인 완료 후 debug=False 또는 블록 삭제)
        if debug:
            orig_vis = cv2.resize(frame, (480, 360))
            corr_vis = cv2.resize(dst,   (480, 360))
            combined = np.hstack((orig_vis, corr_vis))
            cv2.putText(combined, "Original",  (10,  30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(combined, "Corrected", (490, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow('DEBUG: Original | Corrected', combined)
            cv2.waitKey(1)

        return dst
