"""
[파일] calibrate_camera.py
[목적] 체커보드 이미지를 이용한 카메라 내부 파라미터 캘리브레이션
[수정 이력]
  - objp에 실제 체커보드 한 칸 크기(mm) 반영 → 거리 추정 정밀도 확보
  - 이미지 로드 실패 예외 처리 추가 → 런타임 크래시 방지
  - 해상도 자동 감지 및 주석 강화
"""

import cv2
import numpy as np
import os
import glob
import pickle


def calibrate_camera():
    # =========================================================
    # [설정값] 실제 사용하는 체커보드에 맞게 반드시 수정할 것!
    # 내부 코너 수 = (가로 칸 수 - 1, 세로 칸 수 - 1)
    # 예: 가로 11칸 × 세로 8칸 체커보드 → (10, 7)
    CHECKERBOARD   = (7, 10)   # ← 캘리브레이션 전날 체커보드 세어서 확인!
    SQUARE_SIZE_MM = 25.0      # ← 인쇄된 체커보드 한 칸의 실제 크기 (mm)
    MIN_VALID      = 15        # 최소 유효 이미지 수 (이 미만이면 재촬영 권고)
    # =========================================================

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints = []   # 3D 세계 좌표 (체커보드 평면)
    imgpoints = []   # 2D 이미지 좌표 (검출된 코너)

    # 3D 기준점 생성: (0,0,0), (25,0,0), (50,0,0), ... mm 단위
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = (
        np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        * SQUARE_SIZE_MM   # ← 실제 mm 단위 반영 (수정된 부분)
    )

    image_dir = './checkerboards'
    images = glob.glob(os.path.join(image_dir, '*.png'))

    if not images:
        print(f"[에러] '{image_dir}' 폴더에 .png 이미지가 없습니다.")
        return None

    print(f"총 {len(images)}개 이미지 분석 시작...")
    valid_count = 0
    gray_shape  = None

    for fname in images:
        img = cv2.imread(fname)

        # [수정] 이미지 로드 실패 시 크래시 방지 (원본 코드에 없던 예외 처리)
        if img is None:
            print(f"  [경고] 로드 실패, 건너뜀: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]  # (width, height) 형태로 저장

        # 체커보드 코너 검출
        # ADAPTIVE_THRESH: 조명 불균일 대응
        # FAST_CHECK:      코너 없는 이미지 빠르게 스킵
        # NORMALIZE_IMAGE: 전체 밝기 정규화
        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            objpoints.append(objp)
            # 서브픽셀 정밀도로 코너 위치 정제
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            valid_count += 1

            # [디버깅용] 검출된 코너 시각화 (확인 후 waitKey(0)으로 바꾸면 한 장씩 확인 가능)
            img_vis = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imshow(f'Corner Detection ({valid_count})', img_vis)
            cv2.waitKey(200)
        else:
            print(f"  [미검출] 코너를 찾지 못했습니다: {os.path.basename(fname)}")

    cv2.destroyAllWindows()
    print(f"\n분석 완료: {len(images)}장 중 {valid_count}장 코너 검출 성공")

    # 유효 이미지 수 경고
    if valid_count < MIN_VALID:
        print(f"[경고] 유효 이미지가 {valid_count}장뿐입니다. "
              f"최소 {MIN_VALID}장 이상 확보해야 정밀한 캘리브레이션이 가능합니다.")
        print("  → 다양한 각도(±45°), 거리, 위치에서 추가 촬영을 권장합니다.")
        if valid_count < 5:
            print("[에러] 이미지가 너무 적어 캘리브레이션을 중단합니다.")
            return None

    # 카메라 행렬(mtx)과 왜곡 계수(dist) 계산
    print("\n카메라 행렬 및 왜곡 계수 계산 중...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray_shape, None, None
    )

    # 재투영 오차(RMS) 계산 → 1.0 픽셀 이하면 양호, 0.5 이하면 우수
    mean_error = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        mean_error += cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    rms = mean_error / len(objpoints)
    print(f"  재투영 오차(RMS): {rms:.4f} px  "
          f"{'✓ 우수' if rms < 0.5 else '✓ 양호' if rms < 1.0 else '△ 재촬영 권장'}")

    # 결과 저장
    calibration_data = {
        'camera_matrix' : mtx,
        'dist_coeffs'   : dist,
        'rms_error'     : rms,
        'square_size_mm': SQUARE_SIZE_MM,
        'checkerboard'  : CHECKERBOARD,
        'resolution'    : gray_shape,   # 캘리브레이션 시 사용한 해상도 저장
    }

    output_file = 'camera_calibration.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(calibration_data, f)

    print(f"\n[성공] '{output_file}' 저장 완료")
    print(f"  카메라 행렬:\n{mtx}")
    print(f"  왜곡 계수: {dist.ravel()}")
    print(f"\n[주의] 주행 시 캡처 해상도를 반드시 {gray_shape}로 맞춰주세요!")
    return calibration_data


if __name__ == "__main__":
    calibrate_camera()
