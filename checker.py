import cv2
import datetime
import os

os.makedirs("./checkerboards", exist_ok=True)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 프레임을 가져올 수 없습니다.")
        break

    cv2.imshow("Video", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('a') and ret:
        filename = datetime.datetime.now().strftime("./checkerboards/capture_%Y%m%d_%H%M%S.png")
        cv2.imwrite(filename, frame)
        count += 1
        print(f"[{count}장] {filename} 저장됨")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()