import cv2
import os
from ultralytics import YOLO

# 설정
IMG_DIR = "data/processed/val/images"
MODEL_PATH = "runs/train/pill_exp_20250722_154859/weights/best.pt"
image_list = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".png")])
print(f"총 이미지 수: {len(image_list)}")

# 모델 로드
model = YOLO(MODEL_PATH)

# 전역 인덱스
idx = 0

def show_image(index):
    file_name = image_list[index]
    image_path = os.path.join(IMG_DIR, file_name)
    img = cv2.imread(image_path)

    # 예측 수행
    results = model.predict(image_path, conf=0.1, iou=0.6)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{cls_id} ({conf:.2f})"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    print(f"\n {file_name} / 탐지된 알약 수: {len(results.boxes)}")
    resized = cv2.resize(img, (min(img.shape[1], 960), min(img.shape[0], 720)))  # 보기 편하게 리사이즈
    cv2.imshow("Prediction", resized)

# 초기 이미지 출력
show_image(idx)

# 키보드 입력 루프
while True:
    key = cv2.waitKey(0)

    if key == ord('q'):
        print("종료합니다.")
        break
    elif key == 83 or key == ord('d'):  # → 방향키 또는 d
        idx = (idx + 1) % len(image_list)
        show_image(idx)
    elif key == 81 or key == ord('a'):  # ← 방향키 또는 a
        idx = (idx - 1) % len(image_list)
        show_image(idx)

cv2.destroyAllWindows()