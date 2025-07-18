from ultralytics import YOLO
import random
import matplotlib.pyplot as plt
from PIL import Image

# 1) 학습된 모델 로드
model = YOLO('runs/train/pill_exp/weights/best.pt')

# 2) 테스트 이미지 목록
import glob
test_imgs = glob.glob('data/raw_data/test_images/*.*')
img_file = random.choice(test_imgs)

# 3) 예측
results = model.predict(source=img_file, conf=0.25, imgsz=640, save=False)

# 4) 시각화
boxes = results[0].boxes.xyxy.cpu().numpy()
scores = results[0].boxes.conf.cpu().numpy()
labels = results[0].boxes.cls.cpu().numpy()

img = Image.open(img_file)
plt.figure(figsize=(8,8))
plt.imshow(img)
ax = plt.gca()
for (x1,y1,x2,y2), s in zip(boxes, scores):
    ax.add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, edgecolor='r', lw=2))
    ax.text(x1, y1-5, f"pill {s:.2f}", color='yellow', fontsize=12, weight='bold')
plt.axis('off')
plt.show()