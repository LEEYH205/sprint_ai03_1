import json
import os
from pathlib import Path

# 1) 설정
COCO_JSON     = "data/processed/train/annotations.json"
IMG_DIR       = Path("data/working_subset/images")
OUT_LABEL_DIR = Path("data/working_subset/labels")
OUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

# 2) COCO 불러오기
with open(COCO_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)
# image_id → 이미지 메타맵
img_map = {img["id"]: img for img in coco["images"]}
# filename → image_id 맵
fn2id = {img["file_name"]: img["id"] for img in coco["images"]}

# 3) subset 폴더의 이미지만 처리
for img_path in IMG_DIR.iterdir():
    if img_path.suffix.lower() not in (".jpg", ".png", ".jpeg"):
        continue
    fname = img_path.name
    if fname not in fn2id:
        print(f"⚠️ {fname} 가 COCO JSON에 없습니다.")
        continue

    img_info = img_map[fn2id[fname]]
    w, h     = img_info["width"], img_info["height"]

    # 4) 이미지 ID에 해당하는 어노테이션만 필터
    anns = [a for a in coco["annotations"] if a["image_id"] == img_info["id"]]
    lines = []
    for ann in anns:
        cls = ann["category_id"]
        x, y, bw, bh = ann["bbox"]
        cx = (x + bw/2) / w
        cy = (y + bh/2) / h
        nw = bw / w
        nh = bh / h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    # 5) YOLO TXT로 저장
    out_path = OUT_LABEL_DIR / f"{img_path.stem}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print(f"✅ {out_path} 생성 ({len(lines)} boxes)")

print("🎉 YOLO TXT 변환 완료!")