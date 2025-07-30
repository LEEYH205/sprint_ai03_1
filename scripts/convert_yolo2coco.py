import os
import json
import argparse
from tqdm import tqdm
import yaml

def yolo2coco(img_dir, label_dir, data_yaml, out_json):
    # --- 0) 빈 리스트 미리 선언 ---
    images = []
    annotations = []
    ann_id = 1

    # 1) data.yaml 에서 카테고리 뽑기
    cfg = yaml.safe_load(open(data_yaml, 'r'))
    # YAML loader가 숫자 키를 int로 바꿀 수도 있으니, 양쪽 다 지원
    names = cfg['names']
    def name2cat(idx):
        # idx는 int cls
        # names의 키가 int면 그대로, str면 str(idx)로 꺼내기
        return names.get(idx, names.get(str(idx)))

    categories = [
      {"id": int(cid), "name": names[cid], "supercategory": ""}
      for cid in names
    ]

    # 2) images 리스트 채우기
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_id = os.path.splitext(fname)[0]
        images.append({
          "id": img_id,
          "file_name": fname,
          "width": 640,   # 실제라면 cv2.imread한 후 shape으로 대체
          "height": 512
        })

    # 3) annotations 리스트 채우기
    for fname in tqdm(sorted(os.listdir(label_dir))):
        if not fname.lower().endswith('.txt'):
            continue
        img_id = os.path.splitext(fname)[0]
        with open(os.path.join(label_dir, fname)) as f:
            for line in f:
                cls, xc, yc, w, h = map(float, line.split())
                cls = int(cls)
                # YOLO normalized -> COCO absolute
                iw, ih = 640, 512
                bx = (xc - w/2) * iw
                by = (yc - h/2) * ih
                bw = w * iw
                bh = h * ih

                annotations.append({
                  "id": ann_id,
                  "image_id": img_id,
                  "category_id": int(name2cat(cls)),
                  "bbox": [bx, by, bw, bh],
                  "area": bw*bh,
                  "iscrowd": 0
                })
                ann_id += 1

    coco = {
      "images": images,
      "annotations": annotations,
      "categories": categories
    }

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    print(f"{out_json} 생성 완료!  images: {len(images)}, anns: {len(annotations)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--img_dir',   required=True)
    p.add_argument('--label_dir', required=True)
    p.add_argument('--data_yaml', required=True)
    p.add_argument('--out_json',  required=True)
    args = p.parse_args()
    yolo2coco(
      args.img_dir,
      args.label_dir,
      args.data_yaml,
      args.out_json
    )