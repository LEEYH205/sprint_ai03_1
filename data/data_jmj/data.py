import os, json, shutil
from pathlib import Path

def coco_to_yolo(raw_img_dir, raw_ann_dir, out_img_dir, out_lbl_dir):
    """
    COCO-format JSON을 읽어서
    - out_img_dir: 이미지 복사
    - out_lbl_dir: YOLO txt 라벨 생성
    """
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # ─── raw_ann_dir 안의 모든 JSON을 하나로 합치기 ───
    coco = {'images': [], 'annotations': [], 'categories': []}
    image_ids   = set()
    ann_ids     = set()
    cat_ids     = {}
    # 순회하면서
    for jp in Path(raw_ann_dir).rglob("*.json"):
        with open(jp, 'r', encoding='utf-8') as f:
            part = json.load(f)
        # categories (중복 방지)
        for c in part.get('categories', []):
            if c['id'] not in cat_ids:
                cat_ids[c['id']] = c
        # images
        for img in part.get('images', []):
            if img['id'] not in image_ids:
                coco['images'].append(img)
                image_ids.add(img['id'])
        # annotations
        for ann in part.get('annotations', []):
            if ann['id'] not in ann_ids:
                coco['annotations'].append(ann)
                ann_ids.add(ann['id'])
    coco['categories'] = list(cat_ids.values())

    images = {img['id']: img for img in coco['images']}
    for img in coco['images']:
        src = Path(raw_img_dir) / img['file_name']
        dst = Path(out_img_dir) / img['file_name']
        shutil.copy(src, dst)

    ann_dict = {}
    for ann in coco['annotations']:
        img = images[ann['image_id']]
        w, h = img['width'], img['height']
        x, y, bw, bh = ann['bbox']
        xc, yc = (x + bw/2)/w, (y + bh/2)/h
        nw, nh = bw/w, bh/h
        cls = ann['category_id'] - 1  # zero-based

        fn = Path(out_lbl_dir) / f"{Path(img['file_name']).stem}.txt"
        ann_dict.setdefault(fn, []).append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

    for fn, lines in ann_dict.items():
        fn.write_text(''.join(lines))