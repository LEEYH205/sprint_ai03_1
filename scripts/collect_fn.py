import os
import json
import cv2
import numpy as np
from tqdm import tqdm

def iou(boxA, boxB):
    # box = [x, y, w, h]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    union = boxA[2]*boxA[3] + boxB[2]*boxB[3] - inter
    return inter/union if union>0 else 0

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ann_json', required=True, help='holdout/annotations.json')
    p.add_argument('--pred_json', required=True, help='holdout_preds.json')
    p.add_argument('--img_dir',   required=True, help='data/processed/val/images')
    p.add_argument('--out_dir',   default='fn_examples', help='저장할 폴더')
    p.add_argument('--iou_thr',   type=float, default=0.5, help='IoU FN 판정 임계')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) GT & Pred 로드
    coco = json.load(open(args.ann_json, 'r', encoding='utf-8'))
    preds = json.load(open(args.pred_json, 'r', encoding='utf-8'))

    # 이미지 메타
    id2file = {img['id']: img['file_name'] for img in coco['images']}
    # GT 어노테이션
    gt_by_img = {}
    for ann in coco['annotations']:
        gt_by_img.setdefault(ann['image_id'], []).append(ann)

    # Pred by image
    pred_by_img = {}
    for d in preds:
        pred_by_img.setdefault(d['image_id'], []).append(d)

    fn_count = 0
    for img_id, gt_anns in tqdm(gt_by_img.items()):
        file_name = id2file[img_id]
        img_path = os.path.join(args.img_dir, file_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] 이미지를 못 읽음: {img_path}")
            continue

        preds_this = pred_by_img.get(img_id, [])
        for ann in gt_anns:
            gt_box = ann['bbox']      # [x,y,w,h]
            gt_cat = ann['category_id']

            # 같은 이미지 same class 예측 중 IoU>thr 인 게 있는지 찾기
            hits = [p for p in preds_this if p['category_id']==gt_cat
                                        and iou(gt_box, p['bbox'])>=args.iou_thr]
            if hits:
                continue  # FN 아님

            # 누락된 FN
            fn_count += 1
            # 저장할 폴더: fn_examples/{category_id}/
            dst_dir = os.path.join(args.out_dir, str(gt_cat))
            os.makedirs(dst_dir, exist_ok=True)
            # draw red box
            x,y,w,h = map(int, gt_box)
            crop = img.copy()
            cv2.rectangle(crop, (x,y), (x+w,y+h), (0,0,255), 2)
            cv2.putText(crop, f"FN cat{gt_cat}", (x,y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            out_path = os.path.join(dst_dir, file_name)
            cv2.imwrite(out_path, crop)

    print(f"[DONE] total FN boxes: {fn_count}")

if __name__ == "__main__":
    main()