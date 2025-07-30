import os
import argparse
import csv
import cv2
import json
import yaml
from ensemble_boxes import weighted_boxes_fusion
from ultralytics import YOLO

def load_cat_id_map(data_yaml_path):
    with open(data_yaml_path, 'r', encoding='utf-8') as f:  # ← 여기 중요
        cfg = yaml.safe_load(f)

    if isinstance(cfg['names'], dict):
        return {int(k): int(v) for k, v in cfg['names'].items()}
    elif isinstance(cfg['names'], list):
        return {i: name for i, name in enumerate(cfg['names'])}
    else:
        raise TypeError("data.yaml의 names 필드 형식이 잘못되었습니다.")

def main():
    parser = argparse.ArgumentParser("YOLOv8 inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--img_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default="output")
    parser.add_argument("--csv_file", type=str, default="predictions.csv")
    parser.add_argument("--data_yaml", type=str, default="data.yaml")
    parser.add_argument("--conf_thresh", type=float, default=0.25)
    parser.add_argument("--iou_thresh", type=float, default=0.45)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--ensemble_ckpts", nargs='+', default=[])
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    cat_id_map = load_cat_id_map(args.data_yaml)

    ckpt_paths = [args.checkpoint] + args.ensemble_ckpts
    models = [YOLO(ckpt) for ckpt in ckpt_paths]
    print(f"[INFO] {len(models)} model(s) loaded.")

    img_files = sorted([
        fn for fn in os.listdir(args.img_folder)
        if fn.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    with open(args.csv_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"])
        ann_id = 1

        for img_name in img_files:
            base = os.path.splitext(img_name)[0]
            image_id = int(base) if base.isdigit() else base
            img_path = os.path.join(args.img_folder, img_name)
            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            all_boxes, all_scores, all_labels = [], [], []
            for model in models:
                pred = model.predict(
                    source=img_path,
                    conf=args.conf_thresh,
                    iou=args.iou_thresh,
                    augment=args.tta,
                    save=False
                )[0]
                data = pred.boxes.data.cpu().numpy()
                if data.shape[0] == 0:
                    continue
                boxes = data[:, :4]
                scores = data[:, 4]
                labels = data[:, 5].astype(int)

                norm_boxes = [[x1 / w, y1 / h, x2 / w, y2 / h] for x1, y1, x2, y2 in boxes]
                all_boxes.append(norm_boxes)
                all_scores.append(scores.tolist())
                all_labels.append(labels.tolist())

            if not all_boxes:
                continue

            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                all_boxes, all_scores, all_labels,
                iou_thr=args.iou_thresh,
                skip_box_thr=args.conf_thresh
            )

            annotated = img.copy()
            for (x1n, y1n, x2n, y2n), score, cls_idx in zip(fused_boxes, fused_scores, fused_labels):
                x1, y1 = int(x1n * w), int(y1n * h)
                x2, y2 = int(x2n * w), int(y2n * h)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(annotated, f"{cls_idx}:{score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                category_id = cat_id_map.get(cls_idx, cls_idx)
                writer.writerow([ann_id, image_id, category_id, x1, y1, x2 - x1, y2 - y1, float(score)])
                ann_id += 1

            cv2.imwrite(os.path.join(args.output_folder, img_name), annotated)

    print("\nInference 완료")
    print(f"Annotated images → {args.output_folder}/")
    print(f"Predictions CSV  → {args.csv_file}")

if __name__ == "__main__":
    main()
