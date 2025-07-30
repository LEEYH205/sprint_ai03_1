import csv
import json

def csv_to_coco_res(csv_path, json_path):
    results = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # COCO result 포맷:
            # { "image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float }
            x = float(row['bbox_x'])
            y = float(row['bbox_y'])
            w = float(row['bbox_w'])
            h = float(row['bbox_h'])
            results.append({
                "image_id":    int(row['image_id']),
                "category_id": int(row['category_id']),
                "bbox":        [x, y, w, h],
                "score":       float(row['score'])
            })
    # JSON으로 저장
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"{json_path} 생성 완료 ({len(results)} items)")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--csv',    default='holdout_preds.csv', help='입력 CSV 파일')
    p.add_argument('--output', default='holdout_preds.json', help='생성할 JSON 파일')
    args = p.parse_args()
    csv_to_coco_res(args.csv, args.output)