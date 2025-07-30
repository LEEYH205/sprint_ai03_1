import argparse
import cv2
import pandas as pd
import os

def draw_boxes(image_path, csv_path, conf_thresh=0.25):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    df = pd.read_csv(csv_path)
    image_id = os.path.basename(image_path)

    df = df[df['image_id'] == image_id]
    df = df[df['score'] >= conf_thresh]

    for _, row in df.iterrows():
        x, y, bw, bh = int(row['bbox_x']), int(row['bbox_y']), int(row['bbox_w']), int(row['bbox_h'])
        category_id = int(row['category_id'])
        score = row['score']

        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(img, f"{category_id} ({score:.2f})", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    out_path = "vis_" + os.path.basename(image_path)
    cv2.imwrite(out_path, img)
    print(f"✅ 시각화 결과 저장 완료: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Single image visualization")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    draw_boxes(args.image, args.csv, args.conf)