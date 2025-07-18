import argparse
from models.model import get_yolov8_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--source',  type=str, required=True)
    parser.add_argument('--imgsz',    type=int, default=512)
    parser.add_argument('--conf',     type=float, default=0.25)
    parser.add_argument('--iou',      type=float, default=0.45, help='NMS IoU threshold')
    args = parser.parse_args()

    _, predict = get_yolov8_model(
        pretrained=args.weights,
        imgsz=args.imgsz,
        name='pill_eval'
    )
    preds = predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save_dir='runs/evaluate'
    )
    print("▶ evaluation done, outputs in runs/evaluate/pill_eval/")

if __name__ == '__main__':
    main()