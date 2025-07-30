import argparse
from models.model import get_yolov8_model

def main():
    parser = argparse.ArgumentParser("Evaluate images with trained model")
    parser.add_argument('--weights', type=str, required=True, help="Path to .pt weights")
    parser.add_argument('--source',  type=str, required=True, help="Folder with test images")
    parser.add_argument('--imgsz',   type=int, default=512)
    parser.add_argument('--conf',    type=float, default=0.25)
    parser.add_argument('--iou',     type=float, default=0.45)
    parser.add_argument('--save_dir', type=str, default='runs/evaluate')
    args = parser.parse_args()

    _, predict_fn = get_yolov8_model(
        pretrained=args.weights,
        imgsz=args.imgsz,
        name='pill_eval'
    )

    predict_fn(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save_dir=args.save_dir
    )

    print(f"Evaluation 완료 → 결과는 {args.save_dir}/pill_eval/")

if __name__ == '__main__':
    main()
