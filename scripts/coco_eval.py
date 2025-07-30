import io, json, argparse, yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann_json",  required=True)
    p.add_argument("--pred_json", required=True)
    p.add_argument("--data_yaml", default="data.yaml")
    p.add_argument("--iou_type",  choices=["bbox","segm"], default="bbox")
    args = p.parse_args()

    # 1) GT load + 필드 보강
    with io.open(args.ann_json, 'r', encoding='utf-8', errors='ignore') as f:
        ann = json.load(f)
    ann.setdefault("info", {})
    ann.setdefault("licenses", [])
    if "categories" not in ann:
        names = yaml.safe_load(open(args.data_yaml))["names"]
        ann["categories"] = [
            {"id": int(k), "name": names[k], "supercategory": ""}
            for k in names
        ]

    # 2) COCO 인덱스 생성
    coco_gt = COCO()
    coco_gt.dataset = ann
    coco_gt.createIndex()

    # 3) 예측 load
    with io.open(args.pred_json, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    coco_dt = coco_gt.loadRes(preds)

    # 4) eval
    evaler = COCOeval(coco_gt, coco_dt, iouType=args.iou_type)
    evaler.params.imgIds = sorted(coco_gt.getImgIds())
    evaler.evaluate()
    evaler.accumulate()
    evaler.summarize()

if __name__=="__main__":
    main()