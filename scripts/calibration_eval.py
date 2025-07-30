import io, json, argparse
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def compute_ece(scores, correctness, bins=10):
    """
    scores: 예측 confidence 배열 (np.ndarray)
    correctness: TP(1)/FP(0) 배열 (np.ndarray)
    bins: bin 개수
    """
    bin_edges = np.linspace(0,1,bins+1)
    ece = 0.0
    for i in range(bins):
        mask = (scores >= bin_edges[i]) & (scores < bin_edges[i+1])
        if mask.sum() == 0:
            continue
        acc  = correctness[mask].mean()
        conf = scores[mask].mean()
        ece += abs(acc - conf) * mask.sum() / len(scores)
    return ece

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ann_json',  required=True, help='GT COCO annotations.json')
    p.add_argument('--pred_json', required=True, help='your predictions in COCO JSON')
    p.add_argument('--iou_thr',   type=float, default=0.5)
    p.add_argument('--bins',      type=int,   default=10)
    args = p.parse_args()

    # 1) GT 어노테이션을 UTF-8로 열어서 coco.dataset에 직접 주입
    with io.open(args.ann_json, 'r', encoding='utf-8', errors='ignore') as f:
        ann = json.load(f)
    ann.setdefault('info', {})    # info 필드 없으면 빈 dict 채우기
    coco_gt = COCO()
    coco_gt.dataset = ann
    coco_gt.createIndex()

    # 2) prediction JSON load
    with io.open(args.pred_json, 'r', encoding='utf-8', errors='ignore') as f:
        preds = json.load(f)
    coco_dt = coco_gt.loadRes(preds)

    # 3) COCOeval 실행
    evaluator = COCOeval(coco_gt, coco_dt, iouType='bbox')
    evaluator.params.imgIds = sorted(coco_gt.getImgIds())
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()

    # 3-1) ECE 계산을 위한 scores / correct 수집
    scores = []
    correct = []
    for pred in preds:
        img_id = pred['image_id']
        cls    = pred['category_id']
        bbox   = pred['bbox']   # [x,y,w,h]
        score  = pred['score']

        # 같은 이미지·클래스의 GT 박스들
        gt_boxes = [
            g['bbox'] for g in ann['annotations']
            if g['image_id']==img_id and g['category_id']==cls
        ]
        # best IoU 계산
        best_iou = 0
        for gb in gt_boxes:
            xa = max(bbox[0], gb[0]); ya = max(bbox[1], gb[1])
            xb = min(bbox[0]+bbox[2], gb[0]+gb[2])
            yb = min(bbox[1]+bbox[3], gb[1]+gb[3])
            inter = max(0, xb-xa) * max(0, yb-ya)
            union = bbox[2]*bbox[3] + gb[2]*gb[3] - inter
            best_iou = max(best_iou, inter/union if union>0 else 0)
        is_tp = best_iou >= args.iou_thr
        scores.append(score)
        correct.append(1 if is_tp else 0)

    scores  = np.array(scores)
    correct = np.array(correct)

    # 4) ECE 계산 및 출력
    ece = compute_ece(scores, correct, bins=args.bins)
    print(f"\nECE ({args.bins} bins): {ece:.4f}")

    # 5) Reliability Diagram 그리기
    edges   = np.linspace(0,1,args.bins+1)
    centers = (edges[:-1] + edges[1:]) / 2
    avg_conf, avg_acc = [], []
    for i in range(args.bins):
        m = (scores >= edges[i]) & (scores < edges[i+1])
        if m.sum() > 0:
            avg_conf.append(scores[m].mean())
            avg_acc.append(correct[m].mean())
        else:
            avg_conf.append(np.nan)
            avg_acc.append(np.nan)

    plt.figure()
    plt.plot(centers, avg_conf, label='avg_conf', marker='o')
    plt.plot(centers, avg_acc,  label='avg_acc',  marker='x')
    plt.plot([0,1],[0,1],'--',    label='ideal')
    plt.xlabel('Confidence bin')
    plt.ylabel('Value')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    main()