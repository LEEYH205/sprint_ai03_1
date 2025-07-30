import argparse
import json
import random
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
import cv2
import pandas as pd
from models.model import get_yolov8_model
from ultralytics import YOLO  # if 필요할 경우

def save_model_record(
    db_path: str,
    name: str,
    weights_path: str,
    epochs: int,
    batch: int,
    imgsz: int,
    lr0: float,
    mAP50: float,
    mAP50_95: float,
    box_loss: float,
    val_split: float = 0.8,
    notes: str = ""
):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT    NOT NULL,
            weights_path  TEXT    NOT NULL,
            trained_at    TEXT    NOT NULL,
            epochs        INTEGER,
            batch_size    INTEGER,
            imgsz         INTEGER,
            lr0           REAL,
            mAP50         REAL,
            mAP50_95      REAL,
            box_loss      REAL,
            val_split     REAL,
            notes         TEXT
        )
    """)
    c.execute("""
        INSERT INTO models (
            name, weights_path, trained_at,
            epochs, batch_size, imgsz, lr0,
            mAP50, mAP50_95, box_loss, val_split, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        name, weights_path, datetime.now().isoformat(),
        epochs, batch, imgsz, lr0,
        mAP50, mAP50_95, box_loss, val_split, notes
    ))
    conn.commit()
    conn.close()


def split_and_convert(raw_img_dir, raw_ann_dir, out_root, split_ratio=0.8, seed=42):
    cat_ids = {}
    for jp in Path(raw_ann_dir).rglob("*.json"):
        part = json.load(open(jp, 'r', encoding='utf-8'))
        for c in part.get('categories', []):
            cat_ids[c['id']] = c

    category_ids = sorted(cat_ids.keys())
    id_to_index   = {cid: idx for idx, cid in enumerate(category_ids)}
    coco = {
            'images': [],
            'annotations': [],
            'categories': list(cat_ids.values())
        }
    
    img_ids, ann_ids = set(), set()
    for jp in Path(raw_ann_dir).rglob("*.json"):
        part = json.load(open(jp, encoding='utf-8'))
        for c in part.get('categories', []):
            if c['id'] not in cat_ids:
                cat_ids[c['id']] = c
        for img in part.get('images', []):
            if img['id'] not in img_ids:
                coco['images'].append(img); img_ids.add(img['id'])
        for ann in part.get('annotations', []):
            if ann['id'] not in ann_ids:
                coco['annotations'].append(ann); ann_ids.add(ann['id'])
    coco['categories'] = list(cat_ids.values())

    imgs = coco['images']
    random.seed(seed)
    random.shuffle(imgs)
    n = int(len(imgs) * split_ratio)
    splits = {'train': imgs[:n], 'val': imgs[n:]}

    for subset, img_list in splits.items():
        img_out = Path(out_root) / subset / 'images'
        lbl_out = Path(out_root) / subset / 'labels'
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img in img_list:
            fn = img['file_name']
            shutil.copy(Path(raw_img_dir) / fn, img_out / fn)
            rel = [a for a in coco['annotations'] if a['image_id'] == img['id']]
            lines = []
            for a in rel:
                x, y, w, h = a['bbox']
                xc = (x + w/2) / img['width']
                yc = (y + h/2) / img['height']
                nw, nh = w / img['width'], h / img['height']
                cls = id_to_index[a['category_id']]
                lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
            (lbl_out / f"{Path(fn).stem}.txt").write_text(''.join(lines))


def find_latest_experiment(runs_dir: Path) -> Path:
    exps = sorted(
        [p for p in runs_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    return exps[0] if exps else None


def main():
    parser = argparse.ArgumentParser("YOLOv8 Inference (with NMS/TTA/Rescale)")
    parser.add_argument('--raw_img_dir',   default='data/raw_data/train_images')
    parser.add_argument('--raw_ann_dir',   default='data/raw_data/train_annotations')
    parser.add_argument('--processed_root',default='data/processed')
    parser.add_argument('--epochs',    type=int,   default=50)
    parser.add_argument('--batch',     type=int,   default=8)
    parser.add_argument('--imgsz',     type=int,   default=640)
    parser.add_argument('--lr0',       type=float, default=0.001)
    parser.add_argument('--exist_ok',  action='store_true')
    parser.add_argument('--name',  choices=['new','resume'], default='new')
    parser.add_argument('--conf_thresh', type=float, default=0.25)
    args = parser.parse_args()

    # 1) exp_name & resume 결정
    if args.name == 'new':
        exp_name = datetime.now().strftime('pill_exp_%Y%m%d_%H%M%S')
        resume = False
    else:
        latest = find_latest_experiment(Path('runs/train'))
        if not latest:
            print("기존 실험이 없습니다. 먼저 --name new 로 새 학습을 시작하세요.")
            return
        exp_name   = latest.name
        last_pt    = latest / 'weights' / 'last.pt'
        results_csv= latest / 'results.csv'
        if not last_pt.exists() or not results_csv.exists():
            print(f"{exp_name}에서 이어갈 체크포인트 또는 결과 CSV를 찾을 수 없습니다.")
            return
        df = pd.read_csv(results_csv)
        if df['epoch'].max() >= args.epochs:
            print(f"이전 실험({exp_name})이 이미 {args.epochs} 에폭 이상 완료되었습니다.")
            return
        resume = True
    # 영은님 데이터 증강 파일을 받아와 실행하기 위해 블락 처리
    # # 2) 데이터 준비
    # split_and_convert(
    #     raw_img_dir=args.raw_img_dir,
    #     raw_ann_dir=args.raw_ann_dir,
    #     out_root=args.processed_root,
    #     split_ratio=0.8,
    #     seed=42
    # )
    # print("train/val split & YOLO txt 생성 완료")

    # 3) data.yaml 생성 (멀티클래스)
    cat_ids = {}
    for jp in Path(args.raw_ann_dir).rglob("*.json"):
        jd = json.load(open(jp, 'r', encoding='utf-8'))
        for c in jd.get('categories', []):
            cid = str(c['id'])  # 실제 알약 코드
            cname = c.get('name', cid)
            cat_ids[cid] = cname

    sorted_ids = sorted(cat_ids.items(), key=lambda x: int(x[0]))  # [(23, '약이름'), ...]
    data_yaml = Path(__file__).parent.parent / 'data.yaml'

    with open(data_yaml, 'w', encoding='utf-8') as f:
        # f.write(f"train: {args.processed_root}/train/images\n")
        # f.write(f"val:   {args.processed_root}/val/images\n")

        f.write(f"train: data/images/train_images\n")
        f.write(f"val:   data/images/val_images\n")
        f.write(f"nc: {len(sorted_ids)}\n")
        f.write("names:\n")
        for _, name in sorted_ids:
            f.write(f"  - '{name}'\n")

    print(f"data.yaml 생성 (multi-class): {data_yaml}")

    # 4) 학습 호출
    pretrained_path = (
        Path('runs/train')/exp_name/'weights'/'last.pt'
    ) if resume else 'yolov8m.pt'
    train_fn, _ = get_yolov8_model(
        pretrained=pretrained_path,
        data_yaml=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        project='runs/train',
        name=exp_name,
        exist_ok=args.exist_ok,
        patience=3
    )
    train_fn(resume=resume)
    print(f"학습 완료 ({exp_name})")

    # 5) DB 저장
    latest_run = Path('runs/train')/exp_name
    df_res     = pd.read_csv(latest_run/'results.csv')
    last       = df_res.iloc[-1]
    save_model_record(
        db_path="models.db",
        name=exp_name,
        weights_path=str(latest_run/'weights'/'best.pt'),
        epochs=int(last['epoch']),
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        mAP50=float(last['metrics/mAP50(B)']),
        mAP50_95=float(last['metrics/mAP50-95(B)']),
        box_loss=float(last['val/box_loss']),
        val_split=0.8,
        notes="auto-saved after training"
    )
    print("모델 기록이 models.db에 자동 저장되었습니다.")

    # 6) CSV 덤프
    conn   = sqlite3.connect("models.db")
    df_all = pd.read_sql("SELECT * FROM models", conn)
    conn.close()
    save_dir = Path("save"); save_dir.mkdir(exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = save_dir/f"{ts}_models.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"모델 기록이 {csv_path}에 저장되었습니다.")


if __name__ == '__main__':
    main()