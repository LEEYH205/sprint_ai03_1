import argparse
import json
import random
import shutil
from pathlib import Path

from models.model import get_yolov8_model
from torchvision import transforms

def split_and_convert(raw_img_dir, raw_ann_dir, out_root, split_ratio=0.8, seed=42):
    """
    raw_data → data/processed/train+val
    1) 모든 COCO JSON 병합
    2) 80:20 split
    3) train/val 아래 images, labels 폴더로 바로 복사 + YOLO txt 생성
    """
    # 1) JSON 병합
    coco = {'images': [], 'annotations': [], 'categories': []}
    img_ids, ann_ids, cat_ids = set(), set(), {}
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

    # 2) shuffle & split
    imgs = coco['images']
    random.seed(seed); random.shuffle(imgs)
    n = int(len(imgs) * split_ratio)
    splits = {'train': imgs[:n], 'val': imgs[n:]}

    # 3) subset별로 images/labels 생성
    for subset, img_list in splits.items():
        img_out = Path(out_root) / subset / 'images'
        lbl_out = Path(out_root) / subset / 'labels'
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img in img_list:
            fn = img['file_name']
            # 이미지 복사
            shutil.copy(Path(raw_img_dir) / fn, img_out / fn)
            # YOLO txt 생성
            rel = [a for a in coco['annotations'] if a['image_id']==img['id']]
            lines = []
            for a in rel:
                x,y,w,h = a['bbox']
                xc = (x + w/2) / img['width']
                yc = (y + h/2) / img['height']
                nw, nh = w / img['width'], h / img['height']
                cls = 0
                lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
            (lbl_out / f"{Path(fn).stem}.txt").write_text(''.join(lines))

def main():
    p = argparse.ArgumentParser("YOLOv8 알약 탐지")
    p.add_argument('--raw_img_dir',   default='data/raw_data/train_images')
    p.add_argument('--raw_ann_dir',   default='data/raw_data/train_annotations')
    p.add_argument('--processed_root',default='data/processed')
    p.add_argument('--epochs',    type=int,   default=50)
    p.add_argument('--batch',     type=int,   default=16)
    p.add_argument('--imgsz',     type=int,   default=640)
    p.add_argument('--pretrained',type=str,   default='yolov8s.pt')
    p.add_argument('--lr0',       type=float, default=0.001)
    p.add_argument('--exist_ok',  action='store_true')
    args = p.parse_args()

    # 증강 정의
    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop((args.imgsz,args.imgsz),scale=(0.8,1),ratio=(0.9,1.1)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.2),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
    ])

    # 1) split & convert
    split_and_convert(
        raw_img_dir=args.raw_img_dir,
        raw_ann_dir=args.raw_ann_dir,
        out_root=args.processed_root,
        split_ratio=0.8,
        seed=42
    )
    print("train/val split & YOLO txt 생성 완료")

    # 2) data.yaml
    data_yaml = Path(__file__).parent.parent / 'data.yaml'
    data_yaml.write_text(f"""\
train: {args.processed_root}/train/images
val:   {args.processed_root}/val/images
nc: 1
names: ['pill']
""")
    print("data.yaml 생성:", data_yaml)

    # 3) 학습
    train_fn, _ = get_yolov8_model(
        pretrained=args.pretrained,
        data_yaml=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        project='runs/train',
        name='pill_exp',
        exist_ok=args.exist_ok
    )
    train_fn()
    print("학습 완료")

if __name__=='__main__':
    main()