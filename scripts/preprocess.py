import argparse
import random
import shutil
from pathlib import Path

def split_dataset(img_src, lbl_src, out_dir, ratio=0.8, seed=42):
    img_src = Path(img_src)
    lbl_src = Path(lbl_src)
    out_dir = Path(out_dir)

    # 폴더 생성
    for split in ("train", "val"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # 이미지 목록 수집
    imgs = [p.name for p in img_src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    random.seed(seed)
    random.shuffle(imgs)

    # split
    n_train = int(len(imgs) * ratio)
    train_imgs = imgs[:n_train]
    val_imgs   = imgs[n_train:]

    for split, subset in (("train", train_imgs), ("val", val_imgs)):
        for img_name in subset:
            # 이미지 복사
            src_img = img_src / img_name
            dst_img = out_dir / split / "images" / img_name
            shutil.copy(src_img, dst_img)

            # 해당 이미지와 매칭되는 JSON 어노테이션 전부 복사
            base = Path(img_name).stem
            for json_path in lbl_src.rglob(f"{base}*.json"):
                rel_dir = json_path.relative_to(lbl_src).parent
                dst_lbl_dir = out_dir / split / "labels" / rel_dir
                dst_lbl_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(json_path, dst_lbl_dir / json_path.name)

    print(f"✔ Split complete: {len(train_imgs)} train / {len(val_imgs)} val images")

def main():
    parser = argparse.ArgumentParser(
        description="Split images + JSON annotations into train/val folders"
    )
    parser.add_argument("--img-dir",   required=True, help="원본 이미지 폴더 경로")
    parser.add_argument("--label-dir", required=True, help="원본 JSON 어노테이션 폴더 경로")
    parser.add_argument("--out-dir",   required=True, help="결과를 저장할 베이스 폴더")
    parser.add_argument("--ratio",     type=float, default=0.8, help="train 비율 (기본: 0.8)")
    parser.add_argument("--seed",      type=int,   default=42,  help="랜덤 시드")
    args = parser.parse_args()

    split_dataset(args.img_dir, args.label_dir, args.out_dir, args.ratio, args.seed)

if __name__ == "__main__":
    main()