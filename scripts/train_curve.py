import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def plot_results(exp_dir: Path):
    # results.csv 로드
    df = pd.read_csv(exp_dir / 'results.csv')
    epochs = df['epoch']

    # 그릴 항목과 서브플롯 위치 지정
    metrics = [
        ('train/box_loss',    'train/box_loss'),
        ('train/cls_loss',    'train/cls_loss'),
        ('train/dfl_loss',    'train/dfl_loss'),
        ('metrics/precision(B)', 'metrics/precision(B)'),
        ('metrics/recall(B)',    'metrics/recall(B)'),
        ('val/box_loss',      'val/box_loss'),
        ('val/cls_loss',      'val/cls_loss'),
        ('val/dfl_loss',      'val/dfl_loss'),
        ('metrics/mAP50(B)',    'metrics/mAP50(B)'),
        ('metrics/mAP50-95(B)', 'metrics/mAP50-95(B)')
    ]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, metrics):
        if col not in df.columns:
            continue
        ax.plot(epochs, df[col], marker='o', linewidth=1, label='results')
        # smoothing (간단히 이동평균)
        df[f'{col}_smooth'] = df[col].rolling(5, min_periods=1).mean()
        ax.plot(epochs, df[f'{col}_smooth'], linestyle='--', label='smooth')
        ax.set_title(title)
        ax.grid(True)
        ax.legend(fontsize='small')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 첫 번째 인자: runs/train/<exp_name> 폴더 경로
    if len(sys.argv) != 2:
        print("Usage: python plot_training_curves.py runs/train/<exp_name>")
        sys.exit(1)

    exp_path = Path(sys.argv[1])
    if not exp_path.exists():
        print(f"경로를 찾을 수 없습니다: {exp_path}")
        sys.exit(1)

    plot_results(exp_path)