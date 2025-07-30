import cv2
import numpy as np
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
    
def rotate_image_with_bbox(image_path, angle, bbox_info):
    """이미지를 회전시키고 바운딩 박스도 함께 변환"""
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return None, None
    
    height, width = image.shape[:2]
    
    # 회전 중심점 (이미지 중앙)
    center = (width // 2, height // 2)
    
    # 회전 행렬 생성
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 이미지 회전
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    # 바운딩 박스 회전 변환
    x, y, w, h = bbox_info
    bbox_points = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ], dtype=np.float32)
    
    # 바운딩 박스 점들을 회전
    rotated_bbox_points = cv2.transform(bbox_points.reshape(-1, 1, 2), rotation_matrix)
    
    # 새로운 바운딩 박스 계산
    min_x = max(0, np.min(rotated_bbox_points[:, 0, 0]))
    max_x = min(width, np.max(rotated_bbox_points[:, 0, 0]))
    min_y = max(0, np.min(rotated_bbox_points[:, 0, 1]))
    max_y = min(height, np.max(rotated_bbox_points[:, 0, 1]))
    
    new_bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
    
    return rotated_image, new_bbox

def generate_angle_variations(image_path, bbox_info, output_dir, base_filename):
    """70°, 75°, 90° 각도별 이미지 생성"""
    
    variations = []
    
    # 원본 이미지 (기본 각도)
    original_image = cv2.imread(image_path)
    if original_image is not None:
        variations.append({
            'image': original_image,
            'bbox': bbox_info,
            'angle': 0,
            'filename': f"{base_filename}_0.png"
        })
    
    # 각도별 회전
    angles = [70, 75, 90]
    for angle in angles:
        rotated_image, new_bbox = rotate_image_with_bbox(image_path, angle, bbox_info)
        if rotated_image is not None and new_bbox is not None:
            variations.append({
                'image': rotated_image,
                'bbox': new_bbox,
                'angle': angle,
                'filename': f"{base_filename}_{angle}.png"
            })
    
    # 이미지 저장
    for variation in variations:
        output_path = os.path.join(output_dir, variation['filename'])
        cv2.imwrite(output_path, variation['image'])
        print(f"저장됨: {variation['filename']} (각도: {variation['angle']}°)")
    
    return variations

def visualize_augmentation(image_path, bbox_info):
    """데이터 증강 결과 시각화"""
    
    # 원본 이미지
    original_image = cv2.imread(image_path)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # 회전된 이미지들
    rotated_70, bbox_70 = rotate_image_with_bbox(image_path, 70, bbox_info)
    rotated_75, bbox_75 = rotate_image_with_bbox(image_path, 75, bbox_info)
    rotated_90, bbox_90 = rotate_image_with_bbox(image_path, 90, bbox_info)
    
    if rotated_70 is not None:
        rotated_70_rgb = cv2.cvtColor(rotated_70, cv2.COLOR_BGR2RGB)
    if rotated_75 is not None:
        rotated_75_rgb = cv2.cvtColor(rotated_75, cv2.COLOR_BGR2RGB)
    if rotated_90 is not None:
        rotated_90_rgb = cv2.cvtColor(rotated_90, cv2.COLOR_BGR2RGB)
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 원본
    axes[0, 0].imshow(original_image_rgb)
    x, y, w, h = bbox_info
    rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
    axes[0, 0].add_patch(rect)
    axes[0, 0].set_title('원본 (0°)')
    axes[0, 0].axis('off')
    
    # 70도
    if rotated_70 is not None:
        axes[0, 1].imshow(rotated_70_rgb)
        x, y, w, h = bbox_70
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        axes[0, 1].add_patch(rect)
        axes[0, 1].set_title('70° 회전')
        axes[0, 1].axis('off')
    
    # 75도
    if rotated_75 is not None:
        axes[1, 0].imshow(rotated_75_rgb)
        x, y, w, h = bbox_75
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        axes[1, 0].add_patch(rect)
        axes[1, 0].set_title('75° 회전')
        axes[1, 0].axis('off')
    
    # 90도
    if rotated_90 is not None:
        axes[1, 1].imshow(rotated_90_rgb)
        x, y, w, h = bbox_90
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        axes[1, 1].add_patch(rect)
        axes[1, 1].set_title('90° 회전')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 샘플 이미지와 바운딩 박스
    sample_image_path = "../data/raw_data/train_images/K-001900-010224-016551-031705_0_2_0_2_70_000_200.png"
    sample_bbox = [645, 859, 210, 158]  # 어노테이션에서 가져온 바운딩 박스
    
    # 출력 디렉토리 생성
    output_dir = "data/augmented_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터 증강 실행
    base_filename = "K-001900-010224-016551-031705_augmented"
    variations = generate_angle_variations(
        sample_image_path, 
        sample_bbox, 
        output_dir, 
        base_filename
    )
    
    print(f"총 {len(variations)}개의 변형 이미지 생성됨")
    
    # 시각화
    visualize_augmentation(sample_image_path, sample_bbox) 