import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import pandas as pd

import matplotlib.font_manager as fm

def korean_font_setting():
    # macOS용 한글 폰트 설정

    # macOS에서 사용 가능한 한글 폰트들
    macos_korean_fonts = [
        '/System/Library/Fonts/AppleGothic.ttf',
        '/System/Library/Fonts/AppleSDGothicNeo.ttc',
        '/Library/Fonts/NanumGothic.ttf',
        '/Library/Fonts/NanumBarunGothic.ttf',
        '/System/Library/Fonts/PingFang.ttc'
    ]

    # 사용 가능한 폰트 찾기
    available_font = None
    for font_path in macos_korean_fonts:
        if os.path.exists(font_path):
            available_font = font_path
            print(f"사용 가능한 폰트 발견: {font_path}")
            break

    if available_font:
        # 폰트 설정
        font_prop = fm.FontProperties(fname=available_font)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        print(f"폰트 설정 완료: {font_prop.get_name()}")
    else:
        # 기본 폰트로 설정
        plt.rcParams['font.family'] = 'AppleGothic'
        plt.rcParams['axes.unicode_minus'] = False
        print("기본 AppleGothic 폰트 사용")
        

def load_all_annotations(annotations_path):
    """모든 어노테이션 파일을 로드하고 약품별로 정리"""
    
    drug_annotations = defaultdict(list)
    image_drug_mapping = defaultdict(list)
    
    # 모든 JSON 파일 찾기
    json_files = []
    for root, dirs, files in os.walk(annotations_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    print(f"총 {len(json_files)}개의 어노테이션 파일 발견")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'images' in data and 'annotations' in data and 'categories' in data:
                image_info = data['images'][0]
                annotation = data['annotations'][0]
                category = data['categories'][0]
                
                # 약품 정보 추출
                drug_code = category['id']
                drug_name = category['name']
                image_name = image_info['file_name']
                bbox = annotation['bbox']
                
                # 약품별 어노테이션 저장
                drug_annotations[drug_code].append({
                    'drug_name': drug_name,
                    'image_name': image_name,
                    'bbox': bbox,
                    'area': annotation['area'],
                    'camera_la': image_info.get('camera_la', 'N/A'),
                    'drug_N': image_info.get('drug_N', 'N/A'),
                    'file_path': json_file
                })
                
                # 이미지별 약품 매핑
                image_drug_mapping[image_name].append({
                    'drug_code': drug_code,
                    'drug_name': drug_name,
                    'bbox': bbox,
                    'area': annotation['area']
                })
                
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return drug_annotations, image_drug_mapping

def analyze_drug_bbox_statistics(drug_annotations):
    """약품별 바운딩 박스 통계 분석"""
    
    stats = []
    
    for drug_code, annotations in drug_annotations.items():
        if not annotations:
            continue
            
        drug_name = annotations[0]['drug_name']
        bbox_areas = [ann['area'] for ann in annotations]
        bbox_widths = [ann['bbox'][2] for ann in annotations]
        bbox_heights = [ann['bbox'][3] for ann in annotations]
        
        stats.append({
            'drug_code': drug_code,
            'drug_name': drug_name,
            'count': len(annotations),
            'avg_area': np.mean(bbox_areas),
            'std_area': np.std(bbox_areas),
            'min_area': np.min(bbox_areas),
            'max_area': np.max(bbox_areas),
            'avg_width': np.mean(bbox_widths),
            'avg_height': np.mean(bbox_heights),
            'avg_aspect_ratio': np.mean(bbox_widths) / np.mean(bbox_heights)
        })
    
    return pd.DataFrame(stats)

def visualize_multi_drug_image(image_path, drug_annotations, save_path=None):
    """다중 약품이 포함된 이미지 시각화"""
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 시각화
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_rgb)
    
    # 색상 팔레트
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan']
    
    # 각 약품별 바운딩 박스 그리기
    for i, drug_info in enumerate(drug_annotations):
        bbox = drug_info['bbox']
        drug_name = drug_info['drug_name']
        color = colors[i % len(colors)]
        
        # 바운딩 박스 그리기
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # 라벨 추가
        ax.text(
            bbox[0], bbox[1] - 10,
            f"{drug_name} (Area: {drug_info['area']})",
            color=color, fontsize=10, weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7)
        )
    
    ax.set_title(f"다중 약품 이미지: {os.path.basename(image_path)}")
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def find_multi_drug_images(image_drug_mapping, min_drugs=2):
    """다중 약품이 포함된 이미지 찾기"""
    
    multi_drug_images = []
    
    for image_name, drugs in image_drug_mapping.items():
        if len(drugs) >= min_drugs:
            multi_drug_images.append({
                'image_name': image_name,
                'drug_count': len(drugs),
                'drugs': drugs
            })
    
    # 약품 수로 정렬
    multi_drug_images.sort(key=lambda x: x['drug_count'], reverse=True)
    
    return multi_drug_images

def analyze_specific_drug(drug_annotations, drug_code):
    """특정 약품의 바운딩 박스 분석"""
    
    if drug_code not in drug_annotations:
        print(f"약품 코드 {drug_code}를 찾을 수 없습니다.")
        return
    
    annotations = drug_annotations[drug_code]
    drug_name = annotations[0]['drug_name']
    
    print(f"\n=== {drug_name} (코드: {drug_code}) 분석 ===")
    print(f"총 어노테이션 수: {len(annotations)}")
    
    # 바운딩 박스 통계
    areas = [ann['area'] for ann in annotations]
    widths = [ann['bbox'][2] for ann in annotations]
    heights = [ann['bbox'][3] for ann in annotations]
    
    print(f"면적 통계:")
    print(f"  평균: {np.mean(areas):.2f}")
    print(f"  표준편차: {np.std(areas):.2f}")
    print(f"  최소: {np.min(areas):.2f}")
    print(f"  최대: {np.max(areas):.2f}")
    
    print(f"크기 통계:")
    print(f"  평균 너비: {np.mean(widths):.2f}")
    print(f"  평균 높이: {np.mean(heights):.2f}")
    print(f"  평균 종횡비: {np.mean(widths)/np.mean(heights):.2f}")
    
    # 각도별 분포
    angle_counts = defaultdict(int)
    for ann in annotations:
        angle = ann['camera_la']
        angle_counts[angle] += 1
    
    print(f"각도별 분포:")
    for angle, count in sorted(angle_counts.items()):
        print(f"  {angle}도: {count}개")
    
    return annotations

def visualize_drug_bbox_distribution(drug_annotations, top_n=10):
    """약품별 바운딩 박스 분포 시각화"""
    
    # 약품별 통계 계산
    stats = analyze_drug_bbox_statistics(drug_annotations)
    
    # 상위 N개 약품 선택
    top_drugs = stats.nlargest(top_n, 'count')
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 약품별 어노테이션 수
    axes[0, 0].barh(range(len(top_drugs)), top_drugs['count'])
    axes[0, 0].set_yticks(range(len(top_drugs)))
    axes[0, 0].set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                               for name in top_drugs['drug_name']])
    axes[0, 0].set_xlabel('어노테이션 수')
    axes[0, 0].set_title('약품별 어노테이션 수')
    
    # 2. 약품별 평균 면적
    axes[0, 1].barh(range(len(top_drugs)), top_drugs['avg_area'])
    axes[0, 1].set_yticks(range(len(top_drugs)))
    axes[0, 1].set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                               for name in top_drugs['drug_name']])
    axes[0, 1].set_xlabel('평균 면적 (픽셀²)')
    axes[0, 1].set_title('약품별 평균 바운딩 박스 면적')
    
    # 3. 면적 vs 어노테이션 수 산점도
    axes[1, 0].scatter(top_drugs['count'], top_drugs['avg_area'])
    for i, row in top_drugs.iterrows():
        axes[1, 0].annotate(row['drug_name'][:10], 
                           (row['count'], row['avg_area']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1, 0].set_xlabel('어노테이션 수')
    axes[1, 0].set_ylabel('평균 면적 (픽셀²)')
    axes[1, 0].set_title('어노테이션 수 vs 평균 면적')
    
    # 4. 종횡비 분포
    axes[1, 1].barh(range(len(top_drugs)), top_drugs['avg_aspect_ratio'])
    axes[1, 1].set_yticks(range(len(top_drugs)))
    axes[1, 1].set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                               for name in top_drugs['drug_name']])
    axes[1, 1].set_xlabel('평균 종횡비 (너비/높이)')
    axes[1, 1].set_title('약품별 평균 종횡비')
    
    plt.tight_layout()
    plt.show()

def main():
    """메인 실행 함수"""
    
    korean_font_setting()
    # 경로 설정
    TRAIN_ANNOTATIONS_PATH = 'data/raw_data/train_annotations'
    TRAIN_IMAGES_PATH = 'data/raw_data/train_images'
    
    print("=== 약품별 바운딩 박스 분석 시작 ===")
    
    # 1. 모든 어노테이션 로드
    print("\n1. 어노테이션 파일 로드 중...")
    drug_annotations, image_drug_mapping = load_all_annotations(TRAIN_ANNOTATIONS_PATH)
    
    print(f"총 {len(drug_annotations)}개 약품 발견")
    print(f"총 {len(image_drug_mapping)}개 이미지 발견")
    
    # 2. 약품별 통계 분석
    print("\n2. 약품별 통계 분석...")
    stats_df = analyze_drug_bbox_statistics(drug_annotations)
    print(f"통계 분석 완료: {len(stats_df)}개 약품")
    
    # 3. 상위 약품들 출력
    print("\n3. 상위 10개 약품 (어노테이션 수 기준):")
    top_drugs = stats_df.nlargest(10, 'count')
    for _, row in top_drugs.iterrows():
        print(f"  {row['drug_name']}: {row['count']}개")
    
    # 4. 다중 약품 이미지 찾기
    print("\n4. 다중 약품 이미지 찾기...")
    multi_drug_images = find_multi_drug_images(image_drug_mapping, min_drugs=2)
    print(f"다중 약품 이미지: {len(multi_drug_images)}개")
    
    if multi_drug_images:
        print("상위 5개 다중 약품 이미지:")
        for i, img_info in enumerate(multi_drug_images[:5]):
            print(f"  {i+1}. {img_info['image_name']}: {img_info['drug_count']}개 약품")
            for drug in img_info['drugs']:
                print(f"     - {drug['drug_name']}")
    
    # 5. 특정 약품 상세 분석 (예시)
    if drug_annotations:
        print("\n5. 특정 약품 상세 분석...")
        # 첫 번째 약품으로 예시
        first_drug_code = list(drug_annotations.keys())[0]
        analyze_specific_drug(drug_annotations, first_drug_code)
    
    # 6. 시각화
    print("\n6. 시각화 생성...")
    visualize_drug_bbox_distribution(drug_annotations, top_n=10)
    
    # 7. 다중 약품 이미지 시각화 (첫 번째 예시)
    if multi_drug_images:
        print("\n7. 다중 약품 이미지 시각화...")
        first_multi_drug = multi_drug_images[0]
        image_path = os.path.join(TRAIN_IMAGES_PATH, first_multi_drug['image_name'])
        
        if os.path.exists(image_path):
            visualize_multi_drug_image(image_path, first_multi_drug['drugs'])
        else:
            print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    
    print("\n=== 분석 완료 ===")
    
    return drug_annotations, image_drug_mapping, stats_df

if __name__ == "__main__":
    drug_annotations, image_drug_mapping, stats_df = main() 