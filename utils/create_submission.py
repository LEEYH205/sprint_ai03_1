import os
import json
import glob
import pandas as pd
from collections import defaultdict

def check_yolo_category_mapping(model_path):
    """YOLO 모델의 클래스 ID와 실제 카테고리 ID 매핑 확인"""
    
    print("=== YOLO 카테고리 ID 매핑 확인 ===")
    
    # 1. 실제 어노테이션에서 카테고리 ID 수집
    TRAIN_ANNOTATIONS_PATH = 'data/raw_data/train_annotations'
    
    print("1. 실제 어노테이션에서 카테고리 ID 수집 중...")
    
    real_category_ids = set()
    category_name_to_id = {}
    
    for item in os.listdir(TRAIN_ANNOTATIONS_PATH):
        item_path = os.path.join(TRAIN_ANNOTATIONS_PATH, item)
        if os.path.isdir(item_path) and item.endswith('_json'):
            for root, dirs, files in os.walk(item_path):
                for file in files:
                    if file.endswith('.json'):
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            if 'categories' in data and data['categories']:
                                category = data['categories'][0]
                                category_id = category['id']
                                category_name = category['name']
                                
                                real_category_ids.add(category_id)
                                category_name_to_id[category_name] = category_id
                                
                        except Exception as e:
                            continue
    
    print(f"실제 카테고리 ID 수: {len(real_category_ids)}개")
    print(f"실제 카테고리 ID 목록: {sorted(real_category_ids)}")
    
    # 2. YOLO 모델의 클래스 정보 확인
    print("\n2. YOLO 모델 클래스 정보 확인 중...")
    
    if os.path.exists(model_path):
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            
            # 모델의 클래스 정보 확인
            if hasattr(model, 'names'):
                yolo_class_names = model.names
                print(f"YOLO 클래스 수: {len(yolo_class_names)}개")
                print("YOLO 클래스 목록:")
                for i, name in yolo_class_names.items():
                    print(f"  {i}: {name}")
                
                # YOLO 클래스 이름과 실제 카테고리 이름 매핑
                print("\n3. 클래스 이름 매핑 확인:")
                for yolo_id, yolo_name in yolo_class_names.items():
                    if yolo_name in category_name_to_id:
                        real_id = category_name_to_id[yolo_name]
                        print(f"  YOLO ID {yolo_id} ({yolo_name}) -> 실제 ID {real_id}")
                    else:
                        print(f"  YOLO ID {yolo_id} ({yolo_name}) -> 매핑 없음!")
                
            else:
                print("YOLO 모델에서 클래스 정보를 찾을 수 없습니다.")
                
        except Exception as e:
            print(f"YOLO 모델 로드 오류: {e}")
    else:
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    return real_category_ids, category_name_to_id

def create_corrected_submission_file(model_path, test_dir):
    """올바른 카테고리 ID 매핑으로 제출 파일 생성"""
    
    print("\n=== 올바른 카테고리 ID 매핑으로 제출 파일 생성 ===")
    
    # 실제 카테고리 ID 수집
    real_category_ids, category_name_to_id = check_yolo_category_mapping(model_path)
    
    try:
        from ultralytics import YOLO
        import torch
        
        print("모델 로딩 중...")
        model = YOLO(model_path)
        
        # 디바이스 설정
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"사용 디바이스: {device}")
        
        # 테스트 이미지 경로 리스트
        image_paths = sorted(glob.glob(os.path.join(test_dir, '*.png')))
        print(f"발견된 테스트 이미지: {len(image_paths)}개")
        
        # 결과를 담아둘 리스트
        rows = []
        annotation_id = 1
        
        # 각 이미지에 대해 추론
        for i, img_path in enumerate(image_paths):
            # 이미지 파일명에서 숫자 추출
            img_filename = os.path.basename(img_path)
            img_name_without_ext = os.path.splitext(img_filename)[0]
            
            try:
                import re
                numbers = re.findall(r'\d+', img_name_without_ext)
                if numbers:
                    image_id = int(numbers[-1])
                else:
                    image_id = int(img_name_without_ext)
            except ValueError:
                image_id = i + 1
            
            print(f"처리 중: {img_filename} (image_id: {image_id})")
            
            # YOLO 예측 수행
            results = model.predict(source=img_path, conf=0.25, device=device)
            
            # 각 예측 결과를 행으로 추가
            for res in results:
                for box in res.boxes:
                    # YOLO 클래스 ID (0부터 시작)
                    yolo_class_id = int(box.cls[0].cpu().numpy())
                    
                    # YOLO 클래스 이름
                    yolo_class_name = model.names[yolo_class_id]
                    
                    # 실제 카테고리 ID로 매핑
                    if yolo_class_name in category_name_to_id:
                        category_id = category_name_to_id[yolo_class_name]
                    else:
                        print(f"⚠️  매핑되지 않은 클래스: {yolo_class_name}")
                        continue  # 매핑되지 않은 클래스는 건너뛰기
                    
                    # 신뢰도 점수
                    score = float(box.conf[0].cpu().numpy())
                    
                    # 바운딩 박스 좌표 (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # 바운딩 박스를 (x, y, w, h) 형식으로 변환
                    bbox_x = int(x1)
                    bbox_y = int(y1)
                    bbox_w = int(x2 - x1)
                    bbox_h = int(y2 - y1)
                    
                    # 행 추가
                    rows.append({
                        'annotation_id': annotation_id,
                        'image_id': image_id,
                        'category_id': category_id,
                        'bbox_x': bbox_x,
                        'bbox_y': bbox_y,
                        'bbox_w': bbox_w,
                        'bbox_h': bbox_h,
                        'score': score
                    })
                    
                    annotation_id += 1
        
        # DataFrame 생성 및 CSV 저장
        print(f"총 {len(rows)}개의 예측 결과 생성")
        
        if rows:
            submission = pd.DataFrame(rows, columns=[
                'annotation_id', 'image_id', 'category_id', 
                'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score'
            ])
            
            # CSV 저장
            output_path = 'submission_corrected.csv'
            submission.to_csv(output_path, index=False)
            print(f"수정된 제출 파일 저장 완료: {output_path}")
            
            # 결과 미리보기
            print("\n=== 수정된 제출 파일 미리보기 ===")
            print(submission.head(10))
            
            # 통계 정보
            print(f"\n=== 통계 정보 ===")
            print(f"총 예측 수: {len(submission)}개")
            print(f"고유 이미지 수: {submission['image_id'].nunique()}개")
            print(f"고유 카테고리 수: {submission['category_id'].nunique()}개")
            print(f"평균 신뢰도: {submission['score'].mean():.3f}")
            
            # 카테고리 ID 분포 확인
            print(f"\n=== 카테고리 ID 분포 ===")
            category_counts = submission['category_id'].value_counts().head(10)
            for cat_id, count in category_counts.items():
                print(f"카테고리 ID {cat_id}: {count}개")
            
            return submission
        else:
            print("⚠️  예측 결과가 없습니다!")
            return None
            
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

if __name__ == "__main__":
            
    # YOLO 모델 로드
    model_path = './models/drug_detection_model_YOLOv5x_b8_e100(88)_20250728_3.pt'
    test_dir = './data/raw_data/test_images'

    # 카테고리 매핑 확인
    real_category_ids, category_name_to_id = check_yolo_category_mapping(model_path)

    # 수정된 제출 파일 생성
    submission = create_corrected_submission_file(model_path, test_dir) 