import os
import json
from collections import defaultdict

def analyze_drug_annotation_coverage():
    """약품코드별 어노테이션 커버리지 분석"""
    
    TRAIN_ANNOTATIONS_PATH = 'data/raw_data/train_annotations'
    
    # 결과 저장용 딕셔너리
    drug_coverage = defaultdict(list)  # 약품코드별 등장 폴더들
    drug_annotations = defaultdict(list)  # 약품코드별 실제 어노테이션 폴더들
    folder_drug_mapping = {}  # 폴더별 약품코드 매핑
    
    # 모든 어노테이션 폴더 찾기
    annotation_folders = []
    for item in os.listdir(TRAIN_ANNOTATIONS_PATH):
        item_path = os.path.join(TRAIN_ANNOTATIONS_PATH, item)
        if os.path.isdir(item_path) and item.endswith('_json'):
            annotation_folders.append(item)
    
    print(f"총 {len(annotation_folders)}개의 어노테이션 폴더 발견")
    
    # 각 어노테이션 폴더 분석
    for folder in annotation_folders:
        folder_path = os.path.join(TRAIN_ANNOTATIONS_PATH, folder)
        
        # 폴더명에서 약품 코드들 추출
        folder_name = folder.replace('_json', '')
        parts = folder_name.split('-')
        
        # 폴더명의 약품 코드들
        folder_drug_codes = []
        if len(parts) >= 2:
            # 첫 번째 약품 코드
            folder_drug_codes.append(parts[1])  # K-001900 -> 001900
            
            # 나머지 약품 코드들
            for i in range(2, len(parts)):
                if len(parts[i]) == 6:  # 6자리 약품 코드
                    folder_drug_codes.append(parts[i])
        
        # 실제 하위 폴더의 약품 코드들
        actual_drug_codes = []
        for subitem in os.listdir(folder_path):
            subitem_path = os.path.join(folder_path, subitem)
            if os.path.isdir(subitem_path) and subitem.startswith('K-'):
                # K- 접두사 제거하여 약품 코드 추출
                drug_code = subitem[2:]  # K-001900 -> 001900
                actual_drug_codes.append(drug_code)
        
        # 각 약품코드별 정보 저장
        for drug_code in folder_drug_codes:
            drug_coverage[drug_code].append(folder)
            
            if drug_code in actual_drug_codes:
                drug_annotations[drug_code].append(folder)
        
        # 폴더별 매핑 저장
        folder_drug_mapping[folder] = {
            'folder_drugs': folder_drug_codes,
            'actual_drugs': actual_drug_codes
        }
    
    return drug_coverage, drug_annotations, folder_drug_mapping

def print_coverage_analysis(drug_coverage, drug_annotations, folder_drug_mapping):
    """커버리지 분석 결과 출력"""
    
    print("\n" + "="*80)
    print("약품코드별 어노테이션 커버리지 분석")
    print("="*80)
    
    # 1. 전체 약품코드 통계
    all_drugs = set(drug_coverage.keys())
    all_annotated_drugs = set(drug_annotations.keys())
    
    print(f"\n1. 전체 약품코드 통계:")
    print("-" * 50)
    print(f"  총 고유 약품코드: {len(all_drugs)}개")
    print(f"  어노테이션 있는 약품코드: {len(all_annotated_drugs)}개")
    print(f"  어노테이션 없는 약품코드: {len(all_drugs - all_annotated_drugs)}개")
    
    # 2. 어노테이션이 없는 약품코드들
    missing_drugs = all_drugs - all_annotated_drugs
    if missing_drugs:
        print(f"\n2. 어노테이션이 없는 약품코드들:")
        print("-" * 50)
        for drug in sorted(missing_drugs):
            folders = drug_coverage[drug]
            print(f"  {drug}: {len(folders)}개 폴더에서 등장")
            for folder in folders[:3]:  # 처음 3개만 표시
                print(f"    - {folder}")
            if len(folders) > 3:
                print(f"    ... 외 {len(folders)-3}개")
    
    # 3. 부분적으로 어노테이션이 있는 약품코드들
    partial_drugs = {}
    for drug in all_drugs:
        total_folders = len(drug_coverage[drug])
        annotated_folders = len(drug_annotations[drug])
        if total_folders > annotated_folders and annotated_folders > 0:
            partial_drugs[drug] = {
                'total': total_folders,
                'annotated': annotated_folders,
                'missing': total_folders - annotated_folders
            }
    
    if partial_drugs:
        print(f"\n3. 부분적으로 어노테이션이 있는 약품코드들:")
        print("-" * 50)
        for drug, stats in sorted(partial_drugs.items(), key=lambda x: x[1]['missing'], reverse=True):
            print(f"  {drug}: {stats['annotated']}/{stats['total']} ({stats['missing']}개 누락)")
    
    # 4. 완전히 어노테이션이 있는 약품코드들
    complete_drugs = {}
    for drug in all_drugs:
        total_folders = len(drug_coverage[drug])
        annotated_folders = len(drug_annotations[drug])
        if total_folders == annotated_folders and total_folders > 0:
            complete_drugs[drug] = total_folders
    
    print(f"\n4. 완전히 어노테이션이 있는 약품코드들:")
    print("-" * 50)
    print(f"  총 {len(complete_drugs)}개 약품코드")
    for drug, count in sorted(complete_drugs.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {drug}: {count}개 폴더")
    
    # 5. 특정 약품코드 상세 분석
    print(f"\n5. 주요 약품코드 상세 분석:")
    print("-" * 50)
    
    # 불일치에서 발견된 약품코드들
    problematic_drugs = ['010224', '003483', '012081', '022347', '002483', '003351', '003544', '016262', '019861', '025469', '027653', '027777', '036637']
    
    for drug in problematic_drugs:
        if drug in all_drugs:
            total_folders = len(drug_coverage[drug])
            annotated_folders = len(drug_annotations[drug])
            missing_folders = total_folders - annotated_folders
            
            print(f"\n  {drug}:")
            print(f"    총 등장: {total_folders}개 폴더")
            print(f"    어노테이션: {annotated_folders}개 폴더")
            print(f"    누락: {missing_folders}개 폴더")
            
            if missing_folders > 0:
                missing_folder_list = [f for f in drug_coverage[drug] if f not in drug_annotations[drug]]
                print(f"    누락된 폴더들:")
                for folder in missing_folder_list:
                    print(f"      - {folder}")

def analyze_specific_drug(drug_code, drug_coverage, drug_annotations, folder_drug_mapping):
    """특정 약품코드 상세 분석"""
    
    print(f"\n" + "="*80)
    print(f"약품코드 {drug_code} 상세 분석")
    print("="*80)
    
    if drug_code not in drug_coverage:
        print(f"약품코드 {drug_code}는 데이터셋에 존재하지 않습니다.")
        return
    
    total_folders = drug_coverage[drug_code]
    annotated_folders = drug_annotations[drug_code]
    missing_folders = [f for f in total_folders if f not in annotated_folders]
    
    print(f"\n1. 기본 통계:")
    print("-" * 50)
    print(f"  총 등장 폴더: {len(total_folders)}개")
    print(f"  어노테이션 폴더: {len(annotated_folders)}개")
    print(f"  누락 폴더: {len(missing_folders)}개")
    print(f"  커버리지: {len(annotated_folders)/len(total_folders)*100:.1f}%")
    
    if missing_folders:
        print(f"\n2. 누락된 폴더들:")
        print("-" * 50)
        for folder in missing_folders:
            folder_info = folder_drug_mapping[folder]
            print(f"  {folder}")
            print(f"    폴더명 약품: {folder_info['folder_drugs']}")
            print(f"    실제 약품: {folder_info['actual_drugs']}")
            print()
    
    if annotated_folders:
        print(f"\n3. 어노테이션이 있는 폴더들:")
        print("-" * 50)
        for folder in annotated_folders[:5]:  # 처음 5개만
            folder_info = folder_drug_mapping[folder]
            print(f"  {folder}")
            print(f"    폴더명 약품: {folder_info['folder_drugs']}")
            print(f"    실제 약품: {folder_info['actual_drugs']}")
            print()

def main():
    """메인 실행 함수"""
    
    print("약품코드별 어노테이션 커버리지 분석 시작...")
    
    # 1. 전체 분석
    drug_coverage, drug_annotations, folder_drug_mapping = analyze_drug_annotation_coverage()
    
    # 2. 결과 출력
    print_coverage_analysis(drug_coverage, drug_annotations, folder_drug_mapping)
    
    # 3. 특정 약품코드 상세 분석
    analyze_specific_drug('010224', drug_coverage, drug_annotations, folder_drug_mapping)
    analyze_specific_drug('003483', drug_coverage, drug_annotations, folder_drug_mapping)
    
    print("\n분석 완료!")

if __name__ == "__main__":
    main() 