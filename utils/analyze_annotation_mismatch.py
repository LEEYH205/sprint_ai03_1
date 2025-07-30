import os
import json
from collections import defaultdict

def analyze_annotation_mismatch():
    """폴더명과 실제 하위 폴더의 불일치 분석"""
    
    TRAIN_ANNOTATIONS_PATH = 'data/raw_data/train_annotations'
    
    # 결과 저장용 딕셔너리
    folder_stats = defaultdict(dict)
    mismatch_stats = defaultdict(int)
    drug_count_distribution = defaultdict(int)
    
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
        
        # 통계 계산
        folder_drug_count = len(folder_drug_codes)
        actual_drug_count = len(actual_drug_codes)
        
        # 불일치 확인
        missing_in_folder = [code for code in folder_drug_codes if code not in actual_drug_codes]
        extra_in_actual = [code for code in actual_drug_codes if code not in folder_drug_codes]
        
        # 결과 저장
        folder_stats[folder] = {
            'folder_drug_codes': folder_drug_codes,
            'actual_drug_codes': actual_drug_codes,
            'folder_count': folder_drug_count,
            'actual_count': actual_drug_count,
            'missing_in_folder': missing_in_folder,
            'extra_in_actual': extra_in_actual,
            'is_match': len(missing_in_folder) == 0 and len(extra_in_actual) == 0
        }
        
        # 분포 기록
        key = f"{actual_drug_count}/{folder_drug_count}"
        mismatch_stats[key] += 1
        drug_count_distribution[folder_drug_count] += 1
        
        # 불일치가 있는 경우 출력
        if not folder_stats[folder]['is_match']:
            print(f"\n불일치 발견: {folder}")
            print(f"  폴더명 약품: {folder_drug_codes}")
            print(f"  실제 약품: {actual_drug_codes}")
            if missing_in_folder:
                print(f"  폴더명에만 있는 약품: {missing_in_folder}")
            if extra_in_actual:
                print(f"  실제에만 있는 약품: {extra_in_actual}")
    
    return folder_stats, mismatch_stats, drug_count_distribution

def print_mismatch_analysis(folder_stats, mismatch_stats, drug_count_distribution):
    """불일치 분석 결과 출력"""
    
    print("\n" + "="*80)
    print("폴더명과 실제 하위 폴더 불일치 분석 결과")
    print("="*80)
    
    # 1. 전체 통계
    total_folders = len(folder_stats)
    matching_folders = sum(1 for stats in folder_stats.values() if stats['is_match'])
    mismatching_folders = total_folders - matching_folders
    
    print(f"\n1. 전체 통계:")
    print("-" * 50)
    print(f"  총 폴더 수: {total_folders}개")
    print(f"  일치하는 폴더: {matching_folders}개 ({matching_folders/total_folders*100:.1f}%)")
    print(f"  불일치하는 폴더: {mismatching_folders}개 ({mismatching_folders/total_folders*100:.1f}%)")
    
    # 2. 약품 개수별 분포
    print(f"\n2. 폴더명의 약품 개수별 분포:")
    print("-" * 50)
    for drug_count, count in sorted(drug_count_distribution.items()):
        print(f"  {drug_count}개 약품: {count}개 폴더")
    
    # 3. 실제/폴더명 비율별 통계
    print(f"\n3. 실제/폴더명 약품 개수 비율:")
    print("-" * 50)
    for ratio, count in sorted(mismatch_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ratio} (실제/폴더명): {count}개 폴더")
    
    # 4. 불일치 상세 분석
    print(f"\n4. 불일치 상세 분석:")
    print("-" * 50)
    
    missing_patterns = defaultdict(int)
    extra_patterns = defaultdict(int)
    
    for folder, stats in folder_stats.items():
        if not stats['is_match']:
            if stats['missing_in_folder']:
                missing_key = f"폴더명에만 {len(stats['missing_in_folder'])}개"
                missing_patterns[missing_key] += 1
            
            if stats['extra_in_actual']:
                extra_key = f"실제에만 {len(stats['extra_in_actual'])}개"
                extra_patterns[extra_key] += 1
    
    print("  폴더명에만 있는 약품 패턴:")
    for pattern, count in sorted(missing_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"    {pattern}: {count}개 폴더")
    
    print("  실제에만 있는 약품 패턴:")
    for pattern, count in sorted(extra_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"    {pattern}: {count}개 폴더")

def analyze_specific_examples(folder_stats):
    """구체적인 예시 분석"""
    
    print("\n" + "="*80)
    print("구체적인 예시 분석")
    print("="*80)
    
    # 3개 약품 폴더 예시
    three_drug_examples = []
    four_drug_examples = []
    
    for folder, stats in folder_stats.items():
        if stats['folder_count'] == 3:
            three_drug_examples.append((folder, stats))
        elif stats['folder_count'] == 4:
            four_drug_examples.append((folder, stats))
    
    print(f"\n1. 3개 약품 폴더 예시 (총 {len(three_drug_examples)}개):")
    print("-" * 50)
    for i, (folder, stats) in enumerate(three_drug_examples[:5]):  # 상위 5개만
        print(f"  예시 {i+1}: {folder}")
        print(f"    폴더명 약품: {stats['folder_drug_codes']}")
        print(f"    실제 약품: {stats['actual_drug_codes']}")
        if not stats['is_match']:
            print(f"    불일치: {stats['missing_in_folder']} / {stats['extra_in_actual']}")
        print()
    
    print(f"\n2. 4개 약품 폴더 예시 (총 {len(four_drug_examples)}개):")
    print("-" * 50)
    for i, (folder, stats) in enumerate(four_drug_examples[:5]):  # 상위 5개만
        print(f"  예시 {i+1}: {folder}")
        print(f"    폴더명 약품: {stats['folder_drug_codes']}")
        print(f"    실제 약품: {stats['actual_drug_codes']}")
        if not stats['is_match']:
            print(f"    불일치: {stats['missing_in_folder']} / {stats['extra_in_actual']}")
        print()

def analyze_drug_code_patterns(folder_stats):
    """약품 코드 패턴 분석"""
    
    print("\n" + "="*80)
    print("약품 코드 패턴 분석")
    print("="*80)
    
    # 모든 약품 코드 수집
    all_folder_drugs = set()
    all_actual_drugs = set()
    
    for stats in folder_stats.values():
        all_folder_drugs.update(stats['folder_drug_codes'])
        all_actual_drugs.update(stats['actual_drug_codes'])
    
    # 교집합과 차집합
    common_drugs = all_folder_drugs & all_actual_drugs
    only_in_folder = all_folder_drugs - all_actual_drugs
    only_in_actual = all_actual_drugs - all_folder_drugs
    
    print(f"\n1. 약품 코드 통계:")
    print("-" * 50)
    print(f"  폴더명에만 있는 약품: {len(only_in_folder)}개")
    print(f"  실제에만 있는 약품: {len(only_in_actual)}개")
    print(f"  공통 약품: {len(common_drugs)}개")
    print(f"  총 고유 약품: {len(all_folder_drugs | all_actual_drugs)}개")
    
    if only_in_folder:
        print(f"\n2. 폴더명에만 있는 약품 코드들:")
        print("-" * 50)
        for drug in sorted(only_in_folder):
            print(f"  {drug}")
    
    if only_in_actual:
        print(f"\n3. 실제에만 있는 약품 코드들:")
        print("-" * 50)
        for drug in sorted(only_in_actual):
            print(f"  {drug}")

def main():
    """메인 실행 함수"""
    
    print("폴더명과 실제 하위 폴더 불일치 분석 시작...")
    
    # 1. 전체 분석
    folder_stats, mismatch_stats, drug_count_distribution = analyze_annotation_mismatch()
    
    # 2. 결과 출력
    print_mismatch_analysis(folder_stats, mismatch_stats, drug_count_distribution)
    
    # 3. 구체적인 예시 분석
    analyze_specific_examples(folder_stats)
    
    # 4. 약품 코드 패턴 분석
    analyze_drug_code_patterns(folder_stats)
    
    print("\n분석 완료!")

if __name__ == "__main__":
    main() 