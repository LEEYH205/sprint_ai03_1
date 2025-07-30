# Sprint AI Project
- 경구약제 이미지 객체 검출(Object Detection) 프로젝트


## 📁 Project Structure

```
sprint_ai03_1/
├── data/                               (→ git ignored)
│   ├── raw_data/                       (→ git ignored)
│   └── data_gjy/                       (→ git ignored)
│   └── data_khn/                       (→ git ignored)
│   └── data_yye/                       (→ git ignored)
│   └── data_lyh/                       (→ git ignored)
│   └── data_jmj/                       (→ git ignored)
│   └── data_final/                     (not yet, git ignored, google drive link)
├── models/                             (→ git ignored)
├── notebooks/
│   └── data_preprocessing_gjy.ipynb    (→ personal)
│   └── data_preprocessing_khn.ipynb    (→ personal)
│   └── data_preprocessing_yye.ipynb    (→ personal)
│   └── data_preprocessing_lyh.ipynb    (→ personal)
│   └── data_preprocessing_jmj.ipynb    (→ personal)
├── scripts/
│   ├── preprocess.py            – raw_data → YOLO train/val 분할·포맷 생성
│   ├── convert_yolo2coco.py     – YOLO TXT → COCO JSON 변환
│   ├── convert_subset.py        – COCO JSON에서 subset YOLO TXT 추출
│   ├── convert_csv2json.py      – 예측 CSV → COCO JSON 변환
│   ├── coco_eval.py             – COCO 툴킷 기반 성능 평가
│   ├── calibration_eval.py      – ECE 계산·Reliability Diagram 시각화
│   ├── collect_fn.py            – False Negative 박스 시각화용 수집 도구
│   └── train_curve.py           – results.csv 기반 학습 곡선 플롯
├── src/
│   ├── train.py                 – 모델 학습 메인 스크립트
│   ├── evaluate.py              – 학습된 모델 성능 평가
│   ├── inference.py             – NMS/TTA 포함 추론 스크립트
│   ├── utils.py                 – 공통 유틸(데이터 증강·라벨 파싱)
│   ├── visualization.py         – 학습·예측 시각화 도구
│   └── check.py                 – validation 이미지 순회 시각화용 툴
├── utils/
│   ├── analyze_annotation_mismatch.py        – 폴더명과 실제 하위 폴더 불일치 분석
│   ├── analyze_drug_annotation_coverage.py   – 약품코드별 어노테이션 커버리지 분석
│   ├── analyze_drug_bbox.py                  – 바운딩 박스 통계 및 시각화
│   ├── bbox_gui_editor.py                    – 바운딩 박스 편집 GUI
│   ├── drug_code_viewer.py                   – 약품 코드별 이미지 뷰어
│   └── data_augmentation.py                  – 이미지 회전을 통한 데이터 증강
│   └── create_submission.py                  - YOLO 모델 예측 및 제출 파일 생성
├── .gitignore
├── README.md
└── requirements.txt                    (not yet)
└── requirements.yaml                   (not yet)
```

## 📂 Directory Description

### `data/`
- **`raw_data/`**: 원본 데이터 파일들 (용량으로 미업로드) (이미지, 어노테이션 등)
- **`data_final/`**: 전처리된 최종 데이터 (용량으로 미업로드)

### `models/`
- 학습된 모델 파일들을 저장하는 디렉토리 (용량으로 미업로드)

### `notebooks/`
- 데이터 전처리 / 모델링 주피터 노트북 (개인용)

### `src/`
- **`evaluate.py`**: 모델 평가 관련 테스트 코드
- **`train.py`**: 모델 학습 관련 테스트 코드
- **`utils.py`**: 유틸리티 함수 테스트 코드
- **`inference.py`**: NMS/TTA 포함 추론 스크립트
- ** `visualization.py`**: 학습·예측 시각화 도구
- ** `check.py`**: validation 이미지 순회 시각화용 툴

### Root Files
- **`.gitignore`**: Git에서 제외할 파일/폴더 설정
- **`README.md`**: 프로젝트 설명서
- **`requirements.txt`**:  Python 패키지 의존성 목록
- **`requirements.yaml`**: Python 패키지 의존성 목록

### 전처리 툴 배포
- 제작자 : 이영호
- 링크 : https://github.com/LEEYH205/bbox-annotation-tools
- PyPi : https://pypi.org/project/bbox-annotation-tools/

### 보고서 링크
- ~~
  
### 노션 링크
- 팀 노션 : https://www.notion.so/Codeit-AI-3-_-1-_-23155af55ff6802898a1ed2a7052caf8
- 공지연 (모델링) : https://www.notion.so/Codeit-AI-3-_-Part2_1-_-_-Daily-23290068d16d80dd8b4cef8e763f36f6
- 김하나 (전처리) : https://www.notion.so/232bea5ebd7c80a2b9ebf4dc95703d01?v=232bea5ebd7c8087abd6000cd5265d34
- 유영은 (전처리) : https://www.notion.so/Codeit-AI-3-_-Part2_1-_-_-Daily-2315954c5686807f9839f52aae3eef7c
- 이영호 (팀장, 모델링) : https://www.notion.so/Codeit-AI-3-_-Part2_1-_-_-Daily_-23155af55ff680b2a0acee07f8e65d15
- 지민종 (파인튜닝) : https://www.notion.so/Codeit-AI-3-_-Part2_1-_-Daily_-2318c9c2de22801cba17ee3d6a45ce0c


