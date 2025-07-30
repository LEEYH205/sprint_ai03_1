```markdown
print_ai03_1/
├── data/
│   ├── raw_data/
│   │   ├── test_images/         – 원본 테스트 이미지 폴더
│   │   ├── train_annotations/   – 원본 학습용 어노테이션 (XML/JSON)
│   │   └── train_images/        – 원본 학습용 이미지
│   └── processed/               – YOLO 형식으로 변환된 train/val 데이터
├── models/
│   └── model.py                 – 모델 아키텍처 정의 (예: YOLO)
├── runs/
│   ├── train/                   – 학습 로그·체크포인트·TensorBoard
│   ├── detect/                  – detect.py 결과 이미지
│   └── evaluate/                – evaluate.py 성능 리포트
├── src/
│   ├── train.py                 – 모델 학습 메인 스크립트
│   ├── evaluate.py              – 학습된 모델 성능 평가
│   ├── inference.py             – NMS/TTA 포함 추론 스크립트
│   ├── utils.py                 – 공통 유틸(데이터 증강·라벨 파싱)
│   ├── visualization.py         – 학습·예측 시각화 도구
│   └── check.py                 – validation 이미지 순회 시각화용 툴
├── scripts/
│   ├── preprocess.py            – raw_data → YOLO train/val 분할·포맷 생성
│   ├── convert_yolo2coco.py     – YOLO TXT → COCO JSON 변환
│   ├── convert_subset.py        – COCO JSON에서 subset YOLO TXT 추출
│   ├── convert_csv2json.py      – 예측 CSV → COCO JSON 변환
│   ├── coco_eval.py             – COCO 툴킷 기반 성능 평가
│   ├── calibration_eval.py      – ECE 계산·Reliability Diagram 시각화
│   ├── collect_fn.py            – False Negative 박스 시각화용 수집 도구
│   └── train_curve.py           – results.csv 기반 학습 곡선 플롯
└── data.yaml                    – 데이터셋 설정 및 하이퍼파라미터
```
