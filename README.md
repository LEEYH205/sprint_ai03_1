# 💊 Pill Detection with YOLOv8 | AI03 Level1 Team Project

> AI 엔지니어링 교육과정 Level 1 팀 프로젝트  
> YOLOv8을 활용한 알약 이미지 객체 탐지 프로젝트

---

## 프로젝트 개요

- **목표**: 알약 이미지에서 각 알약의 위치를 정확히 찾아내고, 어떤 종류인지 분류하는 객체 탐지 모델을 학습시킴
- **방법**: YOLOv8 모델을 기반으로 학습 → 예측 → 평가 → 최종 제출까지 전체 파이프라인 구성
- **사용 기술**: Python, OpenCV, COCO Format, YOLOv8, Ultralytics, Kaggle, Matplotlib

---

## 데이터 구성

- **출처**: [Kaggle - AI03 Level1 Project](https://www.kaggle.com/competitions/ai03-level1-project/data)
- **구성**:
<pre> ```
├── train_iamges/  
│     ├── images/  # train 이미지
│─── train_annotations/ # 하위폴더별 약품코드 json 포함
├── test_images/
│     ├── images/  # train 이미지

```</pre>


- **Annotation Format**: COCO JSON

---

## 데이터 전처리

### COCO 포맷 통합

- `train_annotations` 하위 폴더에 흩어져 있던 `.json` 파일을 하나의 COCO 파일(`train_merge_coco.json`)로 통합
- `os.listdir()` 기반으로 폴더 탐색 및 병합
- 최종 수치:
- 이미지 수: **1,489장**
- 어노테이션 수: **4,526개**
- 클래스 수: **74종**

### 라벨 누락 처리

- 어노테이션 없는 이미지(`annotations == []`) 자동 탐지
- 총 **843장**의 라벨 없는 이미지 탐색 및 시각화 (랜덤 10장, 3xN 형식)

---

## 데이터 분석 및 증강 전략

- 클래스별 이미지 수 시각화 → **데이터 불균형** 존재 확인
- 클래스가 너무 많은 이미지 제거 또는 증강 제외 전략 수립
- Augmentation 기법: `flip`, `scale`, `mosaic`, `translate` 등 적용

---

## 모델 학습 - YOLOv8

### 실험 모델

- `YOLOv8s`, `YOLOv8m`, `YOLOv8l` 비교 실험

### 학습 설정

| 항목 | 설정값 |
|------|--------|
| 이미지 크기 | 640 |
| Epochs | 20 ~ 150 |
| Batch size | 16 |
| Optimizer | SDG / Adam |
| Early Stopping | 5 ~ 10 |
| Cosine LR | True |
| Augmentation | 유무 |

### 성능 지표

- `mAP50`, `mAP75`, `Precision`, `Recall`, `F1 Score`
- 모델별 성능 변화를 다양한 그래프로 시각화하여 비교

---

## 예측 및 제출

- `model.predict()`로 test 이미지 예측 및 시각화
- 예측 결과를 지정된 **제출 포맷 CSV**로 변환
- Conf. Threshold: 0.25

---


## 주요 성과

- 라벨 누락 이미지 자동 탐지 및 정리 → 학습 품질 향상
- 데이터 불균형 문제 대응 → 클래스 기준 필터링 및 증강 전략 구성
- 여러 YOLOv8 버전 실험 → 최적 성능 모델 도출
- COCO 포맷 완전 정제 → 학습/평가 통일된 데이터 기반 확보

---

## 사용 도구

- Python 3.10
- YOLOv8 (Ultralytics)
- OpenCV / PIL
- Matplotlib / Plotly
- Kaggle / Google Colab / VS Code

---

## 향후 계획

- 라벨 누락 이미지 기반 **pseudo-labeling** 적용 실험
- 앙상블 기법 적용 가능성 탐색
- 모델 서빙 및 추론 속도 개선

---

## 참고 링크

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com)
- [COCO Format Guide](https://cocodataset.org/#format-data)

---



