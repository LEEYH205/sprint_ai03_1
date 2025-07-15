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
├── models/                             (not yet)
├── notebooks/
│   └── data_preprocessing_gjy.ipynb    (→ personal)
│   └── data_preprocessing_khn.ipynb    (→ personal)
│   └── data_preprocessing_yye.ipynb    (→ personal)
│   └── data_preprocessing_lyh.ipynb    (→ personal)
│   └── data_preprocessing_jmj.ipynb    (→ personal)
├── src/                                (not yet)
│   ├── test_evaluate.py                (git test file)
│   ├── test_train.py                   (git test file)
│   └── test_utils.py                   (git test file)
├── .gitignore
├── README.md
└── requirements.txt                    (not yet)
└── requirements.yaml                   (not yet)
```

## 📂 Directory Description

### `data/`
- **`raw_data/`**: 원본 데이터 파일들 (이미지, 어노테이션 등)
- **`data_final/`**: 전처리된 최종 데이터

### `models/`
- 학습된 모델 파일들을 저장하는 디렉토리

### `notebooks/`
- 데이터 전처리 노트북 (개인용)

### `src/`
- **`evaluate.py`**: 모델 평가 관련 테스트 코드
- **`train.py`**: 모델 학습 관련 테스트 코드
- **`utils.py`**: 유틸리티 함수 테스트 코드

### Root Files
- **`.gitignore`**: Git에서 제외할 파일/폴더 설정
- **`README.md`**: 프로젝트 설명서
- **`requirements.txt`**:  Python 패키지 의존성 목록
- **`requirements.yaml`**: Python 패키지 의존성 목록
