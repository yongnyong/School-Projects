# Korean Toxic Comment Filter

한국어 채팅/댓글 문장에서 욕설, 혐오, 비방성 표현을 탐지하기 위한 NLP 텍스트 분류 프로젝트입니다.

UnSmile과 HateScore 계열 데이터를 활용해 한국어 혐오표현 분류 데이터를 구성하고, KLUE BERT/RoBERTa, ALBERT, KoELECTRA 기반 모델을 비교 실험했습니다. 또한 텍스트 단독 분류 모델뿐 아니라 감정 분석 모델에서 추출한 emotion feature를 결합하는 구조도 구현했습니다.

## Project Summary

| 항목 | 내용 |
| --- | --- |
| 프로젝트명 | Korean Toxic Comment Filter |
| 주제 | 한국어 악성/혐오 댓글 탐지 |
| 분야 | NLP, Text Classification, Hate Speech Detection |
| 핵심 기술 | PyTorch, Hugging Face Transformers, KLUE BERT/RoBERTa, KoELECTRA, ALBERT |
| 데이터 | UnSmile, HateScore 기반 한국어 혐오표현 데이터 |
| 분류 방식 | 7-class text classification |
| 평가 지표 | Accuracy, Macro F1, LRAP |

## Problem

온라인 채팅과 댓글 환경에서는 욕설, 비방, 혐오 표현이 사용자 경험을 해치고 커뮤니티 안전성을 낮춥니다. 이 프로젝트는 한국어 문장을 입력받아 유해 표현 여부와 유형을 분류하고, 채팅 서비스에서 전송 차단 또는 경고에 활용할 수 있는 모델을 목표로 했습니다.

## Classes

프로젝트에서는 세부 혐오 카테고리를 포트폴리오/실험용 7개 범주로 재구성했습니다.

```text
0. clean
1. 악플/욕설
2. 성차별
3. 연령차별
4. 종교차별
5. 지역차별
6. 기타 혐오
```

원본 데이터의 세부 라벨은 실험 코드에서 단일 7-class 라벨로 변환됩니다.

## Key Features

- UnSmile, HateScore 데이터 전처리 및 통합
- 다중 라벨 형태의 혐오표현 데이터를 단일 7-class 분류 문제로 변환
- 텍스트 단독 분류 모델 구현
- 감정 feature를 결합한 분류 모델 구현
- BERT, RoBERTa, ALBERT, ELECTRA 계열 모델 비교 실험
- Hugging Face `Trainer` 기반 학습/검증/테스트 파이프라인 구성
- Accuracy, Macro F1, LRAP 지표 계산
- TensorBoard 기반 학습 로그 기록

## Model Structure

### Text-Only Model

```text
Korean text
    |
Tokenizer
    |
Transformer encoder
    |
Classification head
    |
7-class prediction
```

### Text + Emotion Feature Model

```text
Korean text
    |
Tokenizer
    |
Transformer encoder -----------------
                                  concat -> classifier -> 7-class prediction
Emotion feature extractor -> MLP ----
```

감정 feature는 `monologg/koelectra-base-v3-goemotions` 모델을 사용해 문장별 emotion score vector로 추출하고, 텍스트 표현과 결합합니다.

## Folder Structure

```text
korean-toxic-comment-filter/
├─ config/
│  ├─ text_only_7cls.yaml
│  └─ text_emo_7cls.yaml
├─ scripts/
│  ├─ train.py
│  ├─ experiments.py
│  ├─ experiments_text_only.py
│  └─ summarize_experiments.py
├─ src/
│  ├─ dataset.py
│  ├─ dataset_text_only.py
│  ├─ emotion.py
│  └─ model.py
├─ utils/
│  ├─ make_combined_train.py
│  ├─ hatescore_unsmile_concat.py
│  └─ utils.py
└─ docs/
   └─ source/
      └─ nlp_termproject.zip
```

## Main Implementation

### 1. Dataset Processing

`utils/make_combined_train.py`와 `utils/hatescore_unsmile_concat.py`는 UnSmile과 HateScore 데이터를 같은 라벨 체계로 맞추고, 학습에 사용할 CSV를 생성합니다.

### 2. Text-Only Classification

`scripts/experiments_text_only.py`는 BERT, RoBERTa, ALBERT, ELECTRA 기반 모델을 반복 실험할 수 있도록 구성했습니다. 각 실험은 learning rate, batch size, epoch 조합별로 실행됩니다.

### 3. Emotion Feature Fusion

`src/emotion.py`에서 감정 분석 모델을 이용해 문장별 emotion vector를 생성하고, `src/model.py`의 `TextEmotionClassifier`가 Transformer text representation과 emotion MLP output을 결합해 최종 분류를 수행합니다.

### 4. Metrics

`utils/utils.py`는 다음 지표를 계산합니다.

- Accuracy
- Macro F1
- LRAP

## How to Run

환경 설치:

```bash
pip install torch transformers pandas scikit-learn pyyaml tensorboard tqdm
```

텍스트 단독 실험:

```bash
python scripts/experiments_text_only.py
```

감정 feature 결합 모델 학습:

```bash
python scripts/train.py
```

실험 결과 요약:

```bash
python scripts/summarize_experiments.py
```

## Portfolio Points

- 한국어 혐오표현 데이터셋을 직접 통합하고 라벨 체계를 재구성했습니다.
- 단일 모델 학습이 아니라 여러 Transformer 계열 모델을 비교 실험하는 구조를 만들었습니다.
- 텍스트 representation에 감정 feature를 결합하는 확장 모델을 구현했습니다.
- Accuracy만 보지 않고 Macro F1과 LRAP를 함께 사용해 불균형 다중 클래스 분류 문제를 평가했습니다.
- 채팅 서비스의 악성 댓글 필터링 기능으로 확장 가능한 NLP 모델 파이프라인을 구성했습니다.

## Notes

- `data/`, `raw_data/`, `models/`, `log/`는 용량과 데이터 라이선스 문제 때문에 GitHub 업로드용 패키지에서는 제외했습니다.
- 원본 전체 자료는 `docs/source/nlp_termproject.zip`에 보관했습니다.
- 발표자료 파일이 확인되면 `docs/presentation/`에 추가해 README 링크를 보강할 수 있습니다.

## Recommended Repository Name

```text
korean-toxic-comment-filter
```
