# Code Overview

## 프로젝트 핵심

이 프로젝트는 한국어 문장을 7개 카테고리로 분류하는 악성/혐오 댓글 필터링 모델입니다. Hugging Face Transformers 기반 텍스트 분류 모델을 사용하며, 텍스트 단독 모델과 감정 feature 결합 모델을 모두 실험할 수 있게 구성되어 있습니다.

## 주요 파일

| 파일 | 역할 |
| --- | --- |
| `src/model.py` | Transformer text representation과 emotion feature를 결합하는 분류 모델 |
| `src/dataset.py` | 텍스트와 감정 feature를 함께 반환하는 Dataset |
| `src/dataset_text_only.py` | 텍스트 단독 분류용 Dataset |
| `src/emotion.py` | KoELECTRA GoEmotions 모델 기반 감정 feature 추출 |
| `scripts/train.py` | 감정 feature 결합 모델 학습 |
| `scripts/experiments.py` | 여러 모델 조합 실험 |
| `scripts/experiments_text_only.py` | 텍스트 단독 Transformer 모델 비교 실험 |
| `scripts/summarize_experiments.py` | 실험 결과 CSV 요약 |
| `utils/utils.py` | config 로드, seed 고정, metric 계산 |
| `utils/make_combined_train.py` | UnSmile + HateScore 데이터 통합 |

## 구현 흐름

1. UnSmile과 HateScore 데이터를 같은 라벨 체계로 맞춥니다.
2. 다중 라벨 컬럼을 7-class 단일 라벨로 변환합니다.
3. Tokenizer로 문장을 input ids와 attention mask로 변환합니다.
4. 텍스트 단독 모델 또는 감정 feature 결합 모델을 학습합니다.
5. Accuracy, Macro F1, LRAP로 성능을 평가합니다.
6. TensorBoard와 CSV 로그로 실험 결과를 남깁니다.

## 모델 비교

코드에는 다음 계열 모델 실험 구성이 포함되어 있습니다.

- `klue/bert-base`
- `klue/roberta-base`
- `kykim/albert-kor-base`
- `monologg/koelectra-base-v3-discriminator`

## 감정 Feature 결합 구조

`TextEmotionClassifier`는 Transformer encoder의 문장 표현과 별도 감정 분석 모델에서 추출한 emotion score vector를 결합합니다.

```text
Transformer hidden representation
        +
Emotion MLP representation
        |
        v
Classification head
```

이 구조는 단순 텍스트 분류에서 놓칠 수 있는 감정적 신호를 보조 feature로 활용하기 위한 실험입니다.

## GitHub 업로드 시 제외한 항목

다음 항목은 공개 저장소에 그대로 올리기 부적절해 제외했습니다.

- 원본 데이터 CSV/XLSX
- 학습된 모델 파일
- TensorBoard log
- 캐시 파일

필요하면 `docs/source/nlp_termproject.zip`의 원본 자료에서 다시 복구할 수 있습니다.
