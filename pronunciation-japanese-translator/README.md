# Pronunciation-Based Japanese Translator

일본어를 직접 입력하기 어려운 상황에서, 사용자가 들리는 일본어 발음을 한글로 입력하면 한국어 의미를 예측하는 문자 단위 Seq2Seq 기반 번역 모델 프로젝트입니다.

기존 번역 API 결과를 그대로 보여주는 방식이 아니라, 일본어 문장 - 한글 발음 - 한국어 의미 데이터를 구성하고 PyTorch 기반 Encoder-Decoder 모델을 직접 학습해 발음 입력에서 의미를 생성하는 흐름을 구현했습니다.

## Project Summary

| 항목 | 내용 |
| --- | --- |
| 프로젝트명 | Pronunciation-Based Japanese Translator |
| 주제 | 한글 발음 입력 기반 일본어 의미 예측 |
| 핵심 기술 | PyTorch, Seq2Seq, GRU, Attention, BLEU 평가 |
| 구현 범위 | 데이터 수집, 전처리, 증강, 학습, 검증, 추론, 평가 |
| 최종 성능 | BLEU 0.5276 |
| 산출물 | 모델 코드, 학습 체크포인트, 발표자료 PDF/PPT |

## Background

해외 현장이나 일상 대화에서 외국어 문장을 들었지만 정확한 문자 입력이 어려운 경우가 있습니다. 예를 들어 일본어 문장 `なんですか`를 정확히 입력하지 못하더라도, 들리는 대로 `난닌 데스카`처럼 한글 발음으로 입력하면 의미를 추정할 수 있는 모델을 목표로 했습니다.

## Key Features

- 문자 단위 Vocabulary 구성
- 발음 입력과 목표 의미 문장 쌍을 Dataset으로 변환
- GRU 기반 Encoder-Decoder 모델 구현
- Attention Decoder를 적용해 입력 시퀀스의 주요 문자 흐름 반영
- 발음 오입력에 대응하기 위한 노이즈 기반 데이터 증강
- Teacher Forcing 기반 학습
- BLEU Score 기반 검증 및 테스트 평가
- Early Stopping으로 최적 모델 저장
- 학습된 모델을 활용한 CLI 추론 기능 제공

## Architecture

```text
Korean pronunciation input
        |
        v
CharVocab encoding
        |
        v
GRU Encoder
        |
        v
Attention Decoder
        |
        v
Korean meaning output
```

## Tech Stack

- Python
- PyTorch
- pandas
- NLTK BLEU
- TensorBoard
- Selenium
- Naver Papago

## Folder Structure

```text
seq2seq_2025_0528 (1)/
├─ model/
│  ├─ seq2seq.py            # Encoder, Attention, Decoder, Seq2Seq model
│  ├─ train.py              # training loop, validation, early stopping
│  ├─ eval.py               # test BLEU evaluation
│  ├─ infer.py              # command-line inference
│  └─ dialog_transation.py  # Papago-based data collection helper
├─ utility/
│  ├─ utils.py              # CharVocab, PronunciationDataset
│  └─ daraset-tuning.py     # dataset adjustment helper
└─ runs/
   ├─ model_best.pt         # trained model checkpoint
   ├─ src_vocab.pkl         # source vocabulary
   └─ tgt_vocab.pkl         # target vocabulary
```

## Main Implementation

### 1. Character Vocabulary

`CharVocab`는 입력 문장을 문자 단위로 분해해 `<pad>`, `<sos>`, `<eos>` 토큰과 함께 정수 인덱스로 변환합니다. 한글 발음 입력처럼 단어 단위보다 문자 흐름이 중요한 데이터에 맞춰 단순하고 직접적인 구조로 설계했습니다.

### 2. Seq2Seq Model

모델은 GRU 기반 Encoder와 Attention Decoder로 구성했습니다.

- Encoder: 입력 발음 시퀀스를 hidden state로 압축
- Attention: Decoder가 매 출력 시점마다 입력 시퀀스의 중요한 위치를 참조
- Decoder: 이전 출력과 attention context를 바탕으로 다음 문자를 예측

### 3. Training

`train.py`에서는 학습 데이터와 검증 데이터를 불러온 뒤, Teacher Forcing을 적용해 모델을 학습합니다. 검증 단계에서는 BLEU Score를 계산하고, 성능이 개선될 때마다 `model_best.pt`로 저장합니다.

### 4. Data Augmentation

발표 원본에서는 표준 발음에만 과적합되는 문제를 확인했고, 이를 줄이기 위해 발음 노이즈를 적용했습니다.

- 발음 변형 노이즈
- 랜덤 삭제
- 랜덤 치환
- 원본 데이터 기준 5배 증강

최종 모델은 다양한 노이즈를 함께 적용해 오입력 발음에 대한 대응력을 높이는 방향으로 구성했습니다.

### 5. Inference

`infer.py`는 학습된 모델과 vocabulary 파일을 불러와 사용자가 입력한 발음 문자열에 대한 예측 결과를 출력합니다.

## How to Run

```bash
cd "seq2seq_2025_0528 (1)/model"
python train.py
```

학습 후 추론:

```bash
cd "seq2seq_2025_0528 (1)/model"
python infer.py
```

테스트 평가:

```bash
cd "seq2seq_2025_0528 (1)/model"
python eval.py
```

## Portfolio Points

- 번역 API 호출에만 의존하지 않고, 직접 Seq2Seq 모델 구조를 구현했습니다.
- 발음 기반 입력을 문자 단위 데이터로 모델링해 NLP 전처리 과정을 경험했습니다.
- Attention 구조를 적용해 기본 Seq2Seq보다 입력 문맥 반영을 강화했습니다.
- BLEU Score와 Early Stopping을 통해 모델 성능을 검증하고 학습 과정을 관리했습니다.
- Selenium을 활용해 Papago 결과를 수집하고 학습 데이터 구성 흐름을 자동화했습니다.
- 표준 발음에 과적합되는 문제를 발견하고, 노이즈 증강으로 개선을 시도했습니다.

## Presentation

기존 발표자료는 아래 경로에 포함했습니다.

```text
docs/presentation/발음톡_서비스_화면설계_포트폴리오.pptx
docs/presentation/기말과제 발음기반 번역기.pdf
```

## Next Improvements

- 학습 데이터셋과 전처리 과정을 README에 더 명확히 정리
- 예측 결과 예시 추가
- Streamlit 또는 웹 화면을 붙여 시연 가능한 형태로 확장
- Transformer 계열 모델과 성능 비교
- 언어 감지 기능을 붙여 다국어 발음 입력으로 확장
