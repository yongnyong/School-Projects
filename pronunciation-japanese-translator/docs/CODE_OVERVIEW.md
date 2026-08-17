# Code Overview

## 프로젝트 핵심

이 프로젝트는 한글로 입력한 일본어 발음을 문자 단위 시퀀스로 처리하고, 한국어 의미 문장을 생성하는 Seq2Seq 모델입니다. PyTorch로 Encoder, Attention, Decoder 구조를 직접 구현했고, BLEU Score를 통해 생성 결과를 평가합니다.

## 주요 파일 설명

| 파일 | 역할 |
| --- | --- |
| `model/seq2seq.py` | Encoder, Attention, AttentionDecoder, Seq2Seq 모델 정의 |
| `model/train.py` | 데이터 로드, 학습 루프, 검증 BLEU 계산, Early Stopping |
| `model/eval.py` | 테스트 데이터 기준 BLEU 평가 |
| `model/infer.py` | 학습된 모델을 불러와 사용자 입력을 추론 |
| `model/dialog_transation.py` | Papago 웹 자동화를 통한 발음/번역 데이터 수집 보조 |
| `utility/utils.py` | 문자 Vocabulary와 Dataset 클래스 |

## 모델 흐름

1. `CharVocab`이 발음 입력과 목표 문장을 문자 단위 토큰으로 변환합니다.
2. `PronunciationDataset`이 입력/정답 쌍을 PyTorch Dataset 형태로 제공합니다.
3. Encoder가 입력 시퀀스를 GRU hidden state로 인코딩합니다.
4. Attention Decoder가 입력 시퀀스의 중요 위치를 참고하며 다음 문자를 예측합니다.
5. 학습 중에는 Teacher Forcing을 적용해 수렴을 돕습니다.
6. 검증/테스트 단계에서는 BLEU Score로 예측 문장과 정답 문장의 유사도를 평가합니다.

## 발표자료에서 확인한 개선 흐름

초기 구조는 `한글 발음 -> 로마자 변환 -> 일본어 변환 -> 한국어 번역`의 3단계 구조였습니다. 하지만 중간 변환 과정에서 노이즈가 커지고, Seq2Seq와 mBART 조합의 결과가 안정적이지 않아 단순화된 구조로 전환했습니다.

최종 구조는 `일본어 문장 - 한글 발음 - 한국어 의미` 데이터를 직접 구성하고, 한글 발음 입력에서 의미를 바로 예측하는 Seq2Seq with Attention 모델입니다.

## 데이터 증강

표준 발음에 과적합되는 문제를 줄이기 위해 다음 증강을 적용했습니다.

- 발음 변형 노이즈
- 랜덤 삭제
- 랜덤 치환
- 원본 데이터 기준 5배 증강

발표 원본 기준 최종 모델의 BLEU Score는 0.5276입니다.

## 포트폴리오에서 강조할 점

- NLP 모델의 전체 파이프라인을 직접 구성했습니다.
- 단순 API 사용이 아니라 데이터 수집, 전처리, 모델링, 학습, 평가, 추론까지 구현했습니다.
- Attention 구조를 적용해 기본 Seq2Seq보다 입력 문맥 반영을 강화했습니다.
- TensorBoard 로그와 Early Stopping을 사용해 학습 과정을 관리했습니다.
- 표준 발음 입력에는 잘 맞지만 변형 발음에는 약한 문제를 발견했고, 데이터 증강으로 개선을 시도했습니다.

## GitHub 업로드 권장 구조

```text
pronunciation-based-translator/
├─ README.md
├─ model/
├─ utility/
├─ docs/
│  └─ presentation/
│     ├─ 발음톡_서비스_화면설계_포트폴리오.pptx
│     └─ 기말과제 발음기반 번역기.pdf
└─ .gitignore
```

## 업로드 시 주의

- `runs/model_best.pt`는 모델 파일이라 용량이 커질 수 있습니다.
- 학습 데이터에 개인정보나 저작권 문제가 있는 문장이 있다면 공개 저장소에 올리면 안 됩니다.
- Papago 자동화 코드는 포트폴리오 설명에는 좋지만, 실제 운영 기능처럼 보이게 과장하지 않는 것이 좋습니다.
