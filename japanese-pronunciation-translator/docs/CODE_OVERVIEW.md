# Code Overview

## 프로젝트 핵심

한글로 입력한 일본어 발음을 문자 단위 시퀀스로 처리하고, 한국어 의미 문장을 생성하는 Seq2Seq with Attention 모델입니다.

## 주요 파일

| 파일 | 역할 |
| --- | --- |
| `model/seq2seq.py` | Encoder, Attention, AttentionDecoder, Seq2Seq 모델 정의 |
| `model/train.py` | 데이터 로드, 학습 루프, BLEU 검증, Early Stopping |
| `model/eval.py` | 테스트 데이터 기준 BLEU 평가 |
| `model/infer.py` | 학습된 모델을 불러와 사용자 입력 추론 |
| `model/dialog_transation.py` | Papago 웹 자동화를 통한 발음/번역 데이터 수집 보조 |
| `utility/utils.py` | 문자 Vocabulary와 Dataset 클래스 |

## 구현 흐름

1. 발음 입력과 목표 의미 문장을 문자 단위 토큰으로 변환합니다.
2. Encoder가 입력 시퀀스를 GRU hidden state로 인코딩합니다.
3. Attention Decoder가 입력 시퀀스의 중요 위치를 참조하며 다음 문자를 예측합니다.
4. Teacher Forcing을 적용해 학습 안정성을 높입니다.
5. BLEU Score를 기준으로 검증하고, 가장 좋은 모델을 저장합니다.

## 발표자료 기반 개선 과정

초기 구조는 `한글 발음 -> 로마자 변환 -> 일본어 변환 -> 한국어 번역`의 3단계 구조였습니다. 하지만 중간 변환 과정에서 노이즈가 커져, 최종적으로 `한글 발음 -> 한국어 의미`를 직접 예측하는 단순화된 Seq2Seq 구조로 전환했습니다.

## 데이터 증강

표준 발음에 과적합되는 문제를 줄이기 위해 다음 노이즈를 적용했습니다.

- 발음 변형
- 랜덤 삭제
- 랜덤 치환
- 원본 데이터 기준 5배 증강

발표 원본 기준 최종 BLEU Score는 0.5276입니다.
