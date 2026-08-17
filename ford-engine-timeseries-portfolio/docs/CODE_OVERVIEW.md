# Code Overview

## 프로젝트 핵심

FordA/FordB 엔진 시계열 데이터를 정상/이상으로 분류하는 머신러닝 프로젝트입니다. 핵심은 500차원 시계열을 그대로 쓰는 대신, 시계열의 통계/변화율/자기상관/주파수 특성을 추출해 tabular feature로 바꾸고 여러 분류 모델에서 성능을 비교한 점입니다.

## 주요 파일

| 파일 | 역할 |
| --- | --- |
| `src/preprocessing.py` | FordA 데이터 로드, ARFF/TXT 파싱, feature 추출 |
| `src/data_preprocess.py` | FordB 데이터 feature 추출 |
| `src/multi_model.py` | RandomForest, GBDT, XGBoost, LightGBM, SVM, Logistic Regression 비교 |
| `src/logistic_regression.py` | Logistic Regression baseline 평가 |
| `src/logistic_regression_label.py` | 라벨 기반 로지스틱 회귀 실험 보조 |
| `src/split.py` | 데이터 분할 보조 |

## Feature Engineering

한 샘플의 500개 시계열 값을 입력받아 다음 feature를 생성합니다.

- 기본 통계: 평균, 표준편차, 최소/최대, 중앙값, 분위수, IQR
- 분포: 왜도, 첨도
- 변화율: 1차 차분 평균, 차분 표준편차, 절대 변화량 평균
- 시계열: zero crossing, ACF lag 1/5/10/20/50, trend slope
- 주파수: spectral centroid, spectral entropy, dominant frequency, dominant power

## 모델 평가

`multi_model.py`는 각 모델에 대해 다음 지표를 출력합니다.

- AUC
- PR-AUC
- F1@0.5
- F1@best threshold
- Best threshold
- Confusion matrix

또한 tree 계열 모델은 feature importance를, 선형 모델은 coefficient 기반 중요 변수를 출력합니다.

## 발표자료 기반 결론

발표자료에서는 Raw, PCA, Feature Engineering 순서로 평균 성능이 상승했다고 정리했습니다. 최종 best AUC는 FordA 0.9716, FordB 0.8688로 Feature Engineering 방식에서 가장 좋았습니다.

## GitHub 업로드 시 제외한 항목

다음 파일은 용량 때문에 업로드용 패키지에서 제외했습니다.

- FordA/FordB 원본 CSV
- 전처리된 tabular CSV
- 실험 중간 산출물

데이터가 필요하면 로컬 원본 폴더 `C:\Users\user\PycharmProjects\산데분`에서 다시 확인하면 됩니다.
