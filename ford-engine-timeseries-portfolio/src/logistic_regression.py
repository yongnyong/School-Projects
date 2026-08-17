# logistic_regression_fordA.py
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    classification_report, confusion_matrix, precision_recall_curve
)

# 1) 파일 로드
BASE = Path(r"C:\Users\user\PycharmProjects\산데분")
train = pd.read_csv(BASE / "fordB_train_tabular.csv")
test  = pd.read_csv(BASE / "fordB_tabular.csv")
test  = test.loc[test["split"] == "test"].copy()   # 안전장치

# 2) 입력/타깃 분리
X_tr = train.drop(columns=["label", "split"])
y_tr = (train["label"] == 1).astype(int)

X_te = test.drop(columns=["label", "split"])
y_te = (test["label"] == 1).astype(int)

# 3) 결측 안전 처리
X_tr = X_tr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_te = X_te.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# 4) 파이프라인: 스케일러 + 로지스틱 회귀
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        solver="liblinear",      # 소규모/특징수 적당할 때 안정적
        C=1.0,
        max_iter=200,
        random_state=42
    ))
])

# 5) 학습
pipe.fit(X_tr, y_tr)

# 6) 평가(기본 임계값 0.5)
proba = pipe.predict_proba(X_te)[:, 1]
pred05 = (proba >= 0.5).astype(int)

print("=== Test metrics @0.5 ===")
print("AUC:     ", roc_auc_score(y_te, proba))
print("PR-AUC:  ", average_precision_score(y_te, proba))
print("F1(0.5): ", f1_score(y_te, pred05))
print("ConfMat:\n", confusion_matrix(y_te, pred05))
print("\nReport:\n", classification_report(y_te, pred05, digits=4))

# 7) F1 최적 임계값도 참고로 계산
prec, rec, thr = precision_recall_curve(y_te, proba)
f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
best_idx = int(f1s.argmax())
best_thr = float(thr[best_idx])
pred_best = (proba >= best_thr).astype(int)

print("\n=== Test metrics @best F1 threshold ===")
print("Best threshold:", best_thr)
print("F1(best):      ", f1_score(y_te, pred_best))
print("ConfMat:\n", confusion_matrix(y_te, pred_best))
