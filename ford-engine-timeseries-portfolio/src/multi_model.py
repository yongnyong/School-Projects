# multi_models_fordA.py
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    classification_report, confusion_matrix, precision_recall_curve
)

# ===== 0) 데이터 로드 =====
BASE = Path(r"C:\Users\user\PycharmProjects\산데분")
train = pd.read_csv(BASE / "fordB_train_tabular.csv")
test  = pd.read_csv(BASE / "fordB_tabular.csv")
test  = test.loc[test["split"]=="test"].copy()

X_tr = train.drop(columns=["label","split"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)
y_tr = (train["label"]==1).astype(int)
X_te = test.drop(columns=["label","split"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)
y_te = (test["label"]==1).astype(int)

# ===== 1) 모델 정의 =====
models = {}

# 1-1) RandomForest (트리기반, 스케일 불필요)
from sklearn.ensemble import RandomForestClassifier
models["rf"] = Pipeline([
    ("clf", RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=42
    ))
])

# 1-2) GradientBoosting (순수 sklearn)
from sklearn.ensemble import GradientBoostingClassifier
models["gbdt"] = Pipeline([
    ("clf", GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
    ))
])

# 1-3) XGBoost (설치 시 사용)
try:
    from xgboost import XGBClassifier
    models["xgb"] = Pipeline([
        ("clf", XGBClassifier(
            n_estimators=500, learning_rate=0.05,
            max_depth=5, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=42, n_jobs=-1, tree_method="hist"
        ))
    ])
except Exception as e:
    print("[skip] xgboost unavailable:", e)

# 1-4) LightGBM (설치 시 사용)
try:
    from lightgbm import LGBMClassifier
    models["lgbm"] = Pipeline([
        ("clf", LGBMClassifier(
            n_estimators=500, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42
        ))
    ])
except Exception as e:
    print("[skip] lightgbm unavailable:", e)

# 1-5) SVM(RBF) 확률보정(속도 느리면 C/γ 줄이기)
from sklearn.svm import SVC
models["svm_rbf"] = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(C=2.0, gamma="scale", probability=True, random_state=42))
])

# 1-6) 선형 SVM + 확률보정(빠름)
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
models["linear_svm"] = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42), cv=5))
])

# 1-7) 로지스틱(참고용 baseline)
from sklearn.linear_model import LogisticRegression
models["logreg"] = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(solver="liblinear", C=1.0, max_iter=200, random_state=42))
])

# ===== 2) 평가 함수 =====
def eval_model(name, pipe, Xtr, ytr, Xte, yte):
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:,1]
    pred05 = (proba >= 0.5).astype(int)

    auc = roc_auc_score(yte, proba)
    pr  = average_precision_score(yte, proba)
    f1_ = f1_score(yte, pred05)

    # best-F1 threshold
    prec, rec, thr = precision_recall_curve(yte, proba)
    f1s = 2*prec[:-1]*rec[:-1] / (prec[:-1]+rec[:-1] + 1e-12)
    best_idx = int(f1s.argmax())
    best_thr = float(thr[best_idx])
    pred_best = (proba >= best_thr).astype(int)
    f1_best = f1_score(yte, pred_best)

    print(f"\n=== [{name}] Test ===")
    print(f"AUC={auc:.4f}  PR-AUC={pr:.4f}  F1@0.5={f1_:.4f}  F1@best={f1_best:.4f}  thr*={best_thr:.3f}")
    print("ConfMat@best:\n", confusion_matrix(yte, pred_best))
    return {
        "model": name, "AUC": auc, "PR-AUC": pr,
        "F1@0.5": f1_, "F1@best": f1_best, "thr*": best_thr
    }

# ===== 3) 실행 =====
summary = []
for name, pipe in models.items():
    res = eval_model(name, pipe, X_tr, y_tr, X_te, y_te)
    summary.append(res)

print("\n=== Summary ===")
print(pd.DataFrame(summary).sort_values("AUC", ascending=False))

# ===== 4) 중요도 보기(가능한 모델만) =====
def show_importance(name, pipe, Xcols):
    try:
        clf = pipe.named_steps.get("clf", pipe)
        # 트리계열
        if hasattr(clf, "feature_importances_"):
            imp = pd.Series(clf.feature_importances_, index=Xcols).sort_values(ascending=False)
            print(f"\nTop-15 importances [{name}]")
            print(imp.head(15).round(4))
        # 선형계열
        elif hasattr(clf, "coef_"):
            coef = clf.coef_[0]
            imp = pd.Series(coef, index=Xcols).sort_values(ascending=False)
            print(f"\nTop+10 coef [{name}]")
            print(imp.head(10).round(4))
            print(f"\nTop-10 coef [{name}]")
            print(imp.tail(10).round(4))
    except Exception as e:
        print(f"[warn] importance for {name}: {e}")

for name, pipe in models.items():
    # 재학습 없이 중요도만 필요하면 pass. 여기선 test 성능 본 모델 그대로 중요도 출력 위해 다시 fit.
    pipe.fit(X_tr, y_tr)
    show_importance(name, pipe, X_tr.columns)
