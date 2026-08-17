# importance_from_logreg.py
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

BASE = Path(r"C:\Users\user\PycharmProjects\산데분")
train = pd.read_csv(BASE / "fordA_train_tabular.csv")
test  = pd.read_csv(BASE / "fordA_tabular.csv")
test  = test.loc[test["split"]=="test"].copy()

X_tr = train.drop(columns=["label","split"])
y_tr = (train["label"]==1).astype(int)

# 동일 파이프라인 재학습(계수 확인용)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(solver="liblinear", C=1.0, max_iter=200, random_state=42))
])
pipe.fit(X_tr, y_tr)

# 계수 추출: coef_는 class=1에 대한 log-odds 기여
coef = pipe.named_steps["clf"].coef_[0]
features = X_tr.columns

imp = pd.Series(coef, index=features).sort_values(ascending=False)

print("\n=== class=1(이상) 확률을 높이는 Top +10 ===")
print(imp.head(10).round(4))

print("\n=== class=1(이상) 확률을 낮추는 Top -10 ===")
print(imp.tail(10).round(4))

# 해석 도움: Odds Ratio
odds = (imp).apply(lambda x: float(__import__("math").exp(x)))
print("\n=== 상위 +10의 Odds Ratio(배수 효과) ===")
print(odds.head(10).round(3))
