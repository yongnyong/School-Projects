from pathlib import Path
import pandas as pd

base = Path(r"C:\Users\user\PycharmProjects\산데분")

train = pd.read_csv(base / "fordA_train_tabular.csv")
test  = pd.read_csv(base / "fordA_tabular.csv")

# test에서 train 섞임 제거
test = test.loc[test["split"] == "test"].copy()

X_tr = train.drop(columns=["label", "split"])
y_tr = (train["label"] == 1).astype(int)

X_te = test.drop(columns=["label", "split"])
y_te = (test["label"] == 1).astype(int)

print(train.shape, test.shape)
print(train["label"].value_counts(), test["label"].value_counts())

