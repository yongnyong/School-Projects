# preprocess_fordb_tabular.py

import numpy as np
import pandas as pd
from pathlib import Path

# 1) 시계열 1개에서 피처 추출 (FordA 때와 동일 구조)
def extract_features(x: np.ndarray, fs: float = 1.0) -> dict:
    x = x.astype(float)
    n = len(x)

    feat = {
        "len": n,
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if n > 1 else 0.0,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p90": float(np.quantile(x, 0.90)),
        "iqr": float(np.quantile(x, 0.75) - np.quantile(x, 0.25)),
    }

    # skew
    if n >= 3:
        m = x.mean()
        s = x.std(ddof=1)
        if s > 0:
            z = (x - m) / s
            feat["skew"] = float(np.mean(z**3))
        else:
            feat["skew"] = 0.0
    else:
        feat["skew"] = 0.0

    # kurtosis
    if n >= 4:
        m = x.mean()
        s = x.std(ddof=1)
        if s > 0:
            z = (x - m) / s
            feat["kurtosis"] = float(np.mean(z**4) - 3.0)
        else:
            feat["kurtosis"] = 0.0
    else:
        feat["kurtosis"] = 0.0

    # zero-crossings
    zc = np.where(np.diff(np.signbit(x)))[0]
    feat["zero_crossings"] = int(len(zc))

    # 1차 차분
    dx = np.diff(x)
    if dx.size == 0:
        dx = np.array([0.0])
    feat["diff_mean"] = float(np.mean(dx))
    feat["diff_std"] = float(np.std(dx, ddof=1)) if dx.size > 1 else 0.0
    feat["diff_abs_mean"] = float(np.mean(np.abs(dx)))

    # ACF 비슷한 값
    def acf_lag(arr: np.ndarray, lag: int) -> float:
        if lag <= 0 or lag >= arr.size:
            return 0.0
        a = arr - arr.mean()
        denom = float((a * a).sum()) + 1e-12
        return float(np.dot(a[:-lag], a[lag:]) / denom)

    for lag in (1, 5, 10, 20, 50):
        feat[f"acf_lag{lag}"] = acf_lag(x, lag)

    # 추세 기울기
    if n >= 2:
        t = np.arange(n)
        slope = np.polyfit(t, x, 1)[0]
    else:
        slope = 0.0
    feat["slope"] = float(slope)

    # FFT 기반 피처
    X = np.fft.rfft(x - x.mean())
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    mag = np.abs(X)
    ps = mag**2
    ps_sum = float(ps.sum()) + 1e-12

    feat["spec_centroid"] = float((freqs * ps).sum() / ps_sum)
    p = ps / ps_sum
    feat["spec_entropy"] = float(-(p * np.log(p + 1e-12)).sum())

    if mag.size > 1:
        idx_peak = int(np.argmax(mag[1:]) + 1)
        feat["dom_freq"] = float(freqs[idx_peak])
        feat["dom_power"] = float(mag[idx_peak])
    else:
        feat["dom_freq"] = 0.0
        feat["dom_power"] = 0.0

    return feat


# 2) 시계열 CSV(0열=label, 나머지=시계열) → tabular CSV
def make_tabular(series_csv_path: str, split_name: str, out_csv_path: str):
    df = pd.read_csv(series_csv_path)

    # 0번 열이 label, 1~끝이 시계열이라고 가정
    y = df.iloc[:, 0].astype(int).to_numpy()
    X = df.iloc[:, 1:].to_numpy()

    feats = [extract_features(row) for row in X]
    tab = pd.DataFrame(feats)
    tab.insert(0, "label", y)
    tab.insert(1, "split", split_name)

    tab.to_csv(out_csv_path, index=False)
    print(f"saved {split_name}:", out_csv_path, "shape:", tab.shape)


if __name__ == "__main__":
    # 네 프로젝트 경로에 맞게 수정
    BASE = Path(r"C:\Users\user\PycharmProjects\산데분")

    # 파일 이름: 질문에서 준 그대로 사용
    train_series_csv = BASE / "FordB.csv"
    test_series_csv  = BASE / "FordB_TEST.csv"

    # FordA와 같은 형식의 tabular 파일 생성
    make_tabular(str(train_series_csv), "train", str(BASE / "fordB_train_tabular.csv"))
    make_tabular(str(test_series_csv),  "test",  str(BASE / "fordB_tabular.csv"))
