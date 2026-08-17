# preprocess_forda.py
import os, io, argparse, numpy as np, pandas as pd

def read_txt(p):
    raw = open(p, "r", encoding="utf-8", errors="ignore").read().strip().replace(",", " ")
    return pd.read_csv(io.StringIO(raw), header=None, sep=r"\s+")

def read_arff(p):
    t = open(p, "r", encoding="utf-8", errors="ignore").read().splitlines()
    i = next(i for i,s in enumerate(t) if s.strip().lower()=="@data")+1
    data = [s.strip() for s in t[i:] if s.strip() and not s.strip().startswith("%")]
    df = pd.read_csv(io.StringIO("\n".join(data)), header=None)
    # 마지막 열이 label → 첫 열로 위치 변경
    cols = list(range(df.shape[1])); df = df[[cols[-1]] + cols[:-1]]
    return df

def load_any(train_p, test_p):
    def _load(p):
        e = os.path.splitext(p)[1].lower()
        return read_txt(p) if e==".txt" else read_arff(p)
    return _load(train_p), _load(test_p)

def np_skew(x):
    n=len(x);
    if n<3: return 0.0
    m=x.mean(); s=x.std(ddof=1);
    return 0.0 if s==0 else float(np.mean(((x-m)/s)**3))
def np_kurt(x):
    n=len(x);
    if n<4: return 0.0
    m=x.mean(); s=x.std(ddof=1);
    return 0.0 if s==0 else float(np.mean(((x-m)/s)**4)-3.0)

def extract_features(x, fs=1.0):
    x=x.astype(float); n=len(x)
    feat=dict(
        len=n, mean=float(np.mean(x)),
        std=float(np.std(x,ddof=1)) if n>1 else 0.0,
        min=float(np.min(x)), max=float(np.max(x)),
        median=float(np.median(x)),
        p10=float(np.quantile(x,0.10)),
        p90=float(np.quantile(x,0.90)),
        iqr=float(np.quantile(x,0.75)-np.quantile(x,0.25)),
        skew=np_skew(x), kurtosis=np_kurt(x),
    )
    zc=np.where(np.diff(np.signbit(x)))[0]; feat["zero_crossings"]=int(len(zc))
    dx=np.diff(x); dx = dx if dx.size else np.array([0.0])
    feat.update(dict(
        diff_mean=float(np.mean(dx)),
        diff_std=float(np.std(dx,ddof=1)) if dx.size>1 else 0.0,
        diff_abs_mean=float(np.mean(np.abs(dx))),
    ))
    def acf(arr,lag):
        if lag<=0 or lag>=arr.size: return 0.0
        a=arr-arr.mean(); d=float((a*a).sum())+1e-12
        return float(np.dot(a[:-lag],a[lag:])/d)
    for lag in (1,5,10,20,50): feat[f"acf_lag{lag}"]=acf(x,lag)
    t=np.arange(n); slope=float(np.polyfit(t,x,1)[0]) if n>=2 else 0.0; feat["slope"]=slope
    X=np.fft.rfft(x-x.mean()); f=np.fft.rfftfreq(n,d=1.0/fs); mag=np.abs(X); ps=mag**2
    pss=float(ps.sum())+1e-12; centroid=float((f*ps).sum()/pss); p=ps/pss
    entropy=float(-(p*np.log(p+1e-12)).sum())
    if mag.size>1: idx=int(np.argmax(mag[1:])+1); dom_freq=float(f[idx]); dom_power=float(mag[idx])
    else: dom_freq=dom_power=0.0
    feat.update(dict(spec_centroid=centroid, spec_entropy=entropy, dom_freq=dom_freq, dom_power=dom_power))
    return feat

def to_series_csv(df,outp):
    T=df.shape[1]-1; cols=["label"]+[f"s{j+1}" for j in range(T)]
    out=df.copy(); out.columns=cols; out.to_csv(outp,index=False); return outp

def to_tabular_csv(df,split_name,outp):
    y=df.iloc[:,0].astype(int).to_numpy(); X=df.iloc[:,1:].to_numpy()
    feats=[extract_features(row) for row in X]
    tab=pd.DataFrame(feats); tab.insert(0,"label",y); tab.insert(1,"split",split_name)
    tab.to_csv(outp,index=False); return outp

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test",  required=True)
    ap.add_argument("--outdir", default=".")
    a=ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    tr,te=load_any(a.train,a.test)

    tr_series=os.path.join(a.outdir,"fordA_train_series.csv")
    te_series=os.path.join(a.outdir,"fordA_test_series.csv")
    to_series_csv(tr,tr_series); to_series_csv(te,te_series)

    tr_tab=os.path.join(a.outdir,"fordA_train_tabular.csv")
    te_tab=os.path.join(a.outdir,"fordA_tabular.csv")
    to_tabular_csv(tr,"train",tr_tab); to_tabular_csv(te,"test",te_tab)

    print("[done]"); print(tr_series); print(te_series); print(tr_tab); print(te_tab)

if __name__=="__main__": main()
