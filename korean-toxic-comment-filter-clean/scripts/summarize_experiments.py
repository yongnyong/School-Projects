# scripts/summarize_experiments.py

import sys
import re
from pathlib import Path

import pandas as pd

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LOG_DIR = PROJECT_ROOT / "log"


def parse_run_name(run_name: str):
    """
    run_name 예시: 'bert_lr2e-05_bs16_ep3'
    -> model='bert', lr=2e-05, batch_size=16, epochs=3
    """
    pattern = r"^(?P<model>[^_]+)_lr(?P<lr>[^_]+)_bs(?P<bs>\d+)_ep(?P<ep>\d+)$"
    m = re.match(pattern, run_name)
    if not m:
        # 패턴이 안 맞으면 최소한 run_name만 리턴
        return {
            "run_name": run_name,
            "model": None,
            "lr": None,
            "batch_size": None,
            "epochs": None,
        }

    d = m.groupdict()
    # 문자열을 적절히 캐스팅
    lr_str = d["lr"]
    try:
        lr = float(lr_str)
    except ValueError:
        lr = lr_str  # 캐스팅 실패하면 원 문자열 그대로 둠

    return {
        "run_name": run_name,
        "model": d["model"],
        "lr": lr,
        "batch_size": int(d["bs"]),
        "epochs": int(d["ep"]),
    }


def load_test_metrics():
    """
    log/ 폴더에서 *_test_metrics.csv 파일들을 모두 읽어
    하나의 DataFrame으로 합친다.
    """
    rows = []

    if not LOG_DIR.exists():
        print(f"[ERROR] log 디렉토리가 없습니다: {LOG_DIR}")
        return pd.DataFrame()

    files = sorted(LOG_DIR.glob("*_test_metrics.csv"))
    if not files:
        print(f"[WARN] *_test_metrics.csv 파일이 log 디렉토리에 없습니다: {LOG_DIR}")
        return pd.DataFrame()

    print(f"[INFO] 발견된 test_metrics 파일 수: {len(files)}")

    for path in files:
        run_name = path.stem.replace("_test_metrics", "")  # 파일명에서 run_name 추출
        meta = parse_run_name(run_name)

        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path)  # fallback

        if df.empty:
            print(f"[WARN] 빈 metrics 파일: {path}")
            continue

        # 보통 1행짜리일 텐데, 혹시 여러 행이면 첫 행만 사용
        metrics = df.iloc[0].to_dict()

        row = {
            **meta,
            **metrics,  # eval_loss, eval_accuracy, eval_macro_f1, eval_lrap 등
        }
        rows.append(row)

    if not rows:
        print("[WARN] 유효한 metrics를 가진 파일이 없습니다.")
        return pd.DataFrame()

    summary_df = pd.DataFrame(rows)
    return summary_df


def main():
    summary_df = load_test_metrics()
    if summary_df.empty:
        print("[INFO] 요약할 데이터가 없습니다.")
        return

    # 존재하는 컬럼 중 정렬 기준 설정
    sort_col = None
    for cand in ["eval_macro_f1", "macro_f1", "eval_lrap"]:
        if cand in summary_df.columns:
            sort_col = cand
            break

    if sort_col:
        summary_df = summary_df.sort_values(sort_col, ascending=False)
        print(f"[INFO] '{sort_col}' 기준으로 내림차순 정렬했습니다.")
    else:
        print("[WARN] macro_f1 / eval_macro_f1 / eval_lrap 컬럼이 없어 정렬하지 않습니다.")

    # summary CSV 저장
    out_path = LOG_DIR / "experiments_summary.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 요약 결과를 저장했습니다 → {out_path}")

    # 콘솔에 상위 몇 개만 출력
    print("\n===== 상위 10개 요약 =====")
    with pd.option_context("display.max_columns", None):
        print(summary_df.head(10))


if __name__ == "__main__":
    main()
