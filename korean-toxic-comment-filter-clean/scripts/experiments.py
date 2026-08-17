# scripts/experiments.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, BertTokenizerFast

from utils.utils import load_config, set_seed, compute_metrics
from src.dataset import TextEmotionDataset
from src.model import TextEmotionClassifier
from src.emotion import EmotionFeatureExtractor


# ---------------------------
# 1. labels 컬럼 생성 (멀티레이블 → 7 클래스)
# ---------------------------
def ensure_labels_column(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if label_col in df.columns:
        print("[INFO] labels 컬럼 이미 존재. 그대로 사용합니다.")
        return df

    print("[INFO] labels 컬럼이 없어 멀티레이블 → 단일 7클래스 라벨로 변환합니다.")

    def row_to_label(row):
        # 2: 여성/가족
        if "여성/가족" in row and row["여성/가족"] == 1:
            return 2
        # 3: 남성
        if "남성" in row and row["남성"] == 1:
            return 3
        # 4: 성소수자
        if "성소수자" in row and row["성소수자"] == 1:
            return 4
        # 5: 인종/국적
        if "인종/국적" in row and row["인종/국적"] == 1:
            return 5
        # 6: 기타 혐오 (연령, 지역, 종교, 기타혐오 중 하나라도)
        if (
            ("연령" in row and row["연령"] == 1)
            or ("지역" in row and row["지역"] == 1)
            or ("종교" in row and row["종교"] == 1)
            or ("기타혐오" in row and row["기타혐오"] == 1)
        ):
            return 6
        # 1: 악플/욕설
        if "악플/욕설" in row and row["악플/욕설"] == 1:
            return 1
        # 0: clean
        if "clean" in row and row["clean"] == 1:
            return 0
        # 그 밖의 애매한 경우도 일단 clean(0)으로 처리
        return 0

    df[label_col] = df.apply(row_to_label, axis=1)
    print("[INFO] labels 컬럼 생성 완료.")
    return df


# ---------------------------
# 2. 감정 feature 생성
# ---------------------------
def add_emotion_features(df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, list[str]]:
    text_col = cfg.data.text_col
    print("[INFO] 감정 feature 생성 시작 (GoEmotions-Korean)...")

    texts = df[text_col].astype(str).tolist()

    emo_extractor = EmotionFeatureExtractor(cfg.emotion.model_name)
    emo_feats, emo_labels = emo_extractor.encode(
        texts,
        batch_size=cfg.emotion.batch_size,
        max_length=cfg.data.max_length,
    )

    print(f"[INFO] 감정 feature shape: {emo_feats.shape}")  # (N, num_emotions)

    emotion_cols = []
    for j, lbl in enumerate(emo_labels):
        col_name = f"{cfg.data.emotion_prefix}{lbl}"
        df[col_name] = emo_feats[:, j]
        emotion_cols.append(col_name)

    print(f"[INFO] emotion_cols 개수: {len(emotion_cols)}")
    return df, emotion_cols


# ---------------------------
# 3. 공통 데이터 준비 (1번만!)
# ---------------------------
def prepare_data(cfg):
    df = pd.read_csv(cfg.data.input_csv, encoding="utf-8-sig")
    text_col = cfg.data.text_col
    label_col = cfg.data.label_col

    df = ensure_labels_column(df, label_col)

    print(f"[INFO] 전체 샘플 수: {len(df)}")
    print("[INFO] 라벨 분포:")
    print(df[label_col].value_counts())

    # 감정 feature 추가
    df, emotion_cols = add_emotion_features(df, cfg)

    # 8:1:1 split (stratify)
    print("[INFO] 8:1:1 train/valid/test split...")
    train_df, temp_df = train_test_split(
        df,
        test_size=cfg.data.test_size,  # 0.2
        stratify=df[label_col],
        random_state=cfg.data.random_state,
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,  # 0.1 / 0.1
        stratify=temp_df[label_col],
        random_state=cfg.data.random_state,
    )

    print(f"[INFO] Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    return train_df, valid_df, test_df, emotion_cols


# ---------------------------
# 4. 모델별/하이퍼파라미터별 실험 설정
# ---------------------------

MODEL_EXPERIMENTS = {
    # "bert": {
    #     "model_name": "klue/bert-base",
    #     "lrs": [2e-5, 3e-5, 5e-5],
    #     "batch_sizes": [16],
    #     "epochs": [3, 4],
    # },
    # "roberta": {
    #     "model_name": "klue/roberta-base",
    #     "lrs": [1e-5, 2e-5, 3e-5],
    #     "batch_sizes": [16,32],
    #     "epochs": [3, 4],
    # },
    "albert": {
        "model_name": "kykim/albert-kor-base",
        "lrs": [1e-5, 2e-5, 3e-5],
        "batch_sizes": [16],
        "epochs": [4, 6],
    },
    "electra": {
        "model_name": "monologg/koelectra-base-v3-discriminator",
        "lrs": [1e-5, 2e-5, 3e-5],
        "batch_sizes": [16],
        "epochs": [2, 3],
    },
}


# ---------------------------
# 5. 메인 experiments 루프
# ---------------------------
def main(cfg_path: str):
    cfg = load_config(cfg_path)
    set_seed(cfg.train.seed)

    # 공통 데이터/감정 feature/스플릿 한 번만 만들기
    train_df, valid_df, test_df, emotion_cols = prepare_data(cfg)
    text_col = cfg.data.text_col
    label_col = cfg.data.label_col

    # 각 모델별로 반복
    for model_key, mconf in MODEL_EXPERIMENTS.items():
        base_model_name = mconf["model_name"]
        lrs = mconf["lrs"]
        batch_sizes = mconf["batch_sizes"]
        epochs_list = mconf["epochs"]

        print(f"\n==============================")
        print(f"[EXPERIMENT] Base model: {model_key} ({base_model_name})")
        print(f"==============================")

        for lr in lrs:
            for bs in batch_sizes:
                for n_epochs in epochs_list:
                    run_name = f"{model_key}_lr{lr}_bs{bs}_ep{n_epochs}"
                    print(f"\n[RUN] {run_name}")

                    output_dir = PROJECT_ROOT / "model" / run_name
                    logging_dir = PROJECT_ROOT / "log" / run_name
                    os.makedirs(output_dir, exist_ok=True)
                    os.makedirs(logging_dir, exist_ok=True)

                    # Tokenizer & Dataset
                    ALBERT_BERT_TOKENIZER_MODELS = {
                        "kykim/albert-kor-base",
                        # 나중에 비슷한 구조 모델 더 생기면 여기에 추가
                    }

                    if base_model_name in ALBERT_BERT_TOKENIZER_MODELS:
                        tokenizer = BertTokenizerFast.from_pretrained(base_model_name)
                    else:
                        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

                    # tokenizer = AutoTokenizer.from_pretrained(base_model_name)

                    train_dataset = TextEmotionDataset(
                        train_df,
                        tokenizer=tokenizer,
                        text_col=text_col,
                        label_col=label_col,
                        emotion_cols=emotion_cols,
                        max_len=cfg.data.max_length,
                    )

                    valid_dataset = TextEmotionDataset(
                        valid_df,
                        tokenizer=tokenizer,
                        text_col=text_col,
                        label_col=label_col,
                        emotion_cols=emotion_cols,
                        max_len=cfg.data.max_length,
                    )

                    test_dataset = TextEmotionDataset(
                        test_df,
                        tokenizer=tokenizer,
                        text_col=text_col,
                        label_col=label_col,
                        emotion_cols=emotion_cols,
                        max_len=cfg.data.max_length,
                    )

                    # Model
                    model = TextEmotionClassifier(
                        base_model_name=base_model_name,
                        num_labels=cfg.model.num_labels,
                        emotion_dim=len(emotion_cols),
                        emotion_hidden_dim=cfg.model.emotion_hidden_dim,
                    )

                    # TrainingArguments
                    training_args = TrainingArguments(
                        output_dir=str(output_dir),
                        num_train_epochs=n_epochs,
                        per_device_train_batch_size=bs,
                        per_device_eval_batch_size=bs * 2,
                        learning_rate=lr,
                        weight_decay=cfg.train.weight_decay,
                        warmup_ratio=cfg.train.warmup_ratio,
                        evaluation_strategy=cfg.train.eval_strategy,
                        save_strategy=cfg.train.save_strategy,
                        metric_for_best_model=cfg.train.metric_for_best_model,
                        load_best_model_at_end=True,
                        logging_dir=str(logging_dir),
                        logging_strategy="steps",
                        logging_steps=cfg.train.logging_steps,
                        save_total_limit=2,
                        report_to=["tensorboard"],
                        run_name=run_name,
                        log_level="error",
                        log_level_replica="error",
                        save_safetensors=False,
                    )

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=valid_dataset,
                        compute_metrics=compute_metrics,
                    )

                    # Train
                    trainer.train()

                    # Save best model & tokenizer
                    trainer.save_model(str(output_dir))
                    tokenizer.save_pretrained(str(output_dir))

                    # Save training log history
                    log_history = trainer.state.log_history
                    logs_df = pd.DataFrame(log_history)
                    logs_csv_path = PROJECT_ROOT / "log" / f"{run_name}_history.csv"
                    logs_df.to_csv(logs_csv_path, index=False, encoding="utf-8-sig")
                    print(f"[INFO] Saved log history → {logs_csv_path}")

                    # Evaluate on test set
                    test_result = trainer.evaluate(test_dataset)
                    print("[INFO] Test result:", test_result)

                    test_metrics_path = PROJECT_ROOT / "log" / f"{run_name}_test_metrics.csv"
                    pd.DataFrame([test_result]).to_csv(
                        test_metrics_path, index=False, encoding="utf-8-sig"
                    )
                    print(f"[INFO] Saved test metrics → {test_metrics_path}")


if __name__ == "__main__":
    default_cfg = PROJECT_ROOT / "config" / "text_emo_7cls.yaml"
    main(str(default_cfg))
