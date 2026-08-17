# scripts/train.py

import os
import sys
from pathlib import Path

# --- src 폴더 import 가능하게 경로 추가 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TrainingArguments, Trainer

from utils.utils import load_config, set_seed, compute_metrics
from src.dataset import TextEmotionDataset
from src.model import TextEmotionClassifier
from src.emotion import EmotionFeatureExtractor


def main(cfg_path: str):
    # 1. Config & Seed
    cfg = load_config(cfg_path)
    set_seed(cfg.train.seed)

    # 2. 데이터 로드
    df = pd.read_csv(cfg.data.input_csv, encoding="utf-8-sig")
    text_col = cfg.data.text_col
    label_col = cfg.data.label_col

    # 만약 labels 컬럼이 없다면 멀티레이블 → 단일 라벨(0~6)로 변환
    if label_col not in df.columns:
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
            # 6: 기타 혐오 (연령, 지역, 종교, 기타혐오)
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

    print(f"[INFO] 전체 샘플 수: {len(df)}")
    print("[INFO] 라벨 분포:")
    print(df[label_col].value_counts())

    # 3. 감정 feature 생성 (GoEmotions-Korean)
    print("[INFO] 감정 feature 생성 시작...")
    texts = df[text_col].astype(str).tolist()

    emo_extractor = EmotionFeatureExtractor(cfg.emotion.model_name)
    emo_feats, emo_labels = emo_extractor.encode(
        texts,
        batch_size=cfg.emotion.batch_size,
        max_length=cfg.data.max_length,
    )

    print(f"[INFO] 감정 feature shape: {emo_feats.shape}")  # (N, num_emotions)

    # 감정 컬럼 이름 생성: emo_<label>
    emotion_cols = []
    for j, lbl in enumerate(emo_labels):
        col_name = f"{cfg.data.emotion_prefix}{lbl}"
        df[col_name] = emo_feats[:, j]
        emotion_cols.append(col_name)

    print(f"[INFO] emotion_cols 개수: {len(emotion_cols)}")

    # 4. 8:1:1 split (stratify)
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

    # 5. Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model_name)

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

    # 6. Model
    model = TextEmotionClassifier(
        base_model_name=cfg.model.base_model_name,
        num_labels=cfg.model.num_labels,
        emotion_dim=len(emotion_cols),
        emotion_hidden_dim=cfg.model.emotion_hidden_dim,
    )

    # 7. TrainingArguments
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=cfg.train.output_dir,
        num_train_epochs=cfg.train.num_train_epochs,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,

        eval_strategy=cfg.train.eval_strategy,     # (또는 evaluation_strategy, 버전에 맞게)
        save_strategy=cfg.train.save_strategy,
        metric_for_best_model=cfg.train.metric_for_best_model,
        load_best_model_at_end=True,

        logging_dir=cfg.train.logging_dir,
        logging_strategy="steps",
        logging_steps=cfg.train.logging_steps,
        save_total_limit=2,

        report_to=["tensorboard"],   # 로깅을 TensorBoard로 보내기
    )


    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    # 9. 학습
    trainer.train()

    # 10. Best model 저장
    trainer.save_model(cfg.train.output_dir)
    tokenizer.save_pretrained(cfg.train.output_dir)

    # 11. 학습 로그 CSV로 저장
    log_history = trainer.state.log_history
    logs_df = pd.DataFrame(log_history)
    logs_path = PROJECT_ROOT / "log" / "text_emo_7cls_training_logs.csv"
    logs_df.to_csv(logs_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 학습 로그 저장: {logs_path}")

    # 12. Test set 평가
    test_result = trainer.evaluate(test_dataset)
    print("[INFO] Test result:", test_result)

    test_result_path = PROJECT_ROOT / "log" / "text_emo_7cls_test_metrics.csv"
    pd.DataFrame([test_result]).to_csv(
        test_result_path, index=False, encoding="utf-8-sig"
    )
    print(f"[INFO] Test metrics 저장: {test_result_path}")


if __name__ == "__main__":
    default_cfg = PROJECT_ROOT / "config" / "text_emo_7cls.yaml"
    main(str(default_cfg))
