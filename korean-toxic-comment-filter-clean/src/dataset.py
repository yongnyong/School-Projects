# src/dataset.py
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class TextEmotionDataset(Dataset):
    """
    텍스트 + 감정 feature 를 함께 사용하는 Dataset.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        text_col: str,
        label_col: Optional[str],
        emotion_cols: List[str],
        max_len: int = 128,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.emotion_cols = emotion_cols
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = str(row[self.text_col])

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}

        # emotion features
        emo = row[self.emotion_cols].values.astype("float32")
        item["emotion_feats"] = torch.tensor(emo, dtype=torch.float32)

        if self.label_col is not None:
            item["labels"] = int(row[self.label_col])

        return item
