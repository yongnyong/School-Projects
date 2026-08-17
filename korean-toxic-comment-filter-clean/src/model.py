# src/model.py
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEmotionClassifier(nn.Module):
    """
    텍스트 인코더(예: klue/roberta-base) + 감정 feature MLP → concat → 7-class classifier
    """

    def __init__(
        self,
        base_model_name: str,
        num_labels: int,
        emotion_dim: int,
        emotion_hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.base_model.config.hidden_size

        self.emotion_mlp = nn.Sequential(
            nn.Linear(emotion_dim, emotion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + emotion_hidden_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        emotion_feats=None,
        labels=None,
        **kwargs,
    ):
        # Trainer가 추가로 넣는 키 중 base_model이 모르는 것들 제거
        kwargs.pop("num_items_in_batch", None)
        kwargs.pop("labels", None)  # labels는 우리가 직접 쓰니까 base_model에 넘기지 않도록

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # pooler_output이 있으면 사용, 없으면 [CLS]
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            text_repr = outputs.pooler_output
        else:
            text_repr = outputs.last_hidden_state[:, 0, :]

        emo_repr = self.emotion_mlp(emotion_feats)
        concat = torch.cat([text_repr, emo_repr], dim=-1)
        logits = self.classifier(concat)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
        }

