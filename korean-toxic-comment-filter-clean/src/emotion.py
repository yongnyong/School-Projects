# src/emotion.py
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class EmotionFeatureExtractor:
    """
    monologg/koelectra-base-v3-goemotions 를 사용하여
    문장 리스트에 대해 [N, num_emotions] 형태의 감정 score 벡터를 생성하는 클래스.
    """

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model.to(self.device)
        self.model.eval()

        # id2label에서 감정 레이블 이름 가져오기
        id2label = self.model.config.id2label
        # id 순서대로 정렬 보장
        self.emotion_labels = [id2label[i] for i in range(len(id2label))]

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 128,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        texts: 문장 리스트
        return:
          - feats: [N, num_emotions] 감정 score (sigmoid 확률)
          - emotion_labels: 감정 이름 리스트 (열 순서)
        """
        all_feats = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                enc = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**enc)
                logits = outputs.logits  # [B, num_emotions]

                # multi-label이라 sigmoid 사용
                probs = torch.sigmoid(logits).cpu().numpy().astype("float32")
                all_feats.append(probs)

        feats = np.concatenate(all_feats, axis=0)
        return feats, self.emotion_labels
