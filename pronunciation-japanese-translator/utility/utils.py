from collections import Counter
import torch
from torch.utils.data import Dataset  # ✅ 이 줄 추가

class CharVocab:
    def __init__(self, texts, specials=["<pad>", "<sos>", "<eos>"]):
        counter = Counter(ch for text in texts if isinstance(text, str) for ch in text)
        self.itos = specials + sorted(counter)
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def encode(self, text):
        if not isinstance(text, str):
            text = ""
        return [self.stoi["<sos>"]] + [self.stoi.get(c, 0) for c in text] + [self.stoi["<eos>"]]

    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids if 0 <= i < len(self.itos) and self.itos[i] not in ("<sos>", "<eos>", "<pad>")])

    def __len__(self):
        return len(self.itos)


class PronunciationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        return torch.tensor(self.src_vocab.encode(src)), torch.tensor(self.tgt_vocab.encode(tgt))
