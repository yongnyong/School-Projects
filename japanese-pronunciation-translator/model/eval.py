import os
import sys
import torch
import pickle
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ✅ utility 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utility')))
from utils import CharVocab, PronunciationDataset
from seq2seq import Encoder, Attention, AttentionDecoder, Seq2Seq

# ---------- 설정 ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../runs/model_best.pt"
SRC_VOCAB_PATH = "../runs/src_vocab.pkl"
TGT_VOCAB_PATH = "../runs/tgt_vocab.pkl"
TEST_PATH = "../data/test.csv"

# ---------- vocab 로드 ----------
with open(SRC_VOCAB_PATH, "rb") as f:
    src_vocab = pickle.load(f)
with open(TGT_VOCAB_PATH, "rb") as f:
    tgt_vocab = pickle.load(f)

# ---------- 데이터 로드 ----------
test_df = pd.read_csv(TEST_PATH, encoding='utf-8')
test_df.columns = test_df.columns.str.strip()

# ✅ NaN 및 float 방지 처리
test_df.dropna(subset=["input", "target"], inplace=True)
test_df = test_df[test_df['input'].apply(lambda x: isinstance(x, str))]
test_df = test_df[test_df['target'].apply(lambda x: isinstance(x, str))]

test_pairs = list(zip(test_df['input'], test_df['target']))

# ---------- Dataset ----------
test_dataset = PronunciationDataset(test_pairs, src_vocab, tgt_vocab)

def collate_batch(batch):
    src_batch, tgt_batch = zip(*batch)
    src_pad = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=0)
    tgt_pad = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=0)
    return src_pad.to(DEVICE), tgt_pad.to(DEVICE)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)

# ---------- 모델 로드 ----------
enc = Encoder(len(src_vocab), 128, 256)
attn = Attention(256)
dec = AttentionDecoder(len(tgt_vocab), 128, 256, attn)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------- 평가 ----------
smoother = SmoothingFunction()
total_bleu = 0
samples = []

with torch.no_grad():
    for src, tgt in tqdm(test_loader, desc="📊 Evaluating on Test Set"):
        output = model(src, tgt, teacher_forcing_ratio=0.0)
        pred_tokens = output.argmax(-1)
        pred_seq = tgt_vocab.decode(pred_tokens[:, 0].cpu().numpy())
        tgt_seq = tgt_vocab.decode(tgt[:, 0].cpu().numpy())
        input_seq = src_vocab.decode(src[:, 0].cpu().numpy())

        bleu = sentence_bleu([list(tgt_seq)], list(pred_seq), smoothing_function=smoother.method1)
        total_bleu += bleu

        if len(samples) < 5:
            samples.append((input_seq, tgt_seq, pred_seq))

avg_bleu = total_bleu / len(test_loader)

print(f"\n✅ Final Test BLEU: {avg_bleu:.4f}")
print("🔍 예시 샘플:")
for inp, ref, hyp in samples:
    print(f"  🔸 Input : {inp}")
    print(f"  🔹 Target: {ref}")
    print(f"  🔺 Pred  : {hyp}\n")

