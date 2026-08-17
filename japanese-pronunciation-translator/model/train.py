import os
import sys
import torch
import pickle
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# ✅ utility 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utility')))
from utils import CharVocab, PronunciationDataset
from seq2seq import Encoder, Attention, AttentionDecoder, Seq2Seq

# ---------- 설정 ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EMB_DIM = 128
HID_DIM = 256
EPOCHS = 100
LEARNING_RATE = 0.001
TEACHER_FORCING_RATIO = 0.6
PATIENCE = 15
MODEL_SAVE_PATH = "../runs/model_best.pt"
TENSORBOARD_LOGDIR = "../runs/seq2seq_train"

writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)
smoother = SmoothingFunction()

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

# ---------- 데이터 로드 ----------
train_df = pd.read_csv('../data/train.csv', encoding='utf-8')
val_df = pd.read_csv('../data/val.csv', encoding='utf-8')
train_df.columns = train_df.columns.str.strip()
val_df.columns = val_df.columns.str.strip()

train_df.dropna(subset=["input", "target"], inplace=True)
val_df.dropna(subset=["input", "target"], inplace=True)
train_df = train_df[train_df['input'].apply(lambda x: isinstance(x, str))]
val_df = val_df[val_df['target'].apply(lambda x: isinstance(x, str))]

train_pairs = list(zip(train_df['input'], train_df['target']))
val_pairs = list(zip(val_df['input'], val_df['target']))

# ---------- Vocab ----------
src_vocab = CharVocab([p[0] for p in train_pairs])
tgt_vocab = CharVocab([p[1] for p in train_pairs])

os.makedirs("../runs", exist_ok=True)
with open("../runs/src_vocab.pkl", "wb") as f:
    pickle.dump(src_vocab, f)
with open("../runs/tgt_vocab.pkl", "wb") as f:
    pickle.dump(tgt_vocab, f)
print("✅ Vocab 저장 완료: ../runs/src_vocab.pkl, tgt_vocab.pkl")

# ---------- Dataset ----------
train_dataset = PronunciationDataset(train_pairs, src_vocab, tgt_vocab)
val_dataset = PronunciationDataset(val_pairs, src_vocab, tgt_vocab)

def collate_batch(batch):
    src_batch, tgt_batch = zip(*batch)
    src_pad = nn.utils.rnn.pad_sequence(src_batch, padding_value=0)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=0)
    return src_pad.to(DEVICE), tgt_pad.to(DEVICE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)

# ---------- 모델 ----------
enc = Encoder(len(src_vocab), EMB_DIM, HID_DIM)
attn = Attention(HID_DIM)
dec = AttentionDecoder(len(tgt_vocab), EMB_DIM, HID_DIM, attn)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=0)
early_stopper = EarlyStopping(patience=PATIENCE)

# ---------- 학습 루프 ----------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    print(f"\n📚 Epoch {epoch+1}/{EPOCHS}")

    for src, tgt in tqdm(train_loader, desc="🔧 Training"):
        optimizer.zero_grad()
        output = model(src, tgt, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        tgt = tgt[1:].reshape(-1)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    writer.add_scalar("Loss/train", avg_loss, epoch + 1)

    # ---------- 검증 ----------
    model.eval()
    total_bleu = 0
    samples = []

    with torch.no_grad():
        for src, tgt in tqdm(val_loader, desc="🧪 Validation"):
            output = model(src, tgt, teacher_forcing_ratio=0.0)
            pred_tokens = output.argmax(-1)
            pred_seq = tgt_vocab.decode(pred_tokens[:, 0].cpu().numpy())
            tgt_seq = tgt_vocab.decode(tgt[:, 0].cpu().numpy())
            input_seq = src_vocab.decode(src[:, 0].cpu().numpy())

            bleu = sentence_bleu([list(tgt_seq)], list(pred_seq), smoothing_function=smoother.method1)
            total_bleu += bleu

            if len(samples) < 3:
                samples.append((input_seq, tgt_seq, pred_seq))

    avg_bleu = total_bleu / len(val_loader)
    writer.add_scalar("BLEU/val", avg_bleu, epoch + 1)

    print(f"📊 [Epoch {epoch+1}] Train Loss: {avg_loss:.4f} | Val BLEU: {avg_bleu:.4f}")
    print("🔍 예시 샘플:")
    for inp, ref, hyp in samples:
        print(f"  🔸 Input : {inp}")
        print(f"  🔹 Target: {ref}")
        print(f"  🔺 Pred  : {hyp}\n")

    if early_stopper(avg_bleu):
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"💾 Model saved to {MODEL_SAVE_PATH}")
    if early_stopper.early_stop:
        print(f"⏹️ Early stopping at epoch {epoch+1}. Best BLEU: {early_stopper.best_score:.4f}")
        break

writer.close()

