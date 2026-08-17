import torch
import pickle
import sys
import os

# ✅ utility 경로 등록
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utility')))
from utils import CharVocab
from seq2seq import Encoder, Attention, AttentionDecoder, Seq2Seq

# ---------- 설정 ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../runs/model_best.pt"
SRC_VOCAB_PATH = "../runs/src_vocab.pkl"
TGT_VOCAB_PATH = "../runs/tgt_vocab.pkl"

# ---------- vocab 로드 ----------
with open(SRC_VOCAB_PATH, "rb") as f:
    src_vocab = pickle.load(f)
with open(TGT_VOCAB_PATH, "rb") as f:
    tgt_vocab = pickle.load(f)

# ---------- 모델 정의 & 로드 ----------
enc = Encoder(len(src_vocab), 128, 256)
attn = Attention(256)
dec = AttentionDecoder(len(tgt_vocab), 128, 256, attn)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------- 추론 함수 ----------
def infer(input_text):
    tokens = [src_vocab.stoi["<sos>"]] + [src_vocab.stoi.get(ch, 0) for ch in input_text] + [src_vocab.stoi["<eos>"]]
    src_tensor = torch.tensor(tokens).unsqueeze(1).to(DEVICE)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        input_token = torch.tensor([tgt_vocab.stoi["<sos>"]], device=DEVICE)
        output_seq = []

        for _ in range(100):
            output, hidden = model.decoder(input_token, hidden, encoder_outputs)
            top1 = output.argmax(1)
            if top1.item() == tgt_vocab.stoi["<eos>"]:
                break
            output_seq.append(top1.item())
            input_token = top1

    return tgt_vocab.decode(output_seq)

# ---------- 실행 ----------
if __name__ == "__main__":
    print("🟢 발음을 입력하면 의미를 예측합니다. (종료: quit)")
    while True:
        input_text = input("🎤 발음을 입력하세요: ").strip()
        if input_text.lower() == "quit":
            break
        result = infer(input_text)
        print(f"📝 예측 결과: {result}")
