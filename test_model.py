import torch.nn.functional as F
import torch.nn as nn
import torch
import re
# test_model.py
# Dự đoán cảm xúc cho câu mới bằng model LSTM đã huấn luyện.


MODEL_FILE = "model.pt"
LABELS = ["negative", "neutral", "positive"]
MAX_LEN = 100  # độ dài tối đa của câu

# ----------------------------
# Tách từ (tokenizer)
# ----------------------------


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t] or ["<unk>"]


# ----------------------------
# Định nghĩa lại mô hình LSTM
# ----------------------------
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        e = self.emb(x)                     # (B, L, D)
        _, (h, _) = self.lstm(e)            # h: (1, B, H)
        out = self.fc(h.squeeze(0))         # (B, num_classes)
        return out

    def predict(self, text, vocab, max_len=MAX_LEN):
        tokens = tokenize(text)
        ids = [vocab.get(t, 1) for t in tokens]  # 1 = <unk>
        ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
        t = torch.tensor([ids], dtype=torch.long)
        self.eval()
        with torch.no_grad():
            logits = self.forward(t)
            probs = F.softmax(logits, dim=-1)[0].numpy()
        return probs


# ----------------------------
# Tải mô hình đã huấn luyện
# ----------------------------
saved = torch.load(MODEL_FILE, map_location="cpu")
vocab = saved["vocab"]
cfg = saved["config"]
state = saved["state_dict"]

model = LSTMModel(
    vocab_size=len(vocab),
    embed_dim=cfg["embed_dim"],
    hidden_dim=cfg["hidden_dim"],
    num_classes=3
)
model.load_state_dict(state)
model.eval()

print("✅ Mô hình LSTM đã tải xong! Gõ 'exit' để thoát.\n")

# ----------------------------
# Giao diện dòng lệnh
# ----------------------------
while True:
    text = input("Nhập câu: ").strip()
    if not text:
        continue
    if text.lower() in ("exit", "quit"):
        break

    probs = model.predict(text, vocab)
    pred = probs.argmax()
    print(
        f"→ {LABELS[pred].upper()} ({[round(float(p), 4) for p in probs]})\n")
