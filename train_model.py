import os
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# ============================
NEG_FILE = "SA-training_negative.txt"
NEU_FILE = "SA-training_neutral.txt"
POS_FILE = "SA-training_positive.txt"
KIEMTRA_FILE = "kiemtra.txt"
MODEL_FILE = "model.pt"

EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 32
EMBED_DIM = 128
HIDDEN_DIM = 128
SEED = 42
MAX_VOCAB = 20000
MAX_LEN = 50
# ============================

random.seed(SEED)
torch.manual_seed(SEED)

# ---- Tokenizer ----


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t] or ["<unk>"]

# ---- Đọc dữ liệu ----


def load_file(path, label):
    if not os.path.exists(path):
        print("Thiếu file:", path)
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [(line.strip(), label) for line in f if line.strip()]


data = []
data += load_file(NEG_FILE, 0)
data += load_file(NEU_FILE, 1)
data += load_file(POS_FILE, 2)
if not data:
    raise SystemExit("Không có dữ liệu để train.")
random.shuffle(data)

# ---- Xây vocab từ dữ liệu huấn luyện ----
counter = Counter()
for text, _ in data:
    counter.update(tokenize(text))
most_common = counter.most_common(MAX_VOCAB)
vocab = {w: i + 2 for i, (w, _) in enumerate(most_common)}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

# ---- Dataset ----


class SentimentDataset(Dataset):
    def __init__(self, samples, vocab, max_len=MAX_LEN):
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len

    def encode(self, text):
        ids = [self.vocab.get(t, 1) for t in tokenize(text)]
        # cắt hoặc pad cho đủ độ dài
        ids = ids[: self.max_len]
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        return ids

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        return torch.tensor(self.encode(text)), torch.tensor(label)

    def __len__(self):
        return len(self.samples)


train_ds = SentimentDataset(data, vocab)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ---- Mô hình LSTM ----


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
        ids = [vocab.get(t, 1) for t in tokenize(text)]
        ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
        t = torch.tensor([ids])
        self.eval()
        with torch.no_grad():
            logits = self.forward(t)
            probs = F.softmax(logits, dim=-1)[0].numpy()
        return probs


# ---- Train ban đầu ----
model = LSTMModel(len(vocab), EMBED_DIM, HIDDEN_DIM)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = correct = total = 0
    for x, y in train_dl:
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    acc = correct / total
    print(f"Epoch {epoch}/{EPOCHS}  Loss={total_loss/total:.4f}  Acc={acc:.3f}")

# ---- Fine-tune bằng kiemtra.txt ----
if os.path.exists(KIEMTRA_FILE):
    print("\n🔧 Phát hiện file kiemtra.txt — bắt đầu fine-tune...")
    samples = []
    with open(KIEMTRA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().rsplit("\t", 1)
            if len(parts) == 2:
                text, label = parts
                try:
                    samples.append((text, int(label)))
                except:
                    pass

    if samples:
        fine_ds = SentimentDataset(samples, vocab)
        fine_dl = DataLoader(fine_ds, batch_size=BATCH_SIZE, shuffle=True)
        fine_opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        for e in range(1, 3):  # học nhẹ 2 epoch
            model.train()
            total_loss = correct = total = 0
            for x, y in fine_dl:
                logits = model(x)
                loss = loss_fn(logits, y)
                fine_opt.zero_grad()
                loss.backward()
                fine_opt.step()
                total_loss += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                total += len(y)
            print(
                f"[Fine-tune] Epoch {e}/2  Loss={total_loss/total:.4f}  Acc={correct/total:.3f}")
    else:
        print("⚠️ Không đọc được dữ liệu từ kiemtra.txt")

# ---- Lưu model ----
save_data = {
    "vocab": vocab,
    "config": {
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
    },
    "state_dict": model.state_dict(),
}
torch.save(save_data, MODEL_FILE)
print(f"\n✅ Đã lưu mô hình LSTM vào {MODEL_FILE}")
