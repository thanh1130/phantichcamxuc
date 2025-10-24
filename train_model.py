import os
import re
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset, DataLoader

# ============================
NEG_FILE = "SA-training_negative.txt"
NEU_FILE = "SA-training_neutral.txt"
POS_FILE = "SA-training_positive.txt"
TEST_FILE = "test.txt"
MODEL_FILE = "model.pt"

EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 32
EMBED_DIM = 128
HIDDEN_DIM = 64
SEED = 42
MAX_VOCAB = 20000
# ============================

random.seed(SEED)
torch.manual_seed(SEED)

# ---- Tokenizer cơ bản ----


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t]

# ---- Đọc dữ liệu ----


def load_file(path, label):
    if not os.path.exists(path):
        print("❌ Thiếu file:", path)
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [(line.strip(), label) for line in f if line.strip()]


data = []
data += load_file(NEG_FILE, 0)
data += load_file(NEU_FILE, 1)
data += load_file(POS_FILE, 2)
if not data:
    raise SystemExit("❌ Không có dữ liệu để train.")
random.shuffle(data)

# ---- Xây vocab ----
counter = Counter()
for text, _ in data:
    counter.update(tokenize(text))
most_common = counter.most_common(MAX_VOCAB)
vocab = {w: i + 1 for i, (w, _) in enumerate(most_common)}
vocab["<UNK>"] = 0

# ---- Dataset ----


class SentimentDataset(Dataset):
    def __init__(self, samples, vocab):
        self.samples = samples
        self.vocab = vocab

    def __len__(self):
        return len(self.samples)

    def encode(self, text):
        ids = [self.vocab.get(t, 0) for t in tokenize(text)]
        return ids if ids else [0]

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        return self.encode(text), label


def collate(batch):
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels)


# ---- Tạo train/test ----
train_ds = SentimentDataset(data, vocab)
train_dl = DataLoader(train_ds, BATCH_SIZE, True, collate_fn=collate)

# đọc dữ liệu test.txt (nếu có)
if os.path.exists(TEST_FILE):
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_data = [line.strip() for line in f if line.strip()]
    test_samples = [(t, 1) for t in test_data]  # nhãn giả để không lỗi
    test_ds = SentimentDataset(test_samples, vocab)
    test_dl = DataLoader(test_ds, BATCH_SIZE, False, collate_fn=collate)
else:
    print("⚠️ Không có file test.txt, bỏ qua bước test.")
    test_dl = None

# ---- Mô hình ----


class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward_ids(self, batch_ids):
        reps = []
        for ids in batch_ids:
            t = torch.tensor(ids, dtype=torch.long)
            e = self.emb(t)
            reps.append(e.mean(0))  # trung bình embedding
        x = torch.stack(reps)
        h = F.relu(self.fc1(x))
        return self.fc2(h)

    def predict(self, text, vocab):
        ids = [vocab.get(t, 0) for t in tokenize(text)] or [0]
        self.eval()
        with torch.no_grad():
            logits = self.forward_ids([ids])
            probs = F.softmax(logits, dim=-1)[0].numpy()
        return probs


# ---- Train ----
model = SimpleModel(len(vocab), EMBED_DIM, HIDDEN_DIM)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = correct = total = 0
    for texts, labels in train_dl:
        logits = model.forward_ids(texts)
        loss = loss_fn(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    acc = correct / total
    print(f"Epoch {epoch}/{EPOCHS}  Loss={total_loss/total:.4f}  Acc={acc:.3f}")

# ---- Dự đoán nhanh trên test.txt (nếu có) ----
if test_dl:
    model.eval()
    print("\n🔎 Kết quả dự đoán test.txt:")
    with torch.no_grad():
        for texts, _ in test_dl:
            logits = model.forward_ids(texts)
            preds = logits.argmax(1).tolist()
            for ids, p in zip(texts, preds):
                label = ["NEG", "NEU", "POS"][p]
                print("→", label)

# ---- Lưu model ----
save_data = {
    "vocab": vocab,
    "config": {
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "epochs": EPOCHS,
    },
    "state_dict": model.state_dict(),
}
torch.save(save_data, MODEL_FILE)
print(f"\n✅ Đã lưu mô hình vào {MODEL_FILE}")
