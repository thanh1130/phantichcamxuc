# test_model.py
# Dùng model.txt để dự đoán cảm xúc cho câu mới.

import re
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_FILE = "model.pt"
LABELS = ["negative", "neutral", "positive"]




def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t]


class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward_ids(self, ids_list):
        reps = []
        for ids in ids_list:
            
            t = torch.tensor(ids, dtype=torch.long)
            e = self.emb(t)
            reps.append(e.mean(0))  
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


saved = torch.load(MODEL_FILE, map_location="cpu") 
vocab = saved["vocab"]
cfg = saved["config"]
state = saved["state_dict"]

model = SimpleModel(len(vocab), cfg["embed_dim"], cfg["hidden_dim"])
model.load_state_dict(state)
model.eval()

print("✅ Mô hình đã tải xong! Gõ 'exit' để thoát.\n")


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
