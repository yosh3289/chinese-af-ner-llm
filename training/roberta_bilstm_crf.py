# -*- coding: utf-8 -*-
"""
RoBERTa-BiLSTM-CRF 训练/评测脚本（中文房颤电子病历 NER）。

说明 / NOTE
-----------
模型结构改写自公开仓库
    jesska/Chinese-medical-entity-recognition(n)
    https://github.com/jesska/Chinese-medical-entity-recognitionn
（RoBERTa/BERT 编码器 → Dropout → 2 层 BiLSTM → Linear → CRF）。
超参数对齐论文 Table 4：learning_rate=3e-5, weight_decay=0.01, batch_size=16,
50 epochs, gradient clipping=5, 序列长度 128, 全参数微调。原仓库为 8 类 CLUE 数据，
此处适配为本研究的 5 类房颤实体（字符级 BIO 标注）。

依赖: torch, transformers, pytorch-crf  (pip install pytorch-crf)
"""
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from torchcrf import CRF

# ====================== 配置（对应论文 Table 4） ======================
ENCODER_NAME = "hfl/chinese-roberta-wwm-ext"   # RoBERTa-wwm；可换 chinese-bert-wwm
TRAIN_DATA_PATH = "你的训练集.json"   # 每行一条 {"text": ..., "label": {类别: {实体: [[start,end],...]}}}
TEST_DATA_PATH = "你的测试集.json"
OUTPUT_DIR = "./output/roberta_bilstm_crf"

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
CLIP_GRAD = 5
LSTM_DROPOUT = 0.5
HIDDEN_DROPOUT = 0.1

ENTITY_CATEGORIES = ["身体部位", "症状表现", "检查检验", "治疗方式", "疾病及诊断"]
# ====================================================================

# 构造 BIO 标签表
LABELS = ["O"]
for cat in ENTITY_CATEGORIES:
    LABELS += [f"B-{cat}", f"I-{cat}"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def char_bio(text, label):
    """把 {类别: {实体: [[start,end]]}} 的跨度标注转成字符级 BIO（end 为开区间）。"""
    tags = ["O"] * len(text)
    for cat, ents in label.items():
        if cat not in ENTITY_CATEGORIES:
            continue
        for _, spans in ents.items():
            for s, e in spans:
                if 0 <= s < e <= len(text):
                    tags[s] = f"B-{cat}"
                    for i in range(s + 1, e):
                        tags[i] = f"I-{cat}"
    return tags


class NerDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                text, tags = d["text"], char_bio(d["text"], d.get("label", {}))
                self.samples.append((list(text), tags))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chars, tags = self.samples[idx]
        enc = self.tokenizer(
            chars, is_split_into_words=True, truncation=True,
            max_length=MAX_LEN, padding="max_length", return_tensors="pt",
        )
        word_ids = enc.word_ids(0)
        label_ids = []
        for wid in word_ids:
            # 特殊标记 [CLS]/[SEP]/[PAD] 记为 O，再由 attention_mask 屏蔽 PAD
            label_ids.append(LABEL2ID[tags[wid]] if wid is not None else LABEL2ID["O"])
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


class RobertaBiLSTMCRF(nn.Module):
    def __init__(self, encoder_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(HIDDEN_DROPOUT)
        self.bilstm = nn.LSTM(
            input_size=hidden, hidden_size=hidden // 2, num_layers=2,
            bidirectional=True, batch_first=True, dropout=LSTM_DROPOUT,
        )
        self.classifier = nn.Linear(hidden, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        seq = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        seq = self.dropout(seq)
        lstm_out, _ = self.bilstm(seq)
        logits = self.classifier(lstm_out)
        mask = attention_mask.bool()
        if labels is not None:
            loss = -self.crf(logits, labels, mask=mask, reduction="mean")
            return loss
        return self.crf.decode(logits, mask=mask)


def evaluate(model, loader):
    """实体级严格匹配 P/R/F1（微平均）。"""
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            preds = model(input_ids, attention_mask)
            golds = batch["labels"].tolist()
            for p_seq, g_seq, m in zip(preds, golds, attention_mask.tolist()):
                length = sum(m)
                g_seq = [ID2LABEL[g] for g in g_seq[:length]]
                p_seq = [ID2LABEL[p] for p in p_seq[:length]]
                ge, pe = set(spans_of(g_seq)), set(spans_of(p_seq))
                tp += len(ge & pe)
                fp += len(pe - ge)
                fn += len(ge - pe)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def spans_of(tags):
    """从 BIO 序列抽取 (类别, start, end) 实体跨度。"""
    spans, start, cat = [], None, None
    for i, t in enumerate(tags):
        if t.startswith("B-"):
            if cat is not None:
                spans.append((cat, start, i))
            cat, start = t[2:], i
        elif t.startswith("I-") and cat == t[2:]:
            continue
        else:
            if cat is not None:
                spans.append((cat, start, i))
            cat, start = None, None
    if cat is not None:
        spans.append((cat, start, len(tags)))
    return spans


def main():
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    train_loader = DataLoader(NerDataset(TRAIN_DATA_PATH, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(NerDataset(TEST_DATA_PATH, tokenizer), batch_size=BATCH_SIZE)

    model = RobertaBiLSTMCRF(ENCODER_NAME, len(LABELS)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["labels"].to(DEVICE),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()
            total += loss.item()
        p, r, f1 = evaluate(model, test_loader)
        print(f"Epoch {epoch:02d} | loss {total/len(train_loader):.4f} | P {p:.4f} R {r:.4f} F1 {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pt")
            print(f"  -> saved best (F1={f1:.4f})")
    print(f"训练完成，最佳 F1 = {best_f1:.4f}")


if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
