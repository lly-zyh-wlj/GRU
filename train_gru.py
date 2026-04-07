import re
import string
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
import os
import nltk
from nltk.corpus import stopwords

# 下载停用词（只运行一次）
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# 自定义噪声过滤
extra_noise = {'subject', 'from', 'organization', 'lines', 'writes', 'article'}

# -------------------- 0. 数据加载 --------------------
def load_20newsgroups_local(data_path="./20news-bydate", categories=None):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集未找到：{data_path}")
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")
    train_data = load_files(train_path, encoding='latin1', shuffle=False)
    test_data = load_files(test_path, encoding='latin1', shuffle=False)

    if categories is not None:
        all_cats = train_data.target_names
        keep_idx = [all_cats.index(c) for c in categories]
        train_mask = np.isin(train_data.target, keep_idx)
        test_mask = np.isin(test_data.target, keep_idx)
        X_train_raw = np.array(train_data.data)[train_mask]
        y_train_raw = np.array(train_data.target)[train_mask]
        X_test_raw = np.array(test_data.data)[test_mask]
        y_test_raw = np.array(test_data.target)[test_mask]
        label_map = {o: n for n, o in enumerate(keep_idx)}
        y_train = [label_map[t] for t in y_train_raw]
        y_test = [label_map[t] for t in y_test_raw]
    else:
        X_train_raw, y_train = train_data.data, train_data.target
        X_test_raw, y_test = test_data.data, test_data.target
    return X_train_raw, X_test_raw, y_train, y_test

# -------------------- 文本预处理 --------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'^subject:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^from:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^organization:.*$', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    words = text.split()
    words = [w for w in words if w not in stop_words and w not in extra_noise and len(w) > 2]
    return ' '.join(words)

# -------------------- 构建词表 --------------------
def build_vocab(texts, min_freq=3):
    freq = Counter()
    for t in texts:
        freq.update(t.split())
    w2i = {'<PAD>': 0, '<UNK>': 1}
    for w, c in freq.items():
        if c >= min_freq:
            w2i[w] = len(w2i)
    return w2i

# -------------------- 序列转换 --------------------
def text_to_seq(text, w2i, max_len=300):
    words = text.split()
    seq = [w2i.get(w, 1) for w in words]
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq += [0] * (max_len - len(seq))
    return seq

# -------------------- 模型：双向 GRU --------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, 1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        x = self.embedding(x)
        out, h = self.gru(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        h = self.dropout(h)
        return self.fc(h)

# -------------------- 训练 & 评估 --------------------
def train_epoch(model, loader, opt, crit, dev):
    model.train()
    loss_sum, acc_sum, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        pred = model(x)
        loss = crit(pred, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * x.size(0)
        acc_sum += (pred.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, acc_sum / total

def evaluate(model, loader, crit, dev):
    model.eval()
    loss_sum, acc_sum, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            pred = model(x)
            loss = crit(pred, y)
            loss_sum += loss.item() * x.size(0)
            acc_sum += (pred.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, acc_sum / total

# -------------------- 主函数 --------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用设备:", device)

    categories = ['alt.atheism', 'soc.religion.christian']
    X_train_raw, X_test_raw, y_train, y_test = load_20newsgroups_local(categories=categories)

    X_train_clean = [preprocess_text(t) for t in X_train_raw]
    X_test_clean = [preprocess_text(t) for t in X_test_raw]

    word_to_idx = build_vocab(X_train_clean)
    vocab_size = len(word_to_idx)
    print("词汇表大小:", vocab_size)

    # 划分训练/验证
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_clean, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    MAX_LEN = 300
    X_tr_seq = [text_to_seq(t, word_to_idx, MAX_LEN) for t in X_tr]
    X_val_seq = [text_to_seq(t, word_to_idx, MAX_LEN) for t in X_val]
    X_te_seq = [text_to_seq(t, word_to_idx, MAX_LEN) for t in X_test_clean]

    # 转 tensor
    def to_tensor(x):
        return torch.tensor(x, dtype=torch.long)

    train_ds = TensorDataset(to_tensor(X_tr_seq), to_tensor(y_tr))
    val_ds = TensorDataset(to_tensor(X_val_seq), to_tensor(y_val))
    test_ds = TensorDataset(to_tensor(X_te_seq), to_tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    # 模型
    model = GRUClassifier(vocab_size, embed_dim=128, hidden_dim=256).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    best_acc = 0
    best_state = None

    print("开始训练...\n")
    for epoch in range(30):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1:2d} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print("\n" + "="*50)
    print(f"最终测试集准确率 = {test_acc:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()