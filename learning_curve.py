# -*- coding: utf-8 -*-
import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data():
    train_df = pd.read_csv(os.path.join(BASE_DIR, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(BASE_DIR, "test_data_unlabeled.csv"))
    X_train = train_df["text"].astype(str).tolist()
    y_train = train_df["target"].astype(int).values
    X_test = test_df["text"].astype(str).tolist()
    return X_train, y_train, X_test


def clean_text(text):
    parts = text.split("\n\n", 1)
    body = parts[1] if len(parts) > 1 else parts[0]
    t = body.lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"\S+@\S+\.\S+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    t0 = time.time()
    print("加载数据...")
    X_raw, y_full, _ = load_data()
    X_clean = [clean_text(x) for x in X_raw]
    X_clean = [x if x else "[empty]" for x in X_clean]

    X_train, X_val, y_train, y_val = train_test_split(
        X_clean, y_full, test_size=0.2, stratify=y_full, random_state=RANDOM_STATE)
    print(f"训练 {len(X_train)} / 验证 {len(X_val)}")

    vec = TfidfVectorizer(max_features=5000, stop_words="english",
                          sublinear_tf=True, ngram_range=(1, 1))
    X_tr = vec.fit_transform(X_train)
    X_va = vec.transform(X_val)
    print(f"词表 {X_tr.shape[1]}")

    MAX_EPOCHS = 40
    model = MLPClassifier(hidden_layer_sizes=(100,), alpha=0.001,
                          max_iter=1, warm_start=True, random_state=RANDOM_STATE)

    epochs, losses, accs = [], [], []
    prev_loss = None
    for e in range(1, MAX_EPOCHS + 1):
        model.fit(X_tr, y_train)
        loss = float(model.loss_)
        acc = accuracy_score(y_val, model.predict(X_va))
        epochs.append(e)
        losses.append(loss)
        accs.append(acc)
        print(f"epoch {e:2d}  loss={loss:.4f}  val_acc={acc:.4f}")
        # 连续几轮 loss 都不动了，说明收敛了，停
        if prev_loss is not None and abs(prev_loss - loss) < 5e-4:
            if len(epochs) >= 3 and all(
                abs(losses[-i] - losses[-i-1]) < 5e-4 for i in (1, 2)
            ):
                break
        prev_loss = loss

    # 双轴，左边 loss 右边准确率
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax1.plot(epochs, losses, color="#4e79a7", marker="o", markersize=3,
             linewidth=1.5, label="training loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss", color="#4e79a7")
    ax1.tick_params(axis="y", labelcolor="#4e79a7")
    ax1.set_ylim(0, max(losses) * 1.15)

    ax2 = ax1.twinx()
    ax2.plot(epochs, accs, color="#e15759", marker="s", markersize=3,
             linewidth=1.5, label="validation accuracy")
    ax2.set_ylabel("Validation Accuracy", color="#e15759")
    ax2.tick_params(axis="y", labelcolor="#e15759")
    ax2.set_ylim(0.7, 1.0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
    ax1.set_title(f"MLP Training Curve (hidden=100, {len(epochs)} epochs)")
    fig.tight_layout()
    out = os.path.join(BASE_DIR, "learning_curve.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"已保存 {out}，共 {len(epochs)} 轮，耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
