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
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data():
    train_df = pd.read_csv(os.path.join(BASE_DIR, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(BASE_DIR, "test_data_unlabeled.csv"))
    X_train = train_df["text"].astype(str).tolist()
    y_train = train_df["target"].astype(int).values
    X_test = test_df["text"].astype(str).tolist()
    return X_train, y_train, X_test


def clean_text(text):
    # 邮件头在第一个空行之前，直接裁掉
    parts = text.split("\n\n", 1)
    body = parts[1] if len(parts) > 1 else parts[0]
    t = body.lower()
    # 去掉 URL、邮箱，只留字母
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"\S+@\S+\.\S+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# 换不同随机种子重跑，看结果稳不稳
STABILITY_SEEDS = [42, 2024, 7]


def run_stability(X_clean, y):
    print("=" * 60)
    print(f"步骤 8/8  稳定性重复实验（{len(STABILITY_SEEDS)} 个随机种子 × 2 个模型）")
    models = {
        "MLP(hidden=100)": lambda rs: MLPClassifier(
            hidden_layer_sizes=(100,), alpha=0.001, max_iter=300, random_state=rs),
        "SVM(linear C=1)": lambda rs: SVC(
            kernel="linear", C=1.0, random_state=rs),
    }
    rows = []
    for mname, maker in models.items():
        accs = []
        for rs in STABILITY_SEEDS:
            # 每个种子重新做 8:2 划分
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_clean, y, test_size=0.2, stratify=y, random_state=rs)
            vec = TfidfVectorizer(
                max_features=5000, stop_words="english",
                sublinear_tf=True, ngram_range=(1, 1))
            X_tr = vec.fit_transform(X_tr)
            X_va = vec.transform(X_va)
            model = maker(rs)
            model.fit(X_tr, y_tr)
            acc = accuracy_score(y_va, model.predict(X_va))
            accs.append(acc)
            rows.append((mname, rs, acc))
            print(f"  [{mname:<14} | seed={rs:<5}] 验证集准确率 = {acc:.4f}")
        mean, std = float(np.mean(accs)), float(np.std(accs))
        print(f"  -> {mname}: 均值 {mean:.4f} ± {std:.4f}（{len(accs)} 次）")

    sdf = pd.DataFrame(rows, columns=["模型", "随机种子", "验证集准确率"])
    sdf.to_csv(os.path.join(BASE_DIR, "stability_results.csv"),
               index=False, encoding="utf-8-sig")

    # 误差棒图
    agg = sdf.groupby("模型")["验证集准确率"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(agg["模型"], agg["mean"], yerr=agg["std"], capsize=6,
           color=["#59a14f", "#f28e2b"], edgecolor="black", linewidth=0.6)
    for i, row in agg.iterrows():
        ax.text(i, row["mean"] + row["std"] + 0.004,
                f"{row['mean']:.4f}±{row['std']:.4f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Stability Across Random Seeds (mean ± std)")
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "stability_comparison.png"), dpi=150)
    plt.close(fig)
    print("  已保存 stability_results.csv / stability_comparison.png")


def main():
    t0 = time.time()
    print("=" * 60)
    print("步骤 1/7  加载数据")
    X_train_raw, y_train_full, X_test_raw = load_data()
    print(f"  原始训练样本: {len(X_train_raw)}   测试样本: {len(X_test_raw)}")

    print("=" * 60)
    print("步骤 2/7  数据清洗（去邮件头/URL/标点/数字，转小写）")
    X_train_clean = [clean_text(x) for x in X_train_raw]
    X_test_clean = [clean_text(x) for x in X_test_raw]
    X_train_clean = [x if x else "[empty]" for x in X_train_clean]
    X_test_clean = [x if x else "[empty]" for x in X_test_clean]
    print("  清洗示例（第一个训练样本清洗后前 120 字符）:")
    print("   ", X_train_clean[0][:120])

    print("=" * 60)
    print("步骤 3/7  划分训练集 / 验证集（8:2 分层，测试集封存）")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_clean, y_train_full,
        test_size=0.2, stratify=y_train_full, random_state=RANDOM_STATE,
    )
    print(f"  最终训练集: {len(X_train)}   验证集: {len(X_val)}   测试集(封存): {len(X_test_clean)}")

    print("=" * 60)
    print("步骤 4/7  TF-IDF 特征化（只在训练集 fit，同一规则套验证/测试）")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 1),
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test_clean)
    print(f"  词表大小: {X_train_tfidf.shape[1]}")

    print("=" * 60)
    print("步骤 5/7  训练多个模型，在验证集上对比调参")
    candidates = [
        ("NB",  "alpha=0.1",        MultinomialNB(alpha=0.1)),
        ("NB",  "alpha=1.0",        MultinomialNB(alpha=1.0)),
        ("SVM", "linear C=0.1",     SVC(kernel="linear", C=0.1, random_state=RANDOM_STATE)),
        ("SVM", "linear C=1.0",     SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)),
        ("SVM", "linear C=10",      SVC(kernel="linear", C=10,  random_state=RANDOM_STATE)),
        ("LR",  "C=0.1",            LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE)),
        ("LR",  "C=1.0",            LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)),
        ("MLP", "hidden=(50,)",     MLPClassifier(hidden_layer_sizes=(50,),  alpha=0.001, max_iter=300, random_state=RANDOM_STATE)),
        ("MLP", "hidden=(100,)",    MLPClassifier(hidden_layer_sizes=(100,), alpha=0.001, max_iter=300, random_state=RANDOM_STATE)),
    ]

    results = []
    best = None
    for name, desc, model in candidates:
        ts = time.time()
        model.fit(X_train_tfidf, y_train)
        acc = accuracy_score(y_val, model.predict(X_val_tfidf))
        used = time.time() - ts
        results.append((name, desc, acc, used))
        print(f"  [{name:>3} | {desc:<16}] 验证集准确率 = {acc:.4f}  耗时 {used:.1f}s")
        if best is None or acc > best[0]:
            best = (acc, name, desc, model)

    df = pd.DataFrame(results, columns=["模型", "参数", "验证集准确率", "耗时(s)"])
    df.to_csv(os.path.join(BASE_DIR, "validation_results.csv"), index=False, encoding="utf-8-sig")
    print("\n验证集结果汇总（已保存 validation_results.csv）:")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    acc_best, name_best, desc_best, model_best = best
    print("=" * 60)
    print(f"步骤 6/7  锁定最优方案：{name_best} ({desc_best})，验证集准确率 {acc_best:.4f}")
    print("  最优模型在验证集上的分类报告:")
    print(classification_report(y_val, model_best.predict(X_val_tfidf),
                                digits=4, zero_division=0))

    print("=" * 60)
    print("步骤 7/7  用最优模型预测封存的测试集 -> predictions.csv")
    y_test_pred = model_best.predict(X_test_tfidf)
    out_path = os.path.join(BASE_DIR, "predictions.csv")
    pd.DataFrame(y_test_pred).to_csv(out_path, index=False, header=False)
    n_pred = len(y_test_pred)
    print(f"  已保存 {out_path}，共 {n_pred} 行（应为 2457）")
    print(f"  预测标签分布: {pd.Series(y_test_pred).value_counts().sort_index().to_dict()}")

    print("=" * 60)
    print("生成可视化图表（验证集对比柱状图 + 混淆矩阵）")

    # 图1：各模型最佳验证准确率
    best_per_model = df.groupby("模型")["验证集准确率"].max().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(best_per_model.index, best_per_model.values,
                  color=["#4e79a7", "#f28e2b", "#e15759", "#59a14f"])
    for b, v in zip(bars, best_per_model.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Best Validation Accuracy by Model")
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "validation_comparison.png"), dpi=150)
    plt.close(fig)

    # 图2：最优模型在验证集上的混淆矩阵
    cm = confusion_matrix(y_val, model_best.predict(X_val_tfidf))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix on Validation Set ({name_best})")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    print("  已保存 validation_comparison.png / confusion_matrix.png")

    run_stability(X_train_clean, y_train_full)

    print("=" * 60)
    print(f"全部完成，总耗时 {time.time() - t0:.1f}s。"
          f"提交文件：predictions.csv（验证集最优模型 {name_best} {desc_best}）")


if __name__ == "__main__":
    main()
