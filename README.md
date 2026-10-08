# 当代人工智能 实验一：文本分类

## 基本信息
- 姓名：温正康
- 学号：10245501411
- 课程：当代人工智能

## 任务
对 20 Newsgroups 风格的新闻文本进行 10 分类。流程：文本清洗 → 8:2 分层划分验证集 → TF-IDF 特征化 → 多模型训练与调参对比 → 最优模型预测测试集。

## 环境
- Python 3.10（conda 环境名 `project1`）
- 依赖：numpy / pandas / scikit-learn / matplotlib

安装：
```bash
conda create -n project1 python=3.10
conda activate project1
pip install numpy pandas scikit-learn matplotlib
```

## 数据
将课程提供的 `data-Project1.zip` 解压到本目录后，应包含：
- `train_data.csv`（7368 条，含标签）
- `test_data_unlabeled.csv`（2457 条，无标签）

> 数据文件体积较大，未纳入本仓库，请从课程材料获取。

## 运行
```bash
conda activate project1
python experiment.py
```

运行后输出：
- `predictions.csv`：对测试集的预测结果（每行一个标签，共 2457 行，无表头）
- `validation_results.csv`：9 组模型/参数的验证集准确率
- `validation_comparison.png`：各模型最佳验证准确率对比
- `confusion_matrix.png`：最优模型在验证集上的混淆矩阵

## 复现说明
代码固定 `random_state=42`，在同一环境下重复运行结果一致。

## 主要结果
| 模型 | 参数 | 验证集准确率 |
|---|---|---|
| **MLP** | hidden=(100,), alpha=1e-3 | **0.8921** |
| MLP | hidden=(50,) | 0.8901 |
| SVM | linear C=1.0 | 0.8867 |
| LR | C=1.0 | 0.8786 |
| NB | alpha=0.1 | 0.8772 |

最优模型：`MLPClassifier(hidden_layer_sizes=(100,), alpha=0.001, max_iter=300, random_state=42)`。
