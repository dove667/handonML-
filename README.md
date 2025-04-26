# handonML
这是本人在25spring写的学习笔记兼实战项目，当时我已经对ML、DL理论有了比较深入的了解，但是一直找不到好的方法练习具体的代码实现：读官方文档效率太低，犹如大海捞针，网上优质视频也不多，而且学习写代码最重要的是自己动手写。  

本项目涵盖了数据集下载、清洗、预处理，模型流水线的搭建、训练和评估一整套流程，notebook中穿插了我自己写代码的时候debug过程中的血泪教训和总结，scripts中包含完整的代码实现。  

本项目适合对机器学习、深度学习及其各种python库有一定了解的朋友，通过由简至难的项目掌握sklearn、xgboost、pytorch等包中的基本使用方法和部分进阶技巧。

---

## 📋 项目结构

```text
├── data/                  # 各阶段示例数据集
│   ├── Adult/              # Phase I 示例：Adult
│   ├── California/            # Phase I 示例：加利福尼亚价
│   ├── titanic/           # Phase II 示例：泰坦尼克生存测
│   ├── cifar10/           # Phase II 示例：CIFAR-10
│   ├── imdb/              # Phase II 示例：IMDB 影评
│   └── custom_detection/  # Phase III 示例：目标检测
│
├── notebooks/             # 配套 Jupyter Notebook
│
├── scripts/BestPractise   # Python 脚本（数据预处理、模型训练、评估）
│   ├── ...           
│
├── requirements.txt       # 依赖列表（scikit-learn, torch, torchvision, ...）
├── README.md              # 本文件
└── TODO.md                # 详细每日待办清单
```

---


# 📚 学习规划详情

## Phase I：机器学习基础 (Week 1)

### Week 1: 传统机器学习工具链
* **Day 1-2**：数据预处理与基础分类
  * 工具：`SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `train_test_split`
  * 模型：`LogisticRegression`
  * 评估：`classification_report`
  * 实战：UCI Adult 数据集收入预测 (目标准确率 >70%)

* **Day 3-4**：回归模型与模型优化
  * 模型：`LinearRegression`, ``XGBoost``
  * 优化：`cross_val_score`, `GridSearchCV`
  * 实战：California Housing 房价预测 (目标 R² >0.7)

* **Day 5-7**：聚类与降维
  * 特征： `PCA`
  * 模型：`KMeans` `GMM` `SpectralClustering`
  * 实战：MNIST 手写数字聚类与可视化（KMeans 聚类，t-SNE/PCA 可视化，评估聚类效果）

## Phase II：深度学习实战 (Week 2-4)

### Week 2: PyTorch 基础与 MLP (Day 8-14)
* **Day 8-9**：PyTorch 张量操作 & GPU 加速
* **Day 10**：torch.nn/optim/utils.data 基础操作
* **Day 11**：2层 MLP on MNIST 架构搭建
* **Day 12**：MLP 训练 & 精度评估（准确率>90%）
* **Day 13**：模型导出（joblib + ONNX）

### Week 3: 卷积神经网络与进阶 (Day 15-21)
* **Day 14-16**：简单 CNN on CIFAR10 架构搭建与训练（目标准确率 ≥70%）
* **Day 17**：模型评估 & 调优（进阶尝试 ResNet18，目标准确率 ≥90%）
* **Day 18**：LSTM 文本分类数据预处理
* **Day 19**：LSTM 模型搭建与训练
* **Day 20**：IMDB 精度评估
* **Day 21**：学习率调度、混合精度训练、TensorBoard 可视化、复盘 & 文档

## Phase III：综合项目与前沿探索 (Week 4+)

* **Day 22-24**：自定义目标检测数据预处理 & YOLOv5 训练脚本（目标 mAP ≥0.7）
* **Day 25**：YOLOv5 fine-tuning & 调优
* **Day 26-27**：DCGAN 架构搭建、训练 & 生成样本，StyleGAN 进阶
* **Day 28**：Transformer 基础（nn.Transformer, MultiheadAttention 等）
* **Day 29**：Hugging Face Transformers/BERT/ViT 实战（如BERT fine-tuning on SST-2, ViT on CIFAR-10，准确率≥85%/90%）
* **Day 30**：整理代码至 GitHub，撰写项目文档 & 后续规划（强化学习、AutoML）

## 📈 项目里程碑

| 周数  | 主要目标 | 验收标准 |
|-------|---------|----------|
| Week 1 | 机器学习基础 | Adult 数据集准确率 >70%<br>California Housing R² >0.7 |
| Week 2 | PyTorch 基础 | MNIST 分类准确率 >90% |
| Week 3 | CNN & RNN | CIFAR-10 ≥75%，IMDB ≥85% |
| Week 4+ | 综合/前沿 | YOLOv5/Transformer/BERT/ViT 等前沿模型实战 |

---

- **Day 30**：整理代码至 GitHub，撰写项目文档，规划后续（强化学习、AutoML） 

---
