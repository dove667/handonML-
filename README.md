# handonML-
这是本人在25spring写得学习笔记，适合对机器学习、深度学习及其各种python库有一定了解但是缺乏代码和实战经验朋友，通过由简至难的项目掌握sklearn、pytorch、etc.
---

## 📋 项目结构

```text
├── data/                  # 各阶段示例数据集
│   ├── Adult/              # Phase I 示例：Adult
│   ├── California/            # Phase I 示例：加利福尼亚房价
│   ├── titanic/           # Phase II 示例：泰坦尼克生存预测
│   ├── cifar10/           # Phase II 示例：CIFAR-10
│   ├── imdb/              # Phase II 示例：IMDB 影评
│   └── custom_detection/  # Phase III 示例：目标检测
│
├── notebooks/             # 配套 Jupyter Notebook（按天命名）
│
├── scripts/               # Python 脚本（数据预处理、模型训练、评估）
│   ├── phase1/            # Phase I 脚本
│   ├── phase2/            # Phase II 脚本
│   └── phase3/            # Phase III 脚本
│
├── requirements.txt       # 依赖列表（scikit-learn, torch, torchvision, ...）
├── README.md              # 本文件
└── TODO.md                # 详细每日待办清单
```

---


## 📅 学习规划概览

| 阶段 | 时间       | 核心目标                                   |
|------|------------|---------------------------------------------|
| Phase I  | Day 1–7   | Scikit-learn 核心 API + PyTorch 张量与自动微分  |
| Phase II | Day 8–21  | 进阶特征工程 & 集成学习 + CNN/RNN/优化调度      |
| Phase III| Day 22–30 | 端到端项目：目标检测、GAN + 模型部署与性能分析  |

### Phase I：基础巩固与工具熟练（Day 1–7）
- **Day 0（可选）**：环境搭建与依赖安装
- **Day 1–2**：数据预处理（`SimpleImputer`, `StandardScaler`, `OneHotEncoder`）、`train_test_split`、`LogisticRegression` + `classification_report`；任务：UCI`Adult`数据集 预测收入 >50K 或 <=50K（准确率 >70%）

- **Day 3**：`LinearRegression`, `DecisionTreeRegressor`, `cross_val_score`, `GridSearchCV`；任务：`California Housing`，目标 R² >0.7。

- **Day 4–5**：PyTorch 张量、`.cuda()`, `autograd.backward()`；
没有GPU移步致kaggle notebook或者colab  

- **Day 6–7**：`nn.Module`, `nn.Linear`, `ReLU`, `MSELoss`, `Adam`；任务：2层 MLP on MNIST（准确率 >85%） 

### Phase II：核心功能实战（Day 8–21）
- **Day 8–10**：`SelectKBest`, `PCA`, `RandomForestClassifier`, `GradientBoostingClassifier`；任务：Titanic 生存预测（准确率 ≥80%），使用 KMeans 对 iris 数据集进行聚类并可视化结果。

- **Day 11–12**：`Pipeline`, `joblib`, ONNX 导出；任务：构建可复用预测管道  

- **Day 13–15**：`nn.Conv2d`, `MaxPool2d`, 数据增强；任务：在 CIFAR-10 上训练一个简单 CNN，目标准确率 ≥70%。  

进阶尝试 ResNet18，目标准确率 ≥90%。

- **Day 16–18**：`nn.LSTM`, `nn.Embedding`；任务：IMDB 情感分类  

- **Day 19–21**：`lr_scheduler`, `torch.cuda.amp`, TensorBoard；任务：对 CNN/RNN 模型进行优化与可视化

### Phase III：综合项目与高阶技能（Day 22–30） 
- **Day 22–24**：`YOLOv5` 目标检测；任务：使用预训练的 YOLOv5 在 COCO 数据集上 fine-tuning，然后在自定义数据集（如车辆）上测试，目标 mAP ≥0.7。 

- **Day 25–27**：`DCGAN` 生成对抗网络；任务：使用预训练的 StyleGAN 生成高质量动漫头像

- **Day 28-29**：Transformer 模型实战  
学习 `nn.Transformer`、`nn.MultiheadAttention`、`PositionalEncoding`。了解 Hugging Face 的 Transformers 库（如 BertTokenizer、BertModel）。  
NLP 任务：基于 Hugging Face 的预训练 BERT 模型，在 SST-2 数据集（Stanford Sentiment Treebank）上进行情感分类 fine-tuning，目标准确率 ≥90%。  
CV 任务（可选）：使用 Vision Transformer（ViT）在 CIFAR-10 数据集上进行分类，目标准确率 ≥85%。

- **Day 30**：整理代码至 GitHub，撰写项目文档，规划后续（强化学习、AutoML） 

---
