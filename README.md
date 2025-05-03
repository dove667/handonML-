# handonML
这是本人在25spring写的学习笔记兼实战项目，当时我已经对ML、DL理论有了比较深入的了解，但是一直找不到好的方法练习具体的代码实现：读官方文档效率太低，网上优质视频也不多，而且我觉得学习写代码最重要的是自己动手写。  

本项目涵盖了数据集下载、清洗、预处理，模型流水线的搭建、训练和评估一整套流程，涵盖了sklearn中大部分常用的API和常用的回归、分类、聚类、降维模型。notebook中穿插了我自己写代码的时候debug过程中的血泪教训和总结，探索过程都保留了下来，会显得有些乱。scripts中包含了对应天数的优化训练脚本。  

本项目适合对机器学习、深度学习及其各种python库有一定了解的朋友，通过由简至难的项目掌握sklearn、pytorch中的基本使用方法和部分进阶技巧。

---

## 📋 项目结构

```text
├── data/                  # 各阶段示例数据集
│   ├── Adult/             # Phase I 示例：Adult
│   ├── California/        # Phase I 示例：加利福尼亚价
│   ├── titanic/           # Phase II 示例：泰坦尼克生存测
│   ├── cifar10/           # Phase II 示例：CIFAR-10
│   ├── imdb/              # Phase II 示例：IMDB 影评
│   └── custom_detection/  # Phase III 示例：目标检测
│
├── notebooks/             # 配套 Jupyter Notebook
│
├── scripts/   # Python 脚本（数据预处理、模型训练、评估）
│   ├── ML/
|   ├── DL/
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

* **Day 5-6**：聚类与降维
  * 降维：`PCA`
  * 模型：`KMeans` `GMM` `SpectralClustering`
  * 实战：MNIST 手写数字聚类与可视化（KMeans 聚类，t-SNE/PCA 可视化，评估聚类效果）
  
* **Day 7**： `SVC` `SVR` 比较分析支持向量机的特点

## Phase II：深度学习实战 (Week 2-3)

### Week 2: PyTorch 基础与 经典网络架构 (Day 8-14)
> [pytorch4h速通 youtube](https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=1)   
* **Day 8**：`tensor` & `autograd` & `cuda` &  `Dataset` & `DataLoader`
* **Day 9** `nn.Module` / `optim ` / `onnx` / `tensorboard`
* **Day 10**：MLP on MNIST 架构搭建 & 训练 & 精度评估（准确率>90%）
* **Day 11**：简单 CNN on CIFAR10 架构搭建与训练（目标准确率 ≥70%）
* **Day 12**：简单 RNN on IMDB 情感分类 (目标准确率 ≥80%)
* **Day 13**：模型导出（joblib + ONNX）

### Week 3: 先进网络架构 & transformer (Day 15-21)
> 本周聚焦 Huggingface Transformers 框架，深入理解 BERT、Vision Transformer (ViT)、Prompt Engineering、微调（Fine-tuning）、轻量化方法（如LoRA原理）等。

* **Day 15**：Huggingface Transformers 基础（模型加载、Tokenizer、Pipeline 快速推理）
* **Day 16**：BERT 文本分类实战（SST-2/IMDB，微调与评估，目标准确率≥85%）
* **Day 17**：Prompt Engineering 入门（Zero-shot/Prompt-based 分类、文本生成）
* **Day 18**：Vision Transformer (ViT) on CIFAR-10（微调与评估，目标准确率≥85%）
* **Day 19**：轻量化微调方法原理（LoRA/Adapter 理论，CPU 上小模型实验）
* **Day 20**：Huggingface Datasets & Trainer API 进阶（自定义数据集、训练流程）
* **Day 21**：总结复盘 & 代码整理，撰写 transformer 应用经验文档

## Phase III：综合项目与前沿探索 (Week 4)

> 本阶段聚焦真实场景下的综合应用与前沿探索，涵盖文本/图像生成、微调、轻量化部署、RLHF、AutoML 等。

* **Day 22-23**：文本生成/摘要/问答系统（基于 Huggingface Pipeline，尝试 T5/BART/DistilGPT2 等小模型）
* **Day 24-25**：图像分类/检索/生成（ViT 微调、Diffusers 文生图小模型实验）
* **Day 26**：LoRA/Adapter 微调小模型实战（如 DistilBERT/DistilGPT2）
* **Day 27**：AutoML/轻量化部署（Optuna 超参搜索、ONNX 导出、CPU 推理优化）
* **Day 28**：RLHF 理论与小规模实验（如 reward model 微调、数据构造流程）
* **Day 29**：综合项目：端到端文本/图像智能应用（如“智能摘要+检索”或“文本生成+情感分析”一体化系统）
* **Day 30**：整理代码至 GitHub，撰写项目文档 & 后续规划（如强化学习、MLOps、AIGC 等）

## 📈 项目里程碑

| 周数  | 主要目标 | 验收标准 |
|-------|---------|----------|
| Week 1 | 机器学习基础 | Adult 收入预测acc >70%<br>California Housing 房价预测 R² >0.7 |
| Week 2 | PyTorch 基础 | MNIST 数字分类acc >90% <br> CIFAR10 对象分类 acc>70% <br> IMDB 情感分类 acc >80%
| Week 3 | Transformer & 先进网络 | BERT/ViT 微调准确率 ≥85%，Prompt/LoRA 理论与实验 |
| Week 4+ | 综合/前沿 | 文本/图像生成、LoRA 微调、AutoML、RLHF 等前沿项目实战 |

---
