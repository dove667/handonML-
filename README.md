# handonML-
Learn to use scikict-learn, pytorch and etc through hand on practice and projects
---

## 📋 项目结构

```text
├── data/                  # 各阶段示例数据集
│   ├── iris/              # Phase I 示例：鸢尾花
│   ├── boston/            # Phase I 示例：波士顿房价
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
- **Day 1–2**：数据预处理（`SimpleImputer`, `StandardScaler`, `OneHotEncoder`）、`train_test_split`、`LogisticRegression` + `classification_report`；任务：`iris` 分类（准确率 >95%）
- **Day 3**：`LinearRegression`, `DecisionTreeRegressor`, `cross_val_score`, `GridSearchCV`；任务：波士顿回归（R² >0.7）
- **Day 4–5**：PyTorch 张量、`.cuda()`, `autograd.backward()`；手写线性回归对比 scikit-learn
- **Day 6–7**：`nn.Module`, `nn.Linear`, `ReLU`, `MSELoss`, `Adam`；任务：2层 MLP on MNIST（准确率 >85%）

### Phase II：核心功能实战（Day 8–21）
- **Day 8–10**：`SelectKBest`, `PCA`, `RandomForestClassifier`, `GradientBoostingClassifier`；任务：Titanic 生存预测（准确率 ≥80%）
- **Day 11–12**：`Pipeline`, `joblib`, ONNX 导出；任务：构建可复用预测管道
- **Day 13–15**：`nn.Conv2d`, `MaxPool2d`, 数据增强；任务：CIFAR-10 上训练 ResNet18（准确率 ≥90%）
- **Day 16–18**：`nn.LSTM`, `nn.Embedding`；任务：IMDB 情感分类
- **Day 19–21**：`lr_scheduler`, `torch.cuda.amp`, TensorBoard；任务：对 CNN/RNN 模型进行优化与可视化

### Phase III：综合项目与高阶技能（Day 22–30）
- **Day 22–24**：YOLOv5 目标检测；任务：自定义数据集（如车辆），mAP ≥0.7
- **Day 25–27**：DCGAN 生成对抗网络；任务：生成高质量动漫头像
- **Day 28**：TorchScript & ONNX 模型部署
- **Day 29**：`torch.profiler` 性能分析与显存瓶颈优化
- **Day 30**：整理代码至 GitHub，撰写项目文档，规划后续（强化学习、AutoML）

---
