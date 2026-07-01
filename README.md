# AI Learning Gym

> 一个用 Vibe Coding 辅助构建的现代深度学习学习型项目。  
> 目标不是做一堆炫技 demo，而是把每个机器学习 / 深度学习概念做成可运行、可视化、可实验、可复盘的模块。

## 0. 项目定位

AI Learning Gym 是一个长期学习项目，核心目标是：

1. **用项目反推知识理解**：每学一个模型，就实现一个可运行实验模块。
2. **从零实现核心逻辑**：每个模块至少有一个 from-scratch 版本，避免只会调库。
3. **用主流框架对照验证**：再用 PyTorch / scikit-learn / Hugging Face 等实现对照版本。
4. **可视化模型内部机制**：比如梯度流、决策边界、卷积特征图、RNN hidden state、attention heatmap、diffusion denoising process。
5. **输出实验报告**：记录假设、实现、训练曲线、指标、失败案例和理解总结。
6. **让 AI Agent 辅助工程实现，而不是替代理解**：可以让 Cursor / Claude / Codex 写脚手架、测试、UI、报告模板，但核心公式、模型结构和实验结论必须由人确认。

这个项目适合用来系统学习：

- 传统机器学习
- 自动微分与反向传播
- MLP / CNN / RNN / LSTM / GRU
- Transformer / LLM 基础
- AutoEncoder / VAE / GAN / Diffusion
- Self-supervised Learning / Contrastive Learning
- GNN / Recommender Systems
- Neural ODE / Liquid Neural Network
- Reinforcement Learning
- RAG / Agent / LLM Evaluation
- 训练工程、模型部署与实验管理

---

## 1. 项目原则

### 1.1 什么样的模块才算完成？

每个模块都必须满足以下 6 个条件：

1. **Concept Note**：有一篇概念笔记，解释这个模型解决什么问题、核心公式是什么、适合什么场景。
2. **From Scratch**：不用高级封装，自己实现核心逻辑。
3. **Framework Version**：用 PyTorch / sklearn / Hugging Face 做一个对照版本。
4. **Visualization**：至少有一个可视化，能帮助理解模型内部机制。
5. **Experiment Report**：有实验报告，包括数据、参数、训练曲线、指标、失败案例。
6. **Tests**：有最小单元测试或数值校验，比如 shape 检查、梯度检查、loss 下降检查。

### 1.2 什么不算完成？

以下情况不算完成：

- 只调用一个现成 API，然后展示结果。
- 只有 Notebook，没有模块化代码。
- 只有准确率，没有错误案例分析。
- 没有测试，无法证明实现正确。
- 没有实验记录，之后无法复现。
- 只让 AI 生成代码，自己不理解核心原理。

### 1.3 AI Agent 使用边界

AI 可以做：

- 生成项目结构
- 写测试
- 写训练脚手架
- 写可视化组件
- 重构代码
- 生成实验报告模板
- 补充文档
- 检查 bug 和边界条件

AI 不应该替你做：

- 跳过 from-scratch 实现
- 直接用高级库糊弄核心逻辑
- 编造实验结论
- 把模型跑通就说“理解了”
- 用大模型总结替代你自己的复盘

---

## 2. 推荐技术栈

### 2.1 第一阶段：学习优先，工程克制

建议先用简单、稳定、容易 debug 的技术栈：

- Python 3.11+
- PyTorch
- NumPy
- scikit-learn
- pandas
- matplotlib
- pytest
- ruff
- uv 或 poetry
- Jupyter Notebook
- Streamlit 或 Gradio 作为轻量可视化界面

### 2.2 第二阶段：做成真正的学习平台

当模块多起来之后，再逐步加入：

- FastAPI：提供实验 API
- SQLite / Postgres：保存实验结果
- MLflow / Weights & Biases：实验追踪
- Next.js：更正式的前端 dashboard
- Docker：统一运行环境
- GitHub Actions：自动测试
- Hugging Face Transformers / Diffusers：后期对照实验

### 2.3 不建议一开始就做的东西

- 不要一开始上复杂微服务。
- 不要一开始做账号系统。
- 不要一开始做复杂前端。
- 不要一开始接一堆第三方 API。
- 不要一开始追求大模型训练。
- 不要一开始把它做成 SaaS。

第一目标是：**让每个知识点能被你亲手跑通、看懂、复盘。**

---

## 3. 项目目录结构

建议初始目录如下：

```text
ai-learning-gym/
  README.md
  pyproject.toml
  .gitignore
  .env.example

  src/
    aigym/
      __init__.py

      core/
        tensor/
          value.py
          ops.py
          nn.py
        training/
          trainer.py
          callbacks.py
          checkpoint.py
        metrics/
          classification.py
          regression.py
          retrieval.py
        visualization/
          plots.py
          graphs.py
          images.py
        data/
          loaders.py
          synthetic.py
        utils/
          seed.py
          device.py
          logging.py

      modules/
        m01_autograd/
        m02_svm_margin/
        m03_mlp_optimizer/
        m04_cnn_convlens/
        m05_rnn_seqlab/
        m06_lstm_gru_forecast/
        m07_transformer_attention/
        m08_autoencoder_embedding/
        m09_vae_gan_diffusion/
        m10_gnn_graphlab/
        m11_recommender_reclab/
        m12_lnn_liquid_sensor/
        m13_rl_playground/
        m14_multimodal_clip_lab/
        m15_rag_agent_eval/
        m16_mlops_serving/

  notebooks/
    m01_autograd/
    m02_svm_margin/
    m03_mlp_optimizer/
    ...

  app/
    streamlit_app.py
    pages/
      01_Autograd.py
      02_SVM_Margin.py
      03_MLP_Optimizer.py
      04_ConvLens.py
      05_SeqLab.py
      06_AttentionScope.py
      07_DiffusionLab.py
      08_RAGEval.py

  experiments/
    configs/
    runs/
    reports/

  datasets/
    raw/
    processed/
    README.md

  tests/
    test_autograd.py
    test_svm.py
    test_mlp.py
    test_cnn_shapes.py
    test_transformer_shapes.py

  docs/
    learning_path.md
    module_template.md
    experiment_report_template.md
    agent_prompts.md
```

---

## 4. 总体学习路径

这个项目按 16 个主模块推进。每个模块都对应一个现代机器学习 / 深度学习核心阶段。

| 阶段 | 模块 | 学习重点 | 项目产物 |
|---|---|---|---|
| 1 | Autograd Lab | 自动微分、计算图、链式法则 | Mini autograd engine |
| 2 | Margin Lab | 传统 ML、SVM、分类边界 | 交互式决策边界 playground |
| 3 | MLP Optimizer Lab | MLP、优化器、正则化 | 手写 MLP 训练器 |
| 4 | ConvLens | CNN、卷积、池化、视觉特征 | CNN 特征图可视化 |
| 5 | SeqLab | RNN、BPTT、hidden state | 序列建模实验室 |
| 6 | GateLab | LSTM、GRU、门控机制 | 长序列预测对比 |
| 7 | AttentionScope | Attention、Transformer | Mini Transformer + 注意力热力图 |
| 8 | Embedding Clinic | AutoEncoder、embedding、聚类 | 语义去重 / 表征学习工具 |
| 9 | GenLab | VAE、GAN、Diffusion | 生成模型实验室 |
| 10 | GraphLab | GNN、message passing | 依赖图 / 知识图谱学习 |
| 11 | RecLab | 推荐系统、召回、排序 | 个人学习内容推荐器 |
| 12 | LiquidSensor | Neural ODE、LNN、连续时间模型 | 传感器序列预测 |
| 13 | RL Playground | DQN、Policy Gradient、PPO | 小型强化学习实验室 |
| 14 | MultiModal Lab | CLIP、图文对齐、多模态检索 | 图片-文本检索系统 |
| 15 | RAGEval Agent | RAG、Agent、评测、引用 | 可验证课程问答 Agent |
| 16 | MLOps Lab | 训练工程、部署、监控 | 模型服务与实验追踪系统 |

---

## 5. Module 01：Autograd Lab

### 5.1 学习目标

理解：

- 标量计算图
- 链式法则
- 反向传播
- 拓扑排序
- 梯度累积
- 梯度检查
- 简单神经网络的训练过程

### 5.2 要实现的功能

#### From Scratch

- `Value` 类
- 基础运算：`+`、`-`、`*`、`/`、`pow`
- 激活函数：`tanh`、`relu`、`sigmoid`
- 自动构建计算图
- `backward()`
- 简单 MLP：`Neuron`、`Layer`、`MLP`
- SGD 优化器

#### Framework 对照

- 用 PyTorch 实现同样的 MLP。
- 对比 forward 输出和 backward 梯度。

#### 可视化

- 计算图节点可视化。
- 每个节点显示：
  - value
  - grad
  - op
  - children
- 展示梯度如何从 loss 反向流回参数。

### 5.3 验收标准

- `pytest tests/test_autograd.py` 通过。
- 手写 `Value` 的梯度和 PyTorch autograd 在简单表达式上误差小于 `1e-5`。
- 可以训练一个二分类 toy dataset，loss 持续下降。
- 有一篇 `experiments/reports/m01_autograd.md`。

---

## 6. Module 02：Margin Lab

### 6.1 学习目标

理解：

- 线性分类
- Logistic Regression
- SVM margin
- support vectors
- kernel trick
- C / gamma / degree 对边界的影响
- 欠拟合与过拟合

### 6.2 要实现的功能

#### From Scratch

- Logistic Regression
- Linear SVM hinge loss
- KNN
- 简单 RBF kernel SVM 的训练流程，允许先用简化版本

#### Framework 对照

- 使用 scikit-learn：
  - `LogisticRegression`
  - `SVC(kernel="linear")`
  - `SVC(kernel="rbf")`
  - `KNeighborsClassifier`

#### 可视化

- 2D 数据集散点图。
- 决策边界。
- SVM margin。
- support vectors。
- 调整超参数后实时变化。

### 6.3 推荐数据集

- linearly separable synthetic data
- moons
- circles
- blobs

### 6.4 验收标准

- 可以在同一数据集上对比 Logistic Regression、KNN、SVM。
- 可以解释为什么 RBF SVM 能处理非线性数据。
- 有一篇 `experiments/reports/m02_svm_margin.md`。

---

## 7. Module 03：MLP Optimizer Lab

### 7.1 学习目标

理解：

- MLP
- 激活函数
- loss function
- SGD
- Momentum
- RMSProp
- Adam
- weight decay
- dropout
- batch normalization
- learning rate schedule

### 7.2 要实现的功能

#### From Scratch

- MLP forward / backward
- mini-batch training loop
- SGD
- Momentum
- Adam 简化版

#### Framework 对照

- PyTorch MLP
- `torch.optim.SGD`
- `torch.optim.Adam`

#### 可视化

- loss curve
- accuracy curve
- learning rate curve
- weight distribution
- gradient norm

### 7.3 实验问题

你需要回答：

1. 为什么学习率过大会发散？
2. 为什么 Adam 通常比 vanilla SGD 更容易跑起来？
3. 为什么 dropout 能缓解过拟合？
4. 为什么 batch normalization 可以稳定训练？
5. 为什么初始化方式会影响训练？

### 7.4 验收标准

- 至少实现 3 种优化器。
- 至少完成 5 组对比实验。
- 有一篇 `experiments/reports/m03_mlp_optimizer.md`。

---

## 8. Module 04：ConvLens

### 8.1 学习目标

理解：

- 卷积核
- padding
- stride
- pooling
- feature map
- receptive field
- data augmentation
- BatchNorm
- ResNet skip connection
- transfer learning

### 8.2 要实现的功能

#### From Scratch

- 2D convolution 简化实现
- max pooling 简化实现
- LeNet 风格小型 CNN

#### Framework 对照

- PyTorch CNN
- 小型 ResNet block

#### 可视化

- 卷积核作用过程
- 每层 feature map
- activation distribution
- filter visualization
- Grad-CAM 或 saliency map

### 8.3 推荐数据集

- MNIST
- Fashion-MNIST
- CIFAR-10 小子集

### 8.4 实验问题

你需要回答：

1. 卷积为什么比全连接更适合图像？
2. pooling 到底损失了什么信息？
3. deeper CNN 一定更好吗？
4. 数据增强为什么有效？
5. 残差连接解决了什么问题？

### 8.5 验收标准

- 能训练一个 CNN 分类器。
- 能展示任意输入图片在各层的 feature map。
- 至少有一组“错误分类案例分析”。
- 有一篇 `experiments/reports/m04_convlens.md`。

---

## 9. Module 05：SeqLab

### 9.1 学习目标

理解：

- 序列建模
- RNN cell
- hidden state
- BPTT
- 梯度消失
- 梯度爆炸
- teacher forcing
- sequence-to-sequence 基础

### 9.2 要实现的功能

#### From Scratch

- 最小 RNN cell
- 字符级语言模型
- BPTT 简化版

#### Framework 对照

- PyTorch `nn.RNN`
- 字符级文本生成
- 时间序列预测

#### 可视化

- hidden state 随时间变化
- gradient norm over time
- 生成文本采样过程
- 预测误差随序列长度变化

### 9.3 推荐任务

- 字符级文本生成
- 正弦波预测
- 简单传感器序列预测
- 括号匹配 toy task

### 9.4 实验问题

你需要回答：

1. RNN 为什么能处理序列？
2. BPTT 和普通反向传播有什么区别？
3. 为什么长序列上 RNN 容易崩？
4. hidden state 里面到底存了什么？

### 9.5 验收标准

- from-scratch RNN 能在 toy dataset 上 loss 下降。
- PyTorch RNN 能生成可读的字符序列。
- 能展示梯度消失或爆炸的案例。
- 有一篇 `experiments/reports/m05_seqlab.md`。

---

## 10. Module 06：GateLab

### 10.1 学习目标

理解：

- LSTM
- GRU
- gate mechanism
- cell state
- long-term dependency
- sequence forecasting

### 10.2 要实现的功能

#### From Scratch

- LSTM cell 简化实现
- GRU cell 简化实现

#### Framework 对照

- PyTorch `nn.LSTM`
- PyTorch `nn.GRU`
- 和 vanilla RNN 对比

#### 可视化

- input gate
- forget gate
- output gate
- update gate
- reset gate
- cell state 变化

### 10.3 推荐任务

- 长序列复制任务
- 时间序列预测
- 字符级语言建模
- 多步预测

### 10.4 实验问题

你需要回答：

1. LSTM 的 forget gate 为什么关键？
2. GRU 为什么比 LSTM 更简洁？
3. RNN、LSTM、GRU 在长序列任务上差异在哪里？
4. 门控值是否能解释模型行为？

### 10.5 验收标准

- RNN / LSTM / GRU 在同一任务上有对比曲线。
- 能展示 LSTM gate values 的变化。
- 有一篇 `experiments/reports/m06_lstm_gru.md`。

---

## 11. Module 07：AttentionScope

### 11.1 学习目标

理解：

- token embedding
- positional encoding
- scaled dot-product attention
- multi-head attention
- causal mask
- encoder
- decoder
- Transformer block
- language modeling
- perplexity

### 11.2 要实现的功能

#### From Scratch

- scaled dot-product attention
- single-head self-attention
- multi-head attention 简化版
- causal mask
- position embedding
- Transformer block

#### Framework 对照

- PyTorch 实现 mini Transformer
- 可选 Hugging Face 小模型对照

#### 可视化

- attention heatmap
- token probability distribution
- position embedding 相似度
- 不同 head 关注模式

### 11.3 推荐任务

- character-level language modeling
- tiny Shakespeare 风格文本生成
- 简单问答 toy dataset
- sequence classification

### 11.4 实验问题

你需要回答：

1. attention 矩阵里的每个值代表什么？
2. multi-head attention 为什么有用？
3. causal mask 为什么是语言模型必需的？
4. position encoding 解决了什么问题？
5. Transformer 为什么能替代很多 RNN 场景？

### 11.5 验收标准

- 能训练一个 tiny Transformer。
- 能可视化某个输入序列的 attention heatmap。
- 能比较有无 positional encoding 的效果。
- 有一篇 `experiments/reports/m07_attention_scope.md`。

---

## 12. Module 08：Embedding Clinic

### 12.1 学习目标

理解：

- AutoEncoder
- latent space
- dimensionality reduction
- reconstruction loss
- embedding similarity
- clustering
- nearest neighbor retrieval

### 12.2 要实现的功能

#### From Scratch

- 小型 AutoEncoder
- latent vector 提取
- cosine similarity
- nearest neighbor search 简化版

#### Framework 对照

- PyTorch AutoEncoder
- PCA / t-SNE / UMAP 可视化
- 可选 sentence/image embedding 对照

#### 可视化

- 原图 vs 重构图
- latent space 2D projection
- 相似样本检索结果
- 聚类结果

### 12.3 推荐任务

- MNIST / Fashion-MNIST 重构
- 截图相似度检索
- 文档 embedding 聚类
- 图片近似去重

### 12.4 实验问题

你需要回答：

1. AutoEncoder 学到的是压缩还是语义？
2. embedding 相似到底意味着什么？
3. reconstruction loss 低是否等于表征好？
4. 为什么高维 embedding 需要可视化降维？

### 12.5 验收标准

- 能训练 AutoEncoder。
- 能从 latent vector 做相似样本检索。
- 能可视化 latent space。
- 有一篇 `experiments/reports/m08_embedding_clinic.md`。

---

## 13. Module 09：GenLab

### 13.1 学习目标

理解现代生成模型主线：

- VAE
- GAN
- Diffusion Model
- denoising
- noise schedule
- score matching 基础直觉
- classifier-free guidance 基础直觉

### 13.2 要实现的功能

#### VAE

- encoder
- decoder
- latent distribution
- reparameterization trick
- reconstruction loss + KL loss

#### GAN

- generator
- discriminator
- adversarial training
- mode collapse 观察

#### Diffusion

- forward noise process
- reverse denoising process
- tiny U-Net 或 MLP denoiser
- 逐步去噪可视化

### 13.3 推荐任务

- MNIST VAE
- MNIST GAN
- 2D toy distribution diffusion
- 小图像 diffusion demo

### 13.4 可视化

- latent interpolation
- generated samples
- GAN training instability
- diffusion denoising steps
- noise schedule 曲线

### 13.5 实验问题

你需要回答：

1. VAE 为什么生成结果容易模糊？
2. GAN 为什么训练不稳定？
3. Diffusion 为什么从噪声一步步去噪？
4. Diffusion 和 AutoEncoder 的区别在哪里？
5. 生成模型的评价为什么困难？

### 13.6 验收标准

- 至少实现 VAE 和 tiny diffusion。
- GAN 可以作为可选扩展，但要有失败案例分析。
- 能可视化 diffusion 逐步去噪。
- 有一篇 `experiments/reports/m09_genlab.md`。

---

## 14. Module 10：GraphLab

### 14.1 学习目标

理解：

- graph representation
- node feature
- edge feature
- adjacency matrix
- message passing
- GCN
- GraphSAGE
- GAT
- graph-level prediction
- node classification

### 14.2 要实现的功能

#### From Scratch

- adjacency matrix 聚合
- 简单 GCN layer
- message passing toy implementation

#### Framework 对照

- PyTorch Geometric 或 DGL 版本
- GCN / GAT 对比

#### 应用方向

推荐不要只做 Cora 节点分类，可以做一个更工程化的任务：

- npm / pip 依赖图分析
- GitHub repo dependency risk map
- 知识点图谱学习路径推荐
- LeetCode 题目标签图谱

### 14.3 可视化

- 图结构
- 节点 embedding
- 邻居聚合过程
- 风险传播路径

### 14.4 实验问题

你需要回答：

1. GNN 和普通 MLP 的区别是什么？
2. message passing 到底传播了什么？
3. 图太深为什么会 over-smoothing？
4. GAT 的 attention 和 Transformer attention 有什么联系？

### 14.5 验收标准

- 能在 toy graph 上训练 GCN。
- 能可视化节点 embedding。
- 能解释一个预测结果来自哪些邻居信息。
- 有一篇 `experiments/reports/m10_graphlab.md`。

---

## 15. Module 11：RecLab

### 15.1 学习目标

理解：

- 推荐系统基本流程
- explicit feedback / implicit feedback
- collaborative filtering
- matrix factorization
- two-tower model
- recall / ranking
- negative sampling
- ranking metrics

### 15.2 要实现的功能

#### Baseline

- popular recommendation
- tag-based recommendation
- item-item similarity

#### Traditional

- user-item matrix
- matrix factorization
- ALS 或 SGD 版本

#### Deep Learning

- two-tower retrieval model
- simple ranking MLP

#### 应用方向

可以做成一个真实对你有用的学习推荐器：

- 根据 LeetCode 刷题记录推荐下一题。
- 根据课程笔记推荐复习内容。
- 根据 GitHub star / 阅读记录推荐下一个学习项目。
- 和 DrillTrack 结合，推荐薄弱标签题目。

### 15.3 指标

- Precision@K
- Recall@K
- NDCG@K
- Hit Rate
- coverage
- diversity

### 15.4 实验问题

你需要回答：

1. 推荐系统为什么不是简单排序？
2. 召回和排序分别解决什么问题？
3. 为什么 implicit feedback 很难处理？
4. two-tower 为什么适合大规模召回？
5. 推荐系统怎么避免越推越窄？

### 15.5 验收标准

- 至少实现 popular baseline、item-based CF、matrix factorization。
- 至少有一个深度模型版本。
- 能在自己的学习行为数据上跑通。
- 有一篇 `experiments/reports/m11_reclab.md`。

---

## 16. Module 12：LiquidSensor

### 16.1 学习目标

理解：

- continuous-time model
- Neural ODE
- Liquid Time-Constant Network
- continuous-time RNN
- irregular time series
- dynamical system

### 16.2 要实现的功能

#### From Scratch

- 简化连续时间 RNN cell
- 简化 LTC cell
- Euler step 更新

#### Framework 对照

- LSTM
- GRU
- Temporal CNN
- Transformer Encoder
- LTC / Neural ODE 简化实现

### 16.3 推荐任务

- 合成传感器时间序列
- 非均匀采样序列预测
- 长序列外推
- 噪声环境下的轨迹预测

### 16.4 可视化

- hidden dynamics
- time constant 变化
- 不同采样间隔下的预测误差
- LSTM vs LTC 外推对比

### 16.5 实验问题

你需要回答：

1. 连续时间模型和离散 RNN 的区别是什么？
2. LNN / LTC 为什么适合某些时间序列任务？
3. 时间常数动态变化意味着什么？
4. 这类模型的优势和局限在哪里？

### 16.6 验收标准

- 能训练一个 LTC-like cell 处理 toy time series。
- 能和 LSTM / GRU 做对比。
- 能可视化动态时间常数。
- 有一篇 `experiments/reports/m12_liquid_sensor.md`。

---

## 17. Module 13：RL Playground

### 17.1 学习目标

理解：

- Markov Decision Process
- state / action / reward
- value function
- Q-learning
- DQN
- policy gradient
- actor-critic
- PPO 基础直觉
- exploration vs exploitation

### 17.2 要实现的功能

#### From Scratch

- tabular Q-learning
- epsilon-greedy
- replay buffer 简化版

#### Framework 对照

- DQN on CartPole
- PPO 可选

### 17.3 推荐任务

- GridWorld
- CartPole
- MountainCar
- 简单多臂老虎机

### 17.4 可视化

- agent trajectory
- Q value heatmap
- reward curve
- policy changes over time

### 17.5 实验问题

你需要回答：

1. RL 和监督学习有什么本质区别？
2. 为什么 reward 设计很关键？
3. DQN 为什么需要 replay buffer？
4. policy gradient 为什么方差大？
5. PPO 的 clipping 在直觉上解决什么问题？

### 17.6 验收标准

- Q-learning 能解决 GridWorld。
- DQN 能在 CartPole 上稳定提升 reward。
- 有 reward 曲线和失败案例分析。
- 有一篇 `experiments/reports/m13_rl_playground.md`。

---

## 18. Module 14：MultiModal Lab

### 18.1 学习目标

理解：

- image encoder
- text encoder
- contrastive learning
- CLIP-style training
- zero-shot classification
- image-text retrieval

### 18.2 要实现的功能

#### From Scratch

- contrastive loss
- cosine similarity matrix
- image-text matching toy task

#### Framework 对照

- 预训练 CLIP 模型推理
- 自己训练 tiny CLIP-like model

### 18.3 推荐任务

- 图片-文本检索
- zero-shot image classification
- 个人截图语义搜索
- 学习资料图片与笔记对齐

### 18.4 可视化

- image-text similarity matrix
- retrieval results
- embedding projection
- 错误匹配案例

### 18.5 实验问题

你需要回答：

1. 图文对齐到底在学习什么？
2. contrastive loss 为什么有效？
3. zero-shot 分类为什么可能成立？
4. 多模态模型为什么依赖大规模数据？

### 18.6 验收标准

- 能用预训练模型完成图文检索。
- 能训练一个 tiny CLIP-like toy model。
- 能解释 top-k retrieval 结果和错误案例。
- 有一篇 `experiments/reports/m14_multimodal_lab.md`。

---

## 19. Module 15：RAGEval Agent

### 19.1 学习目标

理解：

- document parsing
- chunking
- embedding
- vector search
- reranking
- query rewriting
- citation grounding
- hallucination
- tool calling
- agent state
- evaluation harness

### 19.2 项目定位

这不是“把 PDF 丢给大模型聊天”的套壳项目。

它的目标是做一个**可验证的课程 / 论文问答 Agent**：

- 每次回答必须有引用。
- 每次回答展示检索到的 chunks。
- 每次回答记录：
  - query
  - rewritten query
  - retrieved chunks
  - reranked chunks
  - final answer
  - citations
  - confidence
  - failure reason
- 对回答做自动评测和人工复核。

### 19.3 要实现的功能

#### RAG

- Markdown / PDF 文档导入
- chunking 策略对比
- embedding
- vector search
- reranking
- citation generation

#### Agent

- query rewrite
- retrieve
- answer
- verify
- fallback
- tool use logging

#### Evaluation

- answer correctness
- citation coverage
- retrieval recall
- hallucination rate
- abstention quality

### 19.4 推荐数据

- 你的课程 PDF
- 论文笔记
- LeetCode 解题笔记
- 项目 README 和技术文档

### 19.5 实验问题

你需要回答：

1. RAG 为什么不是“向量库 + 大模型”这么简单？
2. chunk 太大或太小分别有什么问题？
3. rerank 解决了什么？
4. 什么样的回答应该拒答？
5. Agent 的状态管理为什么重要？

### 19.6 验收标准

- 能导入一批自己的学习材料。
- 能回答问题并给出引用。
- 能展示检索日志。
- 至少有 20 条人工标注评测集。
- 有一篇 `experiments/reports/m15_rag_agent_eval.md`。

---

## 20. Module 16：MLOps Lab

### 20.1 学习目标

理解：

- experiment tracking
- config management
- checkpoint
- model registry
- inference server
- batch inference
- monitoring
- drift detection
- quantization
- ONNX export
- Docker deployment

### 20.2 要实现的功能

#### Training Engineering

- config-driven training
- checkpoint save/load
- seed control
- experiment logging
- metric tracking

#### Serving

- FastAPI inference endpoint
- batch prediction
- model versioning
- latency measurement
- simple monitoring dashboard

#### Optimization

- ONNX export
- dynamic quantization
- CPU inference benchmark
- batch size benchmark

### 20.3 推荐任务

把前面任意一个模型部署成服务：

- CNN image classifier
- Transformer text classifier
- AutoEncoder similarity search
- recommender candidate ranking

### 20.4 实验问题

你需要回答：

1. 训练代码和推理代码为什么要分离？
2. checkpoint 应该保存哪些内容？
3. 模型版本如何管理？
4. 如何判断模型上线后性能退化？
5. 量化会带来什么收益和损失？

### 20.5 验收标准

- 有一个模型可以通过 API 调用。
- 有模型版本记录。
- 有延迟和吞吐测试。
- 有 Dockerfile。
- 有一篇 `experiments/reports/m16_mlops_lab.md`。

---

## 21. 推荐推进节奏

### 21.1 如果你想稳扎稳打

建议顺序：

```text
Autograd → SVM → MLP → CNN → RNN → LSTM/GRU → Transformer → AutoEncoder → Diffusion → RAG Agent
```

### 21.2 如果你想快速看到成果

建议顺序：

```text
Autograd → CNN → Transformer → RAG Agent
```

### 21.3 如果你想偏工程作品集

建议顺序：

```text
CNN → EmbeddingClinic → RecLab → RAGEval Agent → MLOps Lab
```

### 21.4 如果你想偏研究理解

建议顺序：

```text
Autograd → SVM → MLP → RNN → LSTM/GRU → Transformer → Diffusion → LNN
```

---

## 22. 每个模块的标准文档模板

每个模块都应该有：

```text
modules/mXX_xxx/
  README.md
  scratch/
    model.py
    train.py
  framework/
    model_torch.py
    train_torch.py
  configs/
    baseline.yaml
  notebooks/
    exploration.ipynb
  tests/
    test_shapes.py
    test_numerics.py
  report.md
```

`report.md` 模板：

```md
# Module XX Report

## 1. Concept

这个模块学习什么？

## 2. Core Formula

核心公式是什么？

## 3. Implementation

from-scratch 版本如何实现？

## 4. Framework Baseline

PyTorch / sklearn 版本如何实现？

## 5. Experiments

| Experiment | Setting | Metric | Result | Notes |
|---|---|---|---|---|

## 6. Visualization

放图，解释图说明了什么。

## 7. Failure Cases

模型在哪些样本上失败？为什么？

## 8. What I Actually Learned

用自己的话总结，不要让 AI 空泛总结。

## 9. Next Steps

下一步应该改什么？
```

---

## 23. Vibe Coding 任务模板

### 23.1 新模块开发 Prompt

```text
你是这个仓库的开发助手。请实现 AI Learning Gym 的 Module XX：<模块名>。

目标：
- 实现 <核心模型/算法> 的 from-scratch 版本。
- 实现一个 PyTorch / sklearn 对照版本。
- 添加最小训练脚本。
- 添加可视化函数。
- 添加 pytest 单元测试。
- 添加 report.md 模板。

严格要求：
1. 不允许只调用高级库完成核心逻辑。
2. from-scratch 版本必须保留清晰注释，解释每一步数学含义。
3. 测试必须覆盖 shape、数值稳定性、loss 是否下降。
4. 不要改动其他模块。
5. 不要生成大文件、模型权重、缓存文件。
6. 所有新增命令写入模块 README。

交付：
- 修改文件列表
- 如何运行
- 测试结果
- 已知限制
```

### 23.2 代码审查 Prompt

```text
请审查 Module XX 的实现，重点检查：

1. 核心数学逻辑是否正确。
2. from-scratch 版本是否真的没有偷用高级封装。
3. 是否存在 shape bug。
4. 是否存在数值稳定性问题。
5. 测试是否足够证明实现正确。
6. report.md 是否有真实实验，而不是空泛描述。
7. 是否有生成文件误提交。

请输出：
- Findings table
- Evidence
- Reproduction steps
- Fix suggestions
- Priority
```

### 23.3 实验复盘 Prompt

```text
请根据 Module XX 的代码、训练日志和图表，帮我整理实验复盘。

要求：
1. 不要编造训练结果。
2. 只基于已有日志和图表总结。
3. 明确说明哪些现象符合预期，哪些不符合。
4. 对失败案例给出可能原因。
5. 最后给出下一轮实验建议。

输出到：
experiments/reports/mXX_<module>.md
```

---

## 24. 第一周开发计划

### Day 1：项目初始化

- 创建 repo。
- 初始化 Python 项目。
- 配置 ruff、pytest、pre-commit。
- 创建目录结构。
- 添加第一个 Streamlit app 空壳。
- 添加 `docs/module_template.md`。

### Day 2：Autograd `Value`

- 实现 `Value`。
- 支持基础运算。
- 支持 `backward()`。
- 添加简单表达式测试。

### Day 3：Autograd 可视化

- 输出计算图。
- 展示 value / grad。
- 添加图结构渲染。

### Day 4：Mini MLP

- 实现 `Neuron`、`Layer`、`MLP`。
- 实现 SGD。
- 在 toy data 上训练。

### Day 5：PyTorch 对照

- 用 PyTorch 实现同样 MLP。
- 对比 loss 曲线。
- 对比梯度。

### Day 6：实验报告

- 写 `m01_autograd.md`。
- 总结反向传播、梯度流、失败案例。

### Day 7：审查和重构

- 让 AI 做 code review。
- 补测试。
- 清理目录。
- 打 tag：`m01-autograd-complete`。

---

## 25. 初始 GitHub Issues

可以一开始就创建这些 issues：

```text
[Core] Initialize project structure and tooling
[Module 01] Implement scalar Value autograd engine
[Module 01] Add backward graph visualization
[Module 01] Implement tiny MLP from scratch
[Module 01] Add PyTorch baseline and gradient comparison
[Module 01] Add autograd tests and report
[Module 02] Implement 2D synthetic datasets
[Module 02] Implement logistic regression and linear SVM from scratch
[Module 02] Add decision boundary visualization
[App] Build Streamlit navigation shell
[Docs] Add module report template
```

---

## 26. 版本路线图

### v0.1：Autograd Foundation

- 项目初始化
- Autograd Lab
- MLP from scratch
- PyTorch baseline
- 计算图可视化

### v0.2：Classical ML + Optimizer

- SVM / Logistic Regression / KNN
- MLP Optimizer Lab
- 决策边界可视化
- optimizer 对比实验

### v0.3：Computer Vision

- CNN from scratch
- PyTorch CNN
- feature map visualization
- CIFAR-10 小实验

### v0.4：Sequence Modeling

- RNN
- LSTM
- GRU
- hidden state visualization
- sequence prediction

### v0.5：Transformer

- self-attention
- mini Transformer
- attention heatmap
- tiny language model

### v0.6：Representation + Generative Models

- AutoEncoder
- VAE
- tiny Diffusion
- embedding retrieval

### v0.7：Graph + Recommendation

- GNN
- recommendation baseline
- two-tower model
- learning content recommender

### v0.8：Modern AI Engineering

- RAG
- Agent eval
- citation grounding
- hallucination testing

### v0.9：MLOps

- experiment tracking
- model serving
- Docker
- model monitoring

### v1.0：Integrated Learning Platform

- Unified dashboard
- All module reports
- Reproducible experiments
- Project portfolio page

---

## 27. README 里的自我约束

这个项目不是为了快速做出“看起来很 AI”的东西，而是为了长期积累机器学习和深度学习能力。

每次开发前问自己：

1. 这个模块对应哪个知识点？
2. 我是否亲手实现了核心逻辑？
3. 我是否知道每个 tensor 的 shape？
4. 我是否知道 loss 为什么下降或不下降？
5. 我是否能解释可视化图表？
6. 我是否记录了失败案例？
7. 如果不用 AI，我是否还能讲清楚这个模块？

如果答案是否定的，这个模块就还没完成。

---

## 28. License

建议使用 MIT License。

---

## 29. Project Status

当前状态：

```text
Status: Planning
Current Milestone: v0.1 Autograd Foundation
Current Module: Module 01 - Autograd Lab
```

下一步：

```bash
# 1. 创建项目
mkdir ai-learning-gym
cd ai-learning-gym

# 2. 初始化 Git
git init

# 3. 创建 README.md
# 将本文档保存为 README.md

# 4. 初始化 Python 项目
uv init

# 5. 添加依赖
uv add numpy pandas matplotlib scikit-learn torch pytest ruff jupyter streamlit

# 6. 运行测试
uv run pytest
```

---

## 30. 最小启动目标

第一阶段不要想太多，只完成一件事：

> 实现一个可以训练 toy MLP 的 Autograd Lab，并把计算图和梯度流可视化出来。

当这个模块完成时，你应该能用自己的话讲清楚：

- forward pass 做了什么
- loss 是怎么来的
- backward pass 怎么沿计算图传播
- 每个参数的梯度代表什么
- SGD 为什么能让 loss 下降
- PyTorch autograd 和你的实现本质上有什么相同点

这就是 AI Learning Gym 的第一个里程碑。
