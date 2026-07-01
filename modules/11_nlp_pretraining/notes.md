# 笔记：NLP 预训练

## 核心问题

Word2Vec 和 BERT 分别学到了什么类型的语言知识？

## 分布式假设

「You shall know a word by the company it keeps.」共现于相似上下文的词，语义相近。词向量 \(\mathbf{v}_w \in \mathbb{R}^d\) 将离散词映射到连续空间。

## Word2Vec / Skip-gram

给定中心词 \(w_c\)，预测上下文词 \(w_o\)（窗口大小 \(m\)）：

\[
P(w_o \mid w_c) = \frac{\exp(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c})}{\sum_{w \in \mathcal{V}} \exp(\mathbf{u}_w^\top \mathbf{v}_{w_c})}
\]

其中 \(\mathbf{v}\) 为输入嵌入，\(\mathbf{u}\) 为输出嵌入。训练目标（单对 \((w_c, w_o)\)）：

\[
\mathcal{L} = -\log P(w_o \mid w_c)
\]

**梯度直觉**：增大中心词与真实上下文词的点积，减小与所有词的归一化竞争。

**CBOW**（未实现）：用上下文词平均预测中心词，与 skip-gram 对偶。

**负采样**（未实现）：将 softmax 替换为 sigmoid 二分类，大幅加速大词表训练。

## 词向量性质

- **语义相似**：共现模式相近 → 余弦相似度高
- **类比**：\(\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}\)
- **局限**：一词一向量，无法区分多义词（如 bank）

## BERT / MLM

随机以概率 \(p_{\text{mask}}\)（通常 15%）将 token 替换为 `[MASK]`，用双向上下文预测原 token：

\[
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_i \mid x_{\backslash \mathcal{M}})
\]

\(\mathcal{M}\) 为被 mask 的位置集合。Transformer encoder 使每个位置 attend 到全句，故表示是**上下文相关**的。

## 对比

| | Word2Vec | BERT/MLM |
|---|----------|----------|
| 表示 | 静态嵌入 \(\mathbf{v}_w\) | 上下文动态 \(\mathbf{h}_i = f(x_1,\ldots,x_n)_i\) |
| 训练目标 | 局部窗口共现 | 全句 mask 预测 |
| 架构 | 浅层线性 + softmax | 深层 Transformer |
| 复杂度 | 低 | 高 |
| 多义词 | 无法区分 | 不同上下文不同向量 |

## 本模块实验

- Skip-gram 从零训练 + PCA 可视化（`tiny_corpus.txt`）
- Tiny MLM 演示 mask 预测（`the cat [MASK] on the mat` → `sits`）

## 常见陷阱

1. **语料太小**：嵌入质量严重依赖共现统计，微型语料只能演示机制
2. **忘记 softmax 数值稳定**：应减去 `logits.max()` 再 exp，避免溢出
3. **混淆输入/输出嵌入**：skip-gram 有两套矩阵 \(W_{\text{in}}, W_{\text{out}}\)；常用 \(W_{\text{in}}\) 作为词向量
4. **窗口过大**：噪声上下文增多，语义信号被稀释
5. **静态嵌入直接用于多义词任务**：如「bank」在金融/河岸语境需不同表示
6. **MLM 只在 mask 位置算 loss**：非 mask 位置不参与语言建模损失

## 自检问题

1. Skip-gram 与 CBOW 的训练目标有何对偶关系？
2. 写出 skip-gram 单对 \((w_c, w_o)\) 的 softmax 交叉熵损失。
3. 为什么 Word2Vec 能捕获类比关系？这与线性代数什么性质有关？
4. BERT 的表示为何是「动态」的？与 Word2Vec 的本质区别是什么？
5. 本模块语料极小，最近邻结果可能不稳定——你如何验证实现正确？
6. `skipgram.py` 中 `logits -= logits.max()` 起什么作用？
