# 报告：NLP Pretraining

## 目标

从零训练 skip-gram 词向量，可视化嵌入空间，演示 tiny MLM。

## 实验结果

### Skip-gram 最近邻

小语料上可学到部分语义关联：
- `king` → `queen`, `woman`
- `queen` → `king`, `semantic`
- `embeddings` → `language`, `word`

### Tiny MLM

150 epoch 后 loss 降至 0.73。Mask 预测 demo：`the cat [MASK] on the mat` → `sits`（正确）。

## 深度检查点

- [x] Skip-gram 从零实现
- [x] 嵌入 PCA 可视化
- [x] Tiny MLM 实验

## 收获

- 静态嵌入捕获共现统计；上下文嵌入（MLM）捕获双向语义
- 语料规模决定嵌入质量，本模块仅作机制演示

## 核心知识点回顾

1. **Skip-gram**：中心词预测上下文，学到静态词向量 \(\mathbf{v}_w\)
2. **Softmax 目标**：最大化共现词对概率；梯度增大中心-上下文点积
3. **词向量性质**：语义近邻、线性类比（king - man + woman ≈ queen）
4. **MLM**：双向上下文预测被 mask 的 token，表示随句子变化
5. **静态 vs 动态**：Word2Vec 一词一向量；BERT 需经模型前向得到上下文向量

## 推荐复习命令

```bash
# 训练 skip-gram 并查看最近邻
python modules/11_nlp_pretraining/from_scratch/skipgram.py

# PCA 可视化嵌入空间
python modules/11_nlp_pretraining/experiments/embedding_viz.py

# 微型 MLM 训练与 mask 预测
python modules/11_nlp_pretraining/reproduce/tiny_mlm.py
```

复习时对照 `notes.md` 自检问题，尝试不看代码推导 skip-gram 的 softmax 梯度。

## 下一步

将预训练嵌入用于 Module 12 下游分类；进一步学习见 Module 13 NLP fine-tuning。
