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

## 下一步

将预训练嵌入用于 Module 12 下游分类。
