# 报告：Attention & Mini Transformer

## 目标

从零实现 scaled dot-product attention，构建 mini transformer，在 copy task 上验证并可视化注意力权重。

## 实现范围

- `from_scratch/attention.py` — Q/K/V attention + self-attention
- `from_scratch/transformer_block.py` — encoder block（attention + FFN + LayerNorm）
- `reproduce/mini_transformer.py` — PyTorch mini transformer on copy task
- `experiments/attention_viz.py` — 注意力热力图

## 实验结果

### Mini Transformer on Easy Copy（delay=2）

100 epoch 后 **val accuracy 89.1%**，与 Module 07 GRU（88.3%）相当，说明 attention 能有效捕获长程依赖。

### 注意力可视化

热力图显示 recall 位置（行 6–9）对早期符号位置（列 0–3）有较强权重，符合 copy task 的预期行为（未训练权重下模式较弱，训练后更明显）。

## 深度检查点

- [x] Scaled dot-product attention 从零实现
- [x] Attention 权重可视化
- [x] Mini transformer 在合成任务上可训练

## 收获

- \(\sqrt{d_k}\) 缩放防止 softmax 梯度消失
- Positional encoding 是 transformer 不可缺少的部分
- Self-attention 将 O(T) 递推路径缩短为 O(1) 直接访问

## 核心知识点回顾

- **Attention**：\(\text{softmax}(QK^T/\sqrt{d_k})V\)，一步连接所有位置对
- **缩放**：\(\sqrt{d_k}\) 防止高维点积使 softmax 饱和、梯度消失
- **Self-attention**：Q/K/V 同源，动态权重替代固定递推
- **Transformer block**：Attention + FFN，各带残差与 LayerNorm（延续 Module 05 skip 思想）
- **位置编码**：弥补 attention 的置换不变性

## 推荐复习命令

```bash
python modules/08_attention_transformer/from_scratch/attention.py
python modules/08_attention_transformer/from_scratch/transformer_block.py
python modules/08_attention_transformer/reproduce/mini_transformer.py
python modules/08_attention_transformer/experiments/attention_viz.py
```

## 下一步

Module 09 Optimization：跨模块优化器对比；Module 11 NLP Pretraining。
