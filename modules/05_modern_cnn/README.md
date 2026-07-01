# 05 Modern CNN

Track D — 理解现代 CNN 架构设计取舍（AlexNet / VGG / ResNet）。

## Core Question

为什么 ResNet 的 skip connection 能缓解深层网络退化问题？

## Practical Project

image classification benchmark comparing classic CNN families.

## Workflow

- Read [D2L Ch.8 Modern CNNs](https://d2l.ai/chapter_convolutional-modern/index.html)
- Write original notes in [notes.md](notes.md)
- Reproduce baselines in [reproduce/](reproduce/)
- Implement core ideas in [from_scratch/](from_scratch/)
- Run ablations in [experiments/](experiments/)
- Summarize in [report.md](report.md)

## Depth Checklist

- [ ] Plain vs residual network training curves on same dataset
- [ ] Notes explain skip connection gradient highway intuition
- [ ] Architecture comparison table (params, depth, accuracy)
- [ ] Report records which design choices mattered most
