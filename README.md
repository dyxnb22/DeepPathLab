# DeepPath Lab

> 项目制深度学习学习仓库：从零实现、复现基线、实验对照、笔记与报告。

**中文入门**：[docs/getting-started.md](docs/getting-started.md) · [学习进度清单](docs/study-checklist.md) · [模块索引](modules/README.md)

> A project-based roadmap for learning modern deep learning from scratch through implementations, reproductions, experiments, visualizations, and reports.

DeepPath Lab is a long-term learning repository built around doing the work, not just reading about it. Each module is meant to become a small but real standalone project that teaches one core idea through personal notes, a reproduction baseline, a from-scratch implementation, targeted experiments, and a short report.

This repository is not a textbook mirror. It uses public resources such as Dive into Deep Learning (D2L), CS231n, and CS224N as references, then stores only original scaffolding, code, notes, experiments, and summaries created in this repo.

## Learning Workflow

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/verify_all.py   # 15/15 模块 smoke test
python3 scripts/run_tests.py      # 核心算法单元测试
```

Then read [docs/getting-started.md](docs/getting-started.md), follow each module's `README.md`, or use [docs/learning-guide.md](docs/learning-guide.md).

```text
Read reference chapter
-> write personal notes
-> reproduce baseline experiment
-> implement core idea from scratch
-> run ablation experiments
-> write final report
```

This workflow keeps the repository grounded in understanding:

- Reference material provides the reading path.
- Personal notes capture what was actually learned.
- Reproduction work checks whether the baseline can be rebuilt.
- From-scratch implementations force contact with the core mechanics.
- Ablations turn "it runs" into "I know why it behaves this way."
- Reports make the result reviewable and reusable later.

## Learning Goal

The end goal is a coherent learning path from deep learning fundamentals to practical NLP work.

That means the repository should help you:

- understand the mechanics of modern deep learning from first principles,
- build confidence by finishing one concrete project per module,
- transition from low-level neural network concepts to sequence models and transformers,
- arrive at NLP with enough implementation depth to understand embeddings, pretraining, fine-tuning, and downstream applications.

## What This Repository Produces

The original project description had a strong learning philosophy that is worth keeping: each module should become a runnable, inspectable learning artifact rather than a one-off demo. In practice, that means we prioritize:

- from-scratch implementations of the core idea,
- framework-based reproduction for comparison,
- visualizations that expose internal behavior,
- experiments with clear hypotheses,
- reports that record results, failures, and takeaways.

AI tooling can help with scaffolding, tests, refactors, and documentation, but it should not replace understanding or invent conclusions.

## Main Learning Tracks

- Track A: Foundations
- Track B: Deep Learning Core
- Track C: Sequence Models and Transformers
- Track D: Computer Vision
- Track E: NLP and Representation Learning
- Track F: Recommender Systems and Reinforcement Learning
- Track G: Modern Extensions

## Current Modules

All 15 modules are implemented under [modules](modules/):

| # | Module | Project |
|---|--------|---------|
| 01 | Preliminaries & Autograd | mini autograd engine |
| 02 | Linear Models | regression + softmax playground |
| 03 | Multilayer Perceptrons | MLP trainer with diagnostics |
| 04 | Convolutional Neural Networks | LeNet + conv from scratch |
| 05 | Modern CNN | plain vs ResNet benchmark |
| 06 | RNN | character-level generator |
| 07 | LSTM & GRU | copy problem + gated RNN |
| 08 | Attention & Transformer | mini transformer + attention viz |
| 09 | Optimization | optimizer comparison lab |
| 10 | Computer Vision Applications | transfer learning + augmentation |
| 11 | NLP Pretraining | skip-gram + tiny MLM |
| 12 | NLP Applications | sentiment classification |
| 13 | NLP Fine-Tuning | freeze vs fine-tune strategies |
| 14 | Recommender Systems | matrix factorization |
| 15 | Reinforcement Learning | gridworld Q-learning |

Each module follows the same structure:

- `notes.md` for personal notes
- `report.md` for the final summary
- `reproduce/` for baseline recreations
- `from_scratch/` for original implementations
- `experiments/` for ablations and analysis

See [TASKS.md](TASKS.md) for module status. Optional Track G extensions (diffusion, GANs) remain future work.

## Clean Reference Policy

DeepPath Lab does not copy external textbook chapters. It only links to external resources and stores my own implementations, experiments, notes, and reports.

See the supporting docs:

- [AGENTS.md](AGENTS.md)
- [TASKS.md](TASKS.md)
- [docs/getting-started.md](docs/getting-started.md) — **推荐首读**：环境、路线图、单模块流程
- [docs/study-checklist.md](docs/study-checklist.md) — 15 模块学习进度清单
- [docs/learning-guide.md](docs/learning-guide.md) — 15 模块运行命令与学习路径
- [modules/README.md](modules/README.md) — 模块索引
- [lib/README.md](lib/README.md) — 共享工具说明
- [docs/learning-sources.md](docs/learning-sources.md)
- [docs/d2l-mapping.md](docs/d2l-mapping.md)
- [docs/agent-execution.md](docs/agent-execution.md)
- [docs/module-template.md](docs/module-template.md)
- [docs/report-template.md](docs/report-template.md)

## Suggested Standard For Module Completion

Useful guidance from the previous README is preserved here in a cleaner form. A module is in good shape when it has:

1. original notes on the concept and its purpose,
2. a clearly defined standalone project outcome,
3. a working baseline reproduction,
4. a from-scratch implementation of the main mechanism,
5. at least one meaningful visualization or diagnostic,
6. experiments or ablations with recorded observations,
7. a concise report with conclusions and open questions.

That keeps the lab focused on understanding, not just API usage.
