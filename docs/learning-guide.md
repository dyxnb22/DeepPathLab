# DeepPath Lab Learning Guide

Master reference for all 15 modules: what each teaches, how to run it, and what to complete first.

## How To Use This Guide

1. Work modules in order when possible — later NLP and vision modules assume earlier foundations.
2. For each module: read `notes.md` → run the key command below → skim `report.md`.
3. Run `python3 scripts/verify_all.py` from the repo root to sanity-check every module.

## Module Overview

| # | Module | Core Concepts | Key Run Command | Prerequisites |
|---|--------|---------------|-----------------|---------------|
| 01 | [Preliminaries & Autograd](../modules/01_preliminaries_autograd/) | computational graphs, chain rule, autograd | `python3 modules/01_preliminaries_autograd/experiments/gradient_check.py` | basic Python, calculus |
| 02 | [Linear Models](../modules/02_linear_models/) | linear regression, softmax, MSE, cross-entropy | `python3 modules/02_linear_models/from_scratch/linear_regression.py` | 01 |
| 03 | [Multilayer Perceptrons](../modules/03_mlp/) | hidden layers, activations, backprop | `python3 modules/03_mlp/from_scratch/mlp_numpy.py` | 01, 02 |
| 04 | [Convolutional Neural Networks](../modules/04_cnn/) | convolutions, pooling, LeNet | `python3 modules/04_cnn/from_scratch/conv2d.py` | 01–03 |
| 05 | [Modern CNN](../modules/05_modern_cnn/) | ResNet, skip connections, depth vs degradation | `python3 modules/05_modern_cnn/from_scratch/residual_block.py` | 04 |
| 06 | [RNN](../modules/06_rnn/) | sequence modeling, BPTT, vanishing gradients | `python3 modules/06_rnn/from_scratch/rnn.py` | 01–03 |
| 07 | [LSTM & GRU](../modules/07_lstm_gru/) | gating, long-range dependencies, copy problem | `python3 modules/07_lstm_gru/from_scratch/lstm_cell.py` | 06 |
| 08 | [Attention & Transformer](../modules/08_attention_transformer/) | self-attention, scaled dot-product, transformer block | `python3 modules/08_attention_transformer/from_scratch/attention.py` | 03, 06–07 |
| 09 | [Optimization](../modules/09_optimization/) | SGD, momentum, Adam, learning-rate sensitivity | `python3 modules/09_optimization/from_scratch/optimizers.py` | 01–03 |
| 10 | [Computer Vision Applications](../modules/10_computer_vision_applications/) | transfer learning, data augmentation | `python3 modules/10_computer_vision_applications/from_scratch/augmentation.py` | 04–05 |
| 11 | [NLP Pretraining](../modules/11_nlp_pretraining/) | skip-gram, word embeddings, tiny MLM | `python3 modules/11_nlp_pretraining/from_scratch/skipgram.py` | 01–03 |
| 12 | [NLP Applications](../modules/12_nlp_applications/) | text classification, BoW, fine-tuning intro | `python3 modules/12_nlp_applications/from_scratch/bow_classifier.py` | 11 |
| 13 | [NLP Fine-Tuning](../modules/13_nlp_fine_tuning/) | freeze vs full fine-tune, linear probe, catastrophic forgetting | `python3 modules/13_nlp_fine_tuning/experiments/freeze_vs_finetune.py` | 11, 12 |
| 14 | [Recommender Systems](../modules/14_recommender_systems/) | collaborative filtering, matrix factorization, SGD on sparse ratings | `python3 modules/14_recommender_systems/experiments/baseline_comparison.py` | 02, 09 |
| 15 | [Reinforcement Learning](../modules/15_reinforcement_learning/) | MDP, Q-learning, epsilon-greedy, TD bootstrap | `python3 modules/15_reinforcement_learning/experiments/epsilon_decay.py` | basic probability |

## Tracks

| Track | Modules | Theme |
|-------|---------|-------|
| A Foundations | 01 | autograd and gradients |
| B Deep Learning Core | 02–03, 09 | linear models, MLPs, optimizers |
| C Sequence & Transformers | 06–08 | RNN → LSTM → attention |
| D Computer Vision | 04–05, 10 | CNNs and downstream vision |
| E NLP | 11–13 | embeddings → classification → fine-tuning |
| F Rec & RL | 14–15 | matrix factorization, tabular RL |

## Per-Module Artifacts

Every module should contain:

| Artifact | Purpose |
|----------|---------|
| `notes.md` | original concept notes |
| `from_scratch/` | core mechanism implemented by hand |
| `reproduce/` | framework baseline for comparison |
| `experiments/` | ablations and diagnostics |
| `report.md` | concrete results and takeaways |

## Shared Utilities

See [lib/README.md](../lib/README.md) for `gradient_check`, `plotting`, and `synthetic_data` helpers used across modules.

## Verification

```bash
python3 scripts/verify_all.py          # all modules
python3 scripts/verify_all.py --only 13 14 15   # subset
```

## Further Reading

- [docs/d2l-mapping.md](d2l-mapping.md) — D2L chapter links per module
- [docs/roadmap.md](roadmap.md) — long-term learning path
- [TASKS.md](../TASKS.md) — module completion status
