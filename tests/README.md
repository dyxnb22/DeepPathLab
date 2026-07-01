# DeepPath Lab 单元测试

针对核心 `from_scratch` 实现的回归测试，与 `scripts/verify_all.py`（smoke test）互补：

| 脚本 | 作用 |
|------|------|
| `verify_all.py` | 各模块主脚本能否在限时内跑通 |
| `run_tests.py` | 核心算法正确性（梯度、损失下降、环境逻辑） |

```bash
python3 scripts/run_tests.py
python3 -m unittest discover -s tests -p 'test_*.py' -v
```
