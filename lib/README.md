# DeepPath Lab Shared Library

Small, dependency-light helpers reused across modules. Import from the repo root:

```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # adjust depth as needed
sys.path.insert(0, str(REPO_ROOT))

from lib.gradient_check import check_scalar_gradients, print_gradient_report
from lib.plotting import save_loss_curve, save_metric_curves, module_output_dir
from lib.synthetic_data import spiral_data, linear_regression_data
```

## gradient_check.py

Numerical gradient verification for scalar autograd and hand-derived gradients.

**When to use:** Module 01 autograd sanity checks; any from-scratch code where you implement backward by hand.

```python
from lib.gradient_check import check_scalar_gradients, print_gradient_report

variables = {"w": 1.5, "b": 0.2}

def loss_fn(v):
    return (v["w"] * 2.0 + v["b"] - 3.0) ** 2

analytic = {"w": 2.0 * (variables["w"] * 2.0 + variables["b"] - 3.0) * 2.0,
            "b": 2.0 * (variables["w"] * 2.0 + variables["b"] - 3.0)}

results = check_scalar_gradients(variables, loss_fn, analytic)
print_gradient_report(results)  # prints PASS/FAIL per variable
```

| Function | Purpose |
|----------|---------|
| `numerical_gradient(fn, x)` | Central finite difference for scalar `fn` |
| `relative_error(analytic, numeric)` | Normalized comparison |
| `check_scalar_gradients(...)` | Batch check over named variables |
| `print_gradient_report(results)` | Human-readable report; returns all-passed bool |

## plotting.py

Save training curves to each module's `outputs/` directory (no interactive display required).

**When to use:** experiments and reproduce scripts that log loss or accuracy over epochs.

```python
from lib.plotting import module_output_dir, save_loss_curve, save_metric_curves

out = module_output_dir("03_mlp")
save_loss_curve([1.2, 0.8, 0.5], out / "train_loss.png", title="MLP Training")

save_metric_curves(
    {"sgd": [0.6, 0.75, 0.82], "adam": [0.7, 0.85, 0.9]},
    out / "optimizer_compare.png",
    title="Optimizer Comparison",
)
```

| Function | Purpose |
|----------|---------|
| `save_loss_curve(losses, path, ...)` | Single-series loss plot |
| `save_metric_curves(metrics, path, ...)` | Multi-series overlay |
| `module_output_dir(name)` | `modules/<name>/outputs/` path helper |

## synthetic_data.py

Reproducible numpy datasets for regression, classification, and nonlinear demos.

**When to use:** Modules 02–03 experiments without downloading external data.

```python
from lib.synthetic_data import linear_regression_data, multiclass_blobs, spiral_data, xor_data

X, y, true_w = linear_regression_data(n_samples=200, seed=42)
X, y = multiclass_blobs(n_per_class=80, n_classes=3)
X, y = spiral_data(n_per_class=100)   # needs MLP — not linearly separable
X, y = xor_data(n_per_quadrant=50)    # classic nonlinear test
```

| Function | Output |
|----------|--------|
| `linear_regression_data(...)` | `X, y, true_w` for MSE regression |
| `binary_classification_data(...)` | two Gaussian blobs, labels 0/1 |
| `multiclass_blobs(...)` | K-class Gaussian clusters |
| `spiral_data(...)` | interleaved spirals (MLP/CNN demos) |
| `xor_data(...)` | XOR-like quadrants |

## Conventions

- All generators accept `seed` for reproducibility.
- Plotting writes PNG files and closes figures (safe for batch runs).
- Gradient checks use `tol=1e-5` by default; tighten for ill-conditioned losses.
