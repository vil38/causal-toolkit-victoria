# causal-toolkit-victoria

A Python package for causal inference methods: ATE estimation for randomized experiments, propensity score methods, and meta-learners for heterogeneous treatment effects. Built for ADSP 32029 Causal Models in Data Science (University of Chicago, Autumn 2026).

## Installation

```bash
git clone https://github.com/vil38/causal-toolkit-victoria.git
cd causal-toolkit-victoria
uv venv
uv pip install -e .
```
On macOS, run `brew install libomp` first (needed by LightGBM).

Run the tests:

```bash
uv run pytest
```

## Quick start

```python
import numpy as np
import pandas as pd
from causal_toolkit_victoria import calculate_ate_ci, calculate_ate_pvalue

np.random.seed(0)
data = pd.DataFrame({"T": np.random.binomial(1, 0.5, 500)})
data["Y"] = 2 * data["T"] + np.random.normal(0, 1, 500)

ate, ci_lower, ci_upper = calculate_ate_ci(data)
ate, t_stat, p_value = calculate_ate_pvalue(data)
```

## Modules and API

### `rct.py` (Week 02): Randomized experiments

The data must have a binary treatment column `T` and an outcome column `Y`.

| Function | Returns | Description |
|---|---|---|
| `calculate_ate_ci(data, alpha=0.05)` | `(ate, ci_lower, ci_upper)` | Difference in means with a normal confidence interval |
| `calculate_ate_pvalue(data)` | `(ate, t_stat, p_value)` | Two-sided test of H0: ATE = 0 |

### `propensity.py` (Week 03): Observational data

| Function | Returns | Description |
|---|---|---|
| `ipw(df, ps_formula, T, Y)` | `float` | ATE by inverse propensity weighting (logistic propensity model) |
| `doubly_robust(df, formula, T, Y)` | `float` | ATE by the doubly robust estimator (logistic propensity + linear outcome models) |

`ps_formula` and `formula` are patsy formulas for the covariates, for example `"x1 + x2"`.

```python
from causal_toolkit_victoria import ipw, doubly_robust

ate_ipw = ipw(df, "x1 + x2", T="t", Y="y")
ate_dr = doubly_robust(df, "x1 + x2", T="t", Y="y")
```

### `meta_learners.py` (Week 04): CATE estimation

All four functions take `(train, test, X, T, y)` where `X` is a list of covariate names, and return a copy of `test` with a new `cate` column.

| Function | Description |
|---|---|
| `s_learner_discrete` | One LightGBM model on (X, T); CATE = mu(x, 1) - mu(x, 0) |
| `t_learner_discrete` | Separate models for treated and control; CATE = mu1(x) - mu0(x) |
| `x_learner_discrete` | X-learner with propensity weighting of the two stage-2 models |
| `double_ml_cate` | Double ML with 5-fold cross-fitted residuals (works for continuous treatment) |

```python
from causal_toolkit_victoria import t_learner_discrete

result = t_learner_discrete(train, test, X=["x1", "x2"], T="t", y="y")
print(result["cate"].mean())
```

## Tests

Tests live in `tests/` and use pytest. They check the return types, the `cate` column, missing values, and that estimates recover known treatment effects in simulated data.

## License

MIT
