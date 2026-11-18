import numpy as np
import pandas as pd

from causal_toolkit_victoria import (
    calculate_ate_ci,
    calculate_ate_pvalue,
    ipw,
    doubly_robust,
)


def rct_data():
    """Randomized data with true ATE = 2.0"""
    np.random.seed(0)
    n = 2000
    t = np.random.binomial(1, 0.5, n)
    y = 2.0 * t + np.random.normal(0, 1, n)
    return pd.DataFrame({"T": t, "Y": y})


def confounded_data():
    """Observational data with true ATE = 2.0"""
    np.random.seed(42)
    n = 2000
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    prob_t = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
    t = np.random.binomial(1, prob_t, n)
    y = 2.0 * t + x1 + 0.5 * x2 + np.random.normal(0, 0.5, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "t": t, "y": y})


def test_ate_ci_contains_true_effect():
    ate, lower, upper = calculate_ate_ci(rct_data())
    assert lower < ate < upper
    assert lower < 2.0 < upper


def test_ate_pvalue_significant():
    ate, t_stat, p_value = calculate_ate_pvalue(rct_data())
    assert t_stat > 0
    assert 0 <= p_value < 0.05


def test_ipw_close_to_true_effect():
    ate = ipw(confounded_data(), "x1 + x2", "t", "y")
    assert isinstance(ate, float)
    assert abs(ate - 2.0) < 0.5


def test_doubly_robust_close_to_true_effect():
    ate = doubly_robust(confounded_data(), "x1 + x2", "t", "y")
    assert isinstance(ate, float)
    assert abs(ate - 2.0) < 0.3
