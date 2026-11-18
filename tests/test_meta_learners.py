import numpy as np
import pandas as pd

from causal_toolkit_victoria import (
    s_learner_discrete,
    t_learner_discrete,
    x_learner_discrete,
    double_ml_cate,
)


# ---------------------------------------------------------------
# Test data generation functions
# ---------------------------------------------------------------

def simple_data():
    """Generate simple data with known treatment effect"""
    np.random.seed(42)
    n = 1000

    # Covariates
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)

    # Treatment assignment (confounded)
    prob_t = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
    t = np.random.binomial(1, prob_t, n)

    # Outcome with constant treatment effect = 2.0
    y = 2.0 * t + x1 + 0.5 * x2 + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({'x1': x1, 'x2': x2, 't': t, 'y': y})

    # Split into train/test
    train = df.iloc[:800].copy()
    test = df.iloc[800:].copy()

    return train, test


def continuous_treatment_data():
    """Generate data with continuous treatment"""
    np.random.seed(789)
    n = 1000

    # Covariates
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)

    # Continuous treatment
    t = 10 + x1 + 2 * x2 + np.random.normal(0, 1, n)

    # Outcome: linear effect of treatment
    y = t + x1 + 0.5 * x2 + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({'x1': x1, 'x2': x2, 't': t, 'y': y})

    train = df.iloc[:800].copy()
    test = df.iloc[800:].copy()

    return train, test


X = ['x1', 'x2']


# ---------------------------------------------------------------
# S-Learner tests
# ---------------------------------------------------------------

def test_s_learner_returns_dataframe():
    train, test = simple_data()
    result = s_learner_discrete(train, test, X, 't', 'y')
    assert isinstance(result, pd.DataFrame)


def test_s_learner_has_cate_column():
    train, test = simple_data()
    result = s_learner_discrete(train, test, X, 't', 'y')
    assert 'cate' in result.columns
    assert len(result) == len(test)


def test_s_learner_constant_effect():
    train, test = simple_data()
    result = s_learner_discrete(train, test, X, 't', 'y')
    true_effect = 2.0
    tolerance = 0.5
    estimated_ate = result['cate'].mean()
    assert abs(estimated_ate - true_effect) < tolerance


def test_s_learner_return_numeric_cate():
    train, test = simple_data()
    result = s_learner_discrete(train, test, X, 't', 'y')
    assert pd.api.types.is_numeric_dtype(result['cate'])


def test_s_learner_no_nan_values():
    train, test = simple_data()
    result = s_learner_discrete(train, test, X, 't', 'y')
    assert not result['cate'].isna().any()


# ---------------------------------------------------------------
# T-Learner, X-Learner, Double ML tests
# ---------------------------------------------------------------

def test_t_learner_returns_dataframe():
    train, test = simple_data()
    result = t_learner_discrete(train, test, X, 't', 'y')
    assert isinstance(result, pd.DataFrame)


def test_x_learner_returns_dataframe():
    train, test = simple_data()
    result = x_learner_discrete(train, test, X, 't', 'y')
    assert isinstance(result, pd.DataFrame)


def test_double_ml_returns_dataframe():
    train, test = simple_data()
    result = double_ml_cate(train, test, X, 't', 'y')
    assert isinstance(result, pd.DataFrame)


def test_double_ml_continuous_treatment():
    train, test = continuous_treatment_data()
    result = double_ml_cate(train, test, X, 't', 'y')
    assert isinstance(result, pd.DataFrame)
    assert 'cate' in result.columns
    assert len(result) == len(test)
    assert not result['cate'].isna().any()
    # True effect of T on y is 1.0
    assert abs(result['cate'].mean() - 1.0) < 0.5


# AI disclosure: Claude (Anthropic) was used to help draft this code.
# I tested and edited it in Jupyter against my own week04.py and take
# responsibility for this submission.
