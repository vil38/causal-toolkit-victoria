"""
Week 03: ATE Estimation with IPW and Doubly Robust Estimators

AI Acknowledgement: I used Claude (Anthropic) to help me plan the notebook
structure and check my implementation steps. I reviewed, tested, and
understand all of the code I am submitting.
"""

import numpy as np
import pandas as pd
import patsy
from sklearn.linear_model import LogisticRegression, LinearRegression

def ipw(df: pd.DataFrame, ps_formula: str, T: str, Y: str) -> float:
    """Estimate the ATE using Inverse Propensity Weighting."""
    X = np.asarray(patsy.dmatrix(ps_formula, df))
    t = df[T].to_numpy()
    y = df[Y].to_numpy()

    model = LogisticRegression(penalty=None, max_iter=1000)
    model.fit(X, t)
    ps = model.predict_proba(X)[:, 1]

    ate = np.mean((t - ps) / (ps * (1 - ps)) * y)
    return float(ate)

def doubly_robust(df: pd.DataFrame, formula: str, T: str, Y: str) -> float:
    """Estimate the ATE using the Doubly Robust estimator."""
    X = np.asarray(patsy.dmatrix(formula, df))
    t = df[T].to_numpy()
    y = df[Y].to_numpy()

    # Propensity score model
    ps_model = LogisticRegression(penalty=None, max_iter=1000)
    ps_model.fit(X, t)
    ps = ps_model.predict_proba(X)[:, 1]

    # Outcome models (control and treated)
    mu0_model = LinearRegression().fit(X[t == 0], y[t == 0])
    mu1_model = LinearRegression().fit(X[t == 1], y[t == 1])

    # Predictions for ALL observations
    mu0 = mu0_model.predict(X)
    mu1 = mu1_model.predict(X)

    # DR formula
    treated_part = np.mean(t * (y - mu1) / ps + mu1)
    control_part = np.mean((1 - t) * (y - mu0) / (1 - ps) + mu0)
    return float(treated_part - control_part)

