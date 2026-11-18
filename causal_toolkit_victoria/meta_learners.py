import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict


def s_learner_discrete(train, test, X, T, y) -> pd.DataFrame:
    """S-Learner: one model mu(X, T). CATE(x) = mu(x, 1) - mu(x, 0)."""
    features = X + [T]

    model = LGBMRegressor(verbose=-1)
    model.fit(train[features], train[y])

    test_t1 = test[X].assign(**{T: 1})[features]
    test_t0 = test[X].assign(**{T: 0})[features]

    out = test.copy()
    out["cate"] = model.predict(test_t1) - model.predict(test_t0)
    return out


def t_learner_discrete(train, test, X, T, y) -> pd.DataFrame:
    """T-Learner: separate models for control and treated. CATE(x) = mu1(x) - mu0(x)."""
    control = train[train[T] == 0]
    treated = train[train[T] == 1]

    mu0 = LGBMRegressor(verbose=-1).fit(control[X], control[y])
    mu1 = LGBMRegressor(verbose=-1).fit(treated[X], treated[y])

    out = test.copy()
    out["cate"] = mu1.predict(test[X]) - mu0.predict(test[X])
    return out


def x_learner_discrete(train, test, X, T, y) -> pd.DataFrame:
    """X-Learner with propensity weighting: CATE(x) = e(x)*tau0(x) + (1 - e(x))*tau1(x)."""
    control = train[train[T] == 0]
    treated = train[train[T] == 1]

    # Stage 1: outcome models (same as T-Learner)
    mu0 = LGBMRegressor(verbose=-1).fit(control[X], control[y])
    mu1 = LGBMRegressor(verbose=-1).fit(treated[X], treated[y])

    # Stage 2: pseudo-treatment effects
    d0 = mu1.predict(control[X]) - control[y].values   # control units
    d1 = treated[y].values - mu0.predict(treated[X])   # treated units

    tau0 = LGBMRegressor(verbose=-1).fit(control[X], d0)
    tau1 = LGBMRegressor(verbose=-1).fit(treated[X], d1)

    # Propensity score e(x)
    ps_model = LogisticRegression(penalty=None)
    ps_model.fit(train[X], train[T])
    e = ps_model.predict_proba(test[X])[:, 1]

    out = test.copy()
    out["cate"] = e * tau0.predict(test[X]) + (1 - e) * tau1.predict(test[X])
    return out


def double_ml_cate(train, test, X, T, y) -> pd.DataFrame:
    """Double ML for a continuous treatment, using cross-fitted residuals."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Cross-fitting: out-of-fold predictions of E[T|X] and E[Y|X]
    t_hat = cross_val_predict(LGBMRegressor(verbose=-1), train[X], train[T], cv=kf)
    y_hat = cross_val_predict(LGBMRegressor(verbose=-1), train[X], train[y], cv=kf)

    t_res = train[T].values - t_hat
    y_res = train[y].values - y_hat

    # Transformed outcome and weights
    y_star = y_res / t_res
    w = t_res ** 2

    # CATE model
    cate_model = LGBMRegressor(verbose=-1)
    cate_model.fit(train[X], y_star, sample_weight=w)

    out = test.copy()
    out["cate"] = cate_model.predict(test[X])
    return out


# AI Acknowledgement:
# I used Claude (Anthropic) to help draft and debug the code.
# I reviewed the code, output, checked it against the assignment instructions,
# and take responsibility for the final submission.