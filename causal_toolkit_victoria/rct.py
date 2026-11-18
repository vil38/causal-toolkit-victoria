import numpy as np
import pandas as pd
from typing import Tuple
from scipy.stats import norm

# Helper
def _ate_and_se(data: pd.DataFrame) -> Tuple[float, float]:
    treated = data.loc[data["T"] == 1, "Y"]
    control = data.loc[data["T"] == 0, "Y"]

    ate = treated.mean() - control.mean()
    se = np.sqrt(treated.var(ddof=1) / len(treated) +
                 control.var(ddof=1) / len(control))
    return float(ate), float(se)

# Task 1
def calculate_ate_ci(data: pd.DataFrame, alpha: float = 0.05) -> Tuple[float, float, float]:
    ate, se = _ate_and_se(data)
    z = norm.ppf(1 - alpha / 2)
    return ate, float(ate - z * se), float(ate + z * se)

# Task 2
def calculate_ate_pvalue(data: pd.DataFrame) -> Tuple[float, float, float]:
    ate, se = _ate_and_se(data)
    t_stat = (ate - 0) / se
    p_value = 2 * (1 - norm.cdf(abs(t_stat)))
    return ate, float(t_stat), float(p_value)

# AI Acknowledgement
# I used Claude Sonnet 5.5 (Anthropic) to help plan the notebook structure and draft the function implementations. I reviewed and tested all code myself.