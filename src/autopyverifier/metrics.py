from __future__ import annotations

import math
from typing import Tuple


def safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def lower_confidence_bound(mean: float, den: int, delta: float = 0.05) -> float:
    if den <= 0:
        return 0.0
    radius = math.sqrt(math.log(2.0 / delta) / (2.0 * den))
    return max(0.0, mean - radius)


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def macro_f1(tp: int, tn: int, fp: int, fn: int) -> float:
    p_pos = safe_div(tp, tp + fp)   
    r_pos = safe_div(tp, tp + fn)

    p_neg = safe_div(tn, tn + fn)   
    r_neg = safe_div(tn, tn + fp)

    f1_pos = f1(p_pos, r_pos)
    f1_neg = f1(p_neg, r_neg)

    return 0.5 * (f1_pos + f1_neg), p_pos, p_neg


def compute_metrics(
    *,
    tp: int,
    tn: int,
    accept: int,
    reject: int,
    n: int,
    beta_pp: float,
    beta_np: float,
    set_size: int,
    delta: float = 0.05,
) -> Tuple[float, float, float, float, bool, float, float, float]:


    fp = accept - tp
    fn = reject - tn

    cov_pos = safe_div(accept, n) 
    cov_neg = safe_div(reject, n)

    score, pp, npv = macro_f1(tp, tn, fp, fn)

    lcb_pp = lower_confidence_bound(pp, accept, delta)
    lcb_np = lower_confidence_bound(npv, reject, delta)

    feasible = (
        lcb_pp >= beta_pp and lcb_np >= beta_np)

    return pp, npv, cov_pos, cov_neg, feasible, score, lcb_pp, lcb_np
