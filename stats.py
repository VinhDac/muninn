"""The statistics the ledger needs: Sharpe, the bar N tries must clear, the deflated
Sharpe, how many of the tries were really independent, and the probability of backtest
overfitting.

References: Bailey and Lopez de Prado (2014), The Deflated Sharpe Ratio; Bailey, Borwein,
Lopez de Prado and Zhu (2017), The Probability of Backtest Overfitting; Li and Ji (2005),
the effective number of correlated tests.
"""
import itertools
import math
from statistics import NormalDist

import numpy as np

EULER = 0.5772156649015329
NORMAL = NormalDist()
DAYS = 252


def sharpe(x):
    """Annualised Sharpe of daily returns, column by column."""
    x = np.asarray(x, dtype=float)
    return x.mean(axis=0) / x.std(axis=0, ddof=1) * math.sqrt(DAYS)


def expected_max(n, sd):
    """Expected largest of n independent normal draws with mean 0 and this spread."""
    if n <= 1:
        return 0.0
    return sd * ((1 - EULER) * NORMAL.inv_cdf(1 - 1 / n) + EULER * NORMAL.inv_cdf(1 - 1 / (n * math.e)))


def psr(x, bar_daily):
    """Probability that the true daily Sharpe of x beats bar_daily, allowing for skew and fat tails."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    sd = x.std(ddof=1)
    sr = x.mean() / sd
    z3 = ((x - x.mean()) / sd) ** 3
    z4 = ((x - x.mean()) / sd) ** 4
    skew, kurt = z3.mean(), z4.mean()
    denominator = 1 - skew * sr + (kurt - 1) / 4 * sr * sr
    return NORMAL.cdf((sr - bar_daily) * math.sqrt(n - 1) / math.sqrt(max(denominator, 1e-12)))


def null_sd_daily(n_days):
    """Spread of a daily Sharpe estimate when the true Sharpe is zero."""
    return 1 / math.sqrt(n_days)


def effective_trials(returns):
    """How many independent tries a set of correlated return series amounts to.

    Li and Ji: sum over eigenvalues of the correlation matrix of 1{l >= 1} + (l - floor(l)).
    Participation ratio: (sum l)^2 / sum l^2. A sign-flipped copy of a strategy adds nothing.
    """
    corr = np.corrcoef(np.asarray(returns, dtype=float), rowvar=False)
    corr = np.nan_to_num(corr)
    eig = np.clip(np.linalg.eigvalsh(corr), 0, None)
    li_ji = float(np.sum((eig >= 1).astype(float) + (eig - np.floor(eig))))
    participation = float(eig.sum() ** 2 / (eig ** 2).sum())
    return li_ji, participation


def pbo(returns, splits=16):
    """Probability of backtest overfitting by combinatorially symmetric cross-validation.

    The in-sample period is cut into `splits` blocks. For every way of choosing half of
    them, the best strategy on that half is ranked on the other half. PBO is the share of
    choices where the winner lands in the bottom half.
    """
    r = np.asarray(returns, dtype=float)
    n_days, n = r.shape
    blocks = np.array_split(np.arange(n_days), splits)
    sums = np.stack([r[b].sum(axis=0) for b in blocks])
    squares = np.stack([(r[b] ** 2).sum(axis=0) for b in blocks])
    counts = np.array([len(b) for b in blocks], dtype=float)

    def sharpe_of(mask):
        k = counts[mask].sum()
        mean = sums[mask].sum(axis=0) / k
        var = squares[mask].sum(axis=0) / k - mean ** 2
        return mean / np.sqrt(np.maximum(var, 1e-18))

    logits = []
    for chosen in itertools.combinations(range(splits), splits // 2):
        train = np.zeros(splits, dtype=bool)
        train[list(chosen)] = True
        a, b = sharpe_of(train), sharpe_of(~train)
        best = int(np.argmax(a))
        omega = (b <= b[best]).sum() / (n + 1)
        logits.append(math.log(omega / (1 - omega)))
    logits = np.array(logits)
    return float((logits <= 0).mean()), logits
