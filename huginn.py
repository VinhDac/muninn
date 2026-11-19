"""Huginn, the thought: a grammar of global macro signals, the variants an agent proposes when told to find alpha.

Seven families (trend, moving-average crosses, reversal, cross-sectional momentum, trend
switched on and off by a market regime, one market leading another, a macro indicator
timing a market), each over several look-backs, universes, with or without volatility
targeting, rebalanced daily, weekly or monthly. Every spec becomes a daily weight matrix;
run() turns weights into returns net of costs.
"""
import itertools

import numpy as np
import pandas as pd

from data import ASSETS

TARGET_VOL = 0.10
LEVERAGE_CAP = 3.0
COST = 0.0003  # 3 basis points per unit of turnover
REBALANCE = {"daily": 1, "weekly": 5, "monthly": 21}

UNIVERSES = {
    "all": list(ASSETS),
    "equities": [a for a, k in ASSETS.items() if k == "equity"],
    "bonds": [a for a, k in ASSETS.items() if k == "bond"],
    "commodities": [a for a, k in ASSETS.items() if k == "commodity"],
    "fx": [a for a, k in ASSETS.items() if k == "fx"],
    "risk": ["SPY", "EFA", "EEM", "EWJ", "HYG", "DBC", "USO", "SLV"],
    "defensive": ["TLT", "IEF", "GLD", "FXY"],
    "ex_oil": [a for a in ASSETS if a != "USO"],
}
FILTERS = ("calm", "stress", "steep", "flat", "usd_up", "usd_down")
FEATURES = ("slope_z", "slope_change", "vix_z", "vix_change", "rate_change")


def zscore(s, n=252):
    return (s - s.rolling(n).mean()) / s.rolling(n).std()


class Market:
    def __init__(self, prices, indicators):
        self.p = prices
        self.r = prices.pct_change().fillna(0.0)
        self.vol = self.r.rolling(63).std() * np.sqrt(252)
        slope = indicators["^TNX"] - indicators["^IRX"]
        vix = indicators["^VIX"]
        self.feature = {
            "slope_z": zscore(slope),
            "slope_change": slope - slope.shift(63),
            "vix_z": zscore(vix),
            "vix_change": vix - vix.shift(21),
            "rate_change": indicators["^TNX"] - indicators["^TNX"].shift(63),
        }
        usd = prices["UUP"] / prices["UUP"].shift(63) - 1
        self.filter = {
            "calm": vix < vix.rolling(252).median(),
            "stress": vix >= vix.rolling(252).median(),
            "steep": slope > 0,
            "flat": slope <= 0,
            "usd_up": usd > 0,
            "usd_down": usd <= 0,
        }
        self._memo = {}

    def memo(self, key, make):
        if key not in self._memo:
            self._memo[key] = make()
        return self._memo[key]

    def trend(self, n):
        return self.memo(("trend", n), lambda: np.sign(self.p / self.p.shift(n) - 1))

    def average(self, n):
        return self.memo(("ma", n), lambda: self.p.rolling(n).mean())

    def weights(self, spec):
        family, params, universe, scaled, rebalance = spec
        names = UNIVERSES[universe] if universe else None
        if family == "trend":
            signal = self.trend(params["n"])
        elif family == "ma_cross":
            signal = np.sign(self.average(params["fast"]) - self.average(params["slow"]))
        elif family == "reversal":
            n, z = params["n"], params["z"]
            dev = (self.p - self.average(n)) / self.p.rolling(n).std()
            signal = -np.sign(dev).where(dev.abs() > z, 0.0)
        elif family == "cross_section":
            n, k = params["n"], params["k"]
            past = (self.p[names] / self.p[names].shift(n) - 1)
            rank = past.rank(axis=1, ascending=False)
            signal = pd.DataFrame(0.0, index=self.p.index, columns=self.p.columns)
            signal[names] = (rank <= k).astype(float) - (rank > len(names) - k).astype(float)
            signal[names] = signal[names].where(past.notna().all(axis=1), np.nan, axis=0)
        elif family == "regime":
            on = self.filter[params["filter"]].astype(float)
            signal = self.trend(params["n"]).mul(on, axis=0)
        elif family == "lead_lag":
            lead = np.sign(self.p[params["lead"]] / self.p[params["lead"]].shift(params["n"]) - 1)
            signal = pd.DataFrame(0.0, index=self.p.index, columns=self.p.columns)
            signal[params["target"]] = lead
            names = [params["target"]]
        elif family == "macro":
            signal = pd.DataFrame(0.0, index=self.p.index, columns=self.p.columns)
            signal[params["target"]] = params["side"] * np.sign(self.feature[params["feature"]])
            names = [params["target"]]
        else:
            raise ValueError(family)
        w = pd.DataFrame(0.0, index=self.p.index, columns=self.p.columns)
        part = signal[names]
        if scaled:
            part = part * (TARGET_VOL / self.vol[names]).clip(upper=LEVERAGE_CAP)
        w[names] = part / len(names)
        step = REBALANCE[rebalance]
        if step > 1:
            keep = np.arange(len(w)) % step == 0
            w = w.where(pd.Series(keep, index=w.index), np.nan, axis=0).ffill()
        return w.fillna(0.0)

    def run(self, w):
        held = w.shift(1).fillna(0.0)
        gross = (held * self.r).sum(axis=1)
        turnover = held.diff().abs().sum(axis=1).fillna(0.0)
        return gross - COST * turnover


def label(spec):
    family, params, universe, scaled, rebalance = spec
    inside = " ".join(f"{k}={v}" for k, v in params.items())
    return f"{family}[{inside}] {universe or ''} {'volT' if scaled else 'raw'} {rebalance}".replace("  ", " ")


def flights():
    out = []
    variants = list(itertools.product((True, False), REBALANCE))
    for n, u in itertools.product((10, 21, 42, 63, 126, 189, 252), UNIVERSES):
        out += [("trend", {"n": n}, u, s, r) for s, r in variants]
    for (f, s_), u in itertools.product(((5, 50), (10, 50), (10, 100), (20, 100), (20, 200), (50, 200)), UNIVERSES):
        out += [("ma_cross", {"fast": f, "slow": s_}, u, s, r) for s, r in variants]
    for n, z, u in itertools.product((3, 5, 10, 21), (0.0, 1.0), UNIVERSES):
        out += [("reversal", {"n": n, "z": z}, u, s, r) for s, r in variants]
    for n, u in itertools.product((21, 63, 126, 252), UNIVERSES):
        size = len(UNIVERSES[u])
        if size < 4:
            continue
        for k in range(1, min(3, size // 2) + 1):
            out += [("cross_section", {"n": n, "k": k}, u, s, r) for s, r in variants]
    for f, n, u in itertools.product(FILTERS, (21, 63, 126, 252), UNIVERSES):
        out += [("regime", {"filter": f, "n": n}, u, s, r) for s, r in variants]
    for lead, target, n in itertools.product(ASSETS, ASSETS, (5, 21, 63)):
        if lead != target:
            out += [("lead_lag", {"lead": lead, "target": target, "n": n}, None, True, r) for r in ("weekly", "monthly")]
    for f, target, side in itertools.product(FEATURES, ASSETS, (1, -1)):
        out += [("macro", {"feature": f, "target": target, "side": side}, None, s, r) for s, r in variants]
    return out
