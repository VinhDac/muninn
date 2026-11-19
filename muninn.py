"""Muninn, the memory: the ledger an agent keeps of every backtest it runs.

Each record is a flight: a name, the family of ideas it belongs to, and its daily
positions. The ledger keeps what the positions earned, and what the same positions would
have earned on calendars rotated by at least a year, where no signal can predict the next
day but each flight keeps its own persistence, its turnover and costs, and its
correlations with every other flight.

    book = Ledger(returns, split="2016-12-31")
    book.record("trend 63d, all assets", "trend", weights)   # every flight, kept or not
    book.p_value(sharpe)      # share of rotated calendars whose best flight beats this Sharpe
    book.bar(0.95)            # the best flight a search of this size finds by luck, 95th percentile
    book.families()           # each family's timing: its Sharpe minus its own rotated calendars

The bar is measured, not assumed: it grows with every flight the search adds, and it
already knows how alike the flights are.
"""
import math

import numpy as np
import pandas as pd

import stats


class Ledger:
    def __init__(self, returns, split, warmup=300, rotations=200, seed=20261119, cost=0.0003, min_active=0.2, keep_returns=True):
        self.index = returns.index
        self.assets = list(returns.columns)
        self.r = returns.to_numpy(dtype=float)
        self.cost = cost
        self.min_active = min_active
        self.keep_returns = keep_returns
        n = len(self.r)
        cut = int(np.searchsorted(returns.index.values, np.datetime64(split), side="right"))
        self.rows_in = np.arange(warmup, cut)
        self.rows_out = np.arange(cut, n)
        rng = np.random.default_rng(seed)
        self.shifts = rng.integers(252, n - 252, size=rotations)
        self.rotated = np.stack([self.r[(self.rows_in + s) % n] for s in self.shifts], axis=2)
        self.names, self.family_of, self.tags = [], [], []
        self.sr_in, self.sr_out, self.null, self.net_in = [], [], [], []
        self.seen = 0

    @property
    def years_in(self):
        return len(self.rows_in) / stats.DAYS

    @property
    def years_out(self):
        return len(self.rows_out) / stats.DAYS

    def held_and_cost(self, weights):
        """Positions decided at a close are held over the next day; costs are paid on the change."""
        w = np.nan_to_num(np.asarray(weights, dtype=float))
        held = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
        cost = self.cost * np.abs(np.diff(held, axis=0, prepend=held[:1])).sum(axis=1)
        return held, cost

    def net(self, weights):
        held, cost = self.held_and_cost(weights)
        return (held * self.r).sum(axis=1) - cost

    def rotated_net(self, weights, k):
        """In-sample returns of these positions on rotated calendar k."""
        held, cost = self.held_and_cost(weights)
        return (held[self.rows_in] * self.rotated[:, :, k]).sum(axis=1) - cost[self.rows_in]

    def record(self, name, family, weights, tag=None):
        """Remember one flight. Flights in the market on too few days are counted but not kept."""
        self.seen += 1
        held, cost = self.held_and_cost(weights)
        net = (held * self.r).sum(axis=1) - cost
        h_in = held[self.rows_in]
        if (np.abs(h_in).sum(axis=1) > 0).mean() <= self.min_active or net[self.rows_in].std() == 0:
            return None
        fake = np.einsum("ta,tak->tk", h_in, self.rotated) - cost[self.rows_in][:, None]
        self.names.append(name)
        self.family_of.append(family)
        self.tags.append(tag)
        self.sr_in.append(float(stats.sharpe(net[self.rows_in])))
        self.sr_out.append(float(stats.sharpe(net[self.rows_out])))
        self.null.append(stats.sharpe(fake))
        if self.keep_returns:
            self.net_in.append(net[self.rows_in])
        return len(self.names) - 1

    def matrix(self):
        """Null Sharpe of every kept flight on every rotated calendar: (flights, rotations)."""
        return np.array(self.null)

    def best_null(self):
        return self.matrix().max(axis=0)

    def p_value(self, sharpe):
        return float((self.best_null() >= sharpe).mean())

    def bar(self, q=0.95):
        return float(np.quantile(self.best_null(), q))

    def implied_tries(self):
        """How many independent flights would give the same expected best by luck alone."""
        target, sd = float(self.best_null().mean()), 1 / math.sqrt(self.years_in)
        lo, hi = 1.0, 1e7
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if stats.expected_max(mid, sd) < target else (lo, mid)
        return math.sqrt(lo * hi)

    def families(self):
        table = pd.DataFrame({
            "family": self.family_of,
            "real_in": self.sr_in,
            "null_in": self.matrix().mean(axis=1),
            "real_out": self.sr_out,
        })
        out = table.groupby("family").agg(n=("real_in", "size"), real_in=("real_in", "mean"),
                                          null_in=("null_in", "mean"), real_out=("real_out", "mean"))
        out["timing"] = out["real_in"] - out["null_in"]
        return out.sort_values("timing", ascending=False)
