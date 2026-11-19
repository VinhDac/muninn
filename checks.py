"""Checks: each one states a rule the study depends on and fails if the rule is broken.

Run: python checks.py   (exit code 1 if any check fails; mutations.py breaks each rule on
purpose and expects a check to fail)
"""
import math
import sys

import numpy as np
import pandas as pd

import data
import huginn
import muninn
import stats

sys.stdout.reconfigure(encoding="utf-8")
CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def panel():
    if not hasattr(panel, "cache"):
        panel.cache = data.panel()
    return panel.cache


@check
def the_data_is_whole():
    """Every asset trades every day of the panel, no daily move is absurd, dates only go forward."""
    prices, indicators = panel()
    r = prices.pct_change().iloc[1:]
    assert prices.notna().all().all() and indicators.notna().all().all(), "gaps in the panel"
    assert prices.index.is_monotonic_increasing and prices.index.is_unique, "dates out of order"
    assert r.abs().max().max() < 0.5, f"a daily move of {r.abs().max().max():.0%}"
    assert len(prices.columns) == 15 and str(prices.index[0].date()) == "2007-04-11", "the universe changed"


@check
def sharpe_is_annualised_by_trading_days():
    """Alternating 0.2% and 0% for ten years: Sharpe is sqrt((n-1)/n) * sqrt(252) exactly."""
    n = 2520
    x = np.tile([0.002, 0.0], n // 2)
    assert abs(stats.sharpe(x) - math.sqrt((n - 1) / n) * math.sqrt(252)) < 1e-9


@check
def the_bar_matches_simulated_luck():
    """The expected best of N normal draws agrees with simulation within 3%, for N of 10 to 1000."""
    rng = np.random.default_rng(1)
    for n in (10, 100, 1000):
        simulated = rng.standard_normal((3000, n)).max(axis=1).mean()
        formula = stats.expected_max(n, 1.0)
        assert abs(formula / simulated - 1) < 0.03, f"N={n}: formula {formula:.3f}, simulation {simulated:.3f}"


@check
def deflated_sharpe_is_calibrated():
    """With no edge at all, the probabilistic Sharpe clears 95% about 5% of the time."""
    rng = np.random.default_rng(2)
    hits = [stats.psr(rng.standard_normal(2000) * 0.01, 0.0) > 0.95 for _ in range(2000)]
    assert 0.03 < np.mean(hits) < 0.07, f"cleared {np.mean(hits):.1%} of the time"


@check
def positions_earn_the_next_day():
    """A position decided at Tuesday's close earns Wednesday's return, net of the cost of opening it."""
    prices, _ = panel()
    r = prices.pct_change().fillna(0.0)
    book = muninn.Ledger(r, split="2016-12-31", rotations=4)
    w = np.zeros(r.shape)
    k = 1000
    w[k, 0] = 1.0
    net = book.net(w)
    assert abs(net[k + 1] - (r.iloc[k + 1, 0] - book.cost)) < 1e-12, "the position did not earn the next day"
    assert abs(net[k]) < 1e-15 and abs(net[k + 2] + book.cost) < 1e-12, "earned on the wrong day or closed for free"


@check
def signals_see_only_the_past():
    """Weights up to a date do not change when every price after it is removed."""
    prices, indicators = panel()
    cut = prices.index[3000]
    full = huginn.Market(prices, indicators)
    past = huginn.Market(prices.loc[:cut], indicators.loc[:cut])
    seen = set()
    for spec in huginn.flights():
        key = (spec[0], spec[2], spec[4])
        if key in seen:
            continue
        seen.add(key)
        a = full.weights(spec).loc[:cut]
        b = past.weights(spec)
        assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True), f"{huginn.label(spec)} looks ahead"


@check
def rotated_calendars_cut_every_forecast():
    """A flight that knows tomorrow's return is brilliant on the real calendar and worthless on rotated ones."""
    prices, _ = panel()
    r = prices.pct_change().fillna(0.0)
    book = muninn.Ledger(r, split="2016-12-31", rotations=60)
    w = pd.DataFrame(0.0, index=r.index, columns=r.columns)
    w["SPY"] = np.sign(r["SPY"].shift(-1)).fillna(0.0)
    book.record("knows tomorrow", "cheat", w)
    assert book.sr_in[0] > 10, f"real Sharpe {book.sr_in[0]:.1f}"
    assert abs(float(np.mean(book.null[0]))) < 0.3, f"rotated Sharpe {np.mean(book.null[0]):.2f}"


@check
def rotated_calendars_keep_what_is_not_timing():
    """Always long the S&P earns about the same Sharpe on rotated calendars as on the real one."""
    prices, _ = panel()
    r = prices.pct_change().fillna(0.0)
    book = muninn.Ledger(r, split="2016-12-31", rotations=200)
    w = pd.DataFrame(0.0, index=r.index, columns=r.columns)
    w["SPY"] = 1.0
    book.record("always long", "beta", w)
    real, fake = book.sr_in[0], float(np.mean(book.null[0]))
    whole = stats.sharpe(r["SPY"].to_numpy()[300:])
    assert abs(fake - whole) < 0.15, f"rotated {fake:.2f} against the whole sample's {whole:.2f} (real {real:.2f})"


@check
def costs_cost():
    """Doubling costs lowers the Sharpe of every flight that trades."""
    prices, indicators = panel()
    market = huginn.Market(prices, indicators)
    cheap = muninn.Ledger(market.r, split="2016-12-31", rotations=2)
    dear = muninn.Ledger(market.r, split="2016-12-31", rotations=2, cost=2 * huginn.COST)
    for spec in huginn.flights()[::97]:
        w = market.weights(spec)
        a, b = cheap.record("x", spec[0], w), dear.record("x", spec[0], w)
        if a is not None and b is not None:
            assert dear.sr_in[b] < cheap.sr_in[a], f"{huginn.label(spec)} gained from costs"


@check
def rebalancing_waits_its_turn():
    """Weekly and monthly flights change positions only on their rebalance days."""
    prices, indicators = panel()
    market = huginn.Market(prices, indicators)
    for spec in [s for s in huginn.flights() if s[4] != "daily"][::53]:
        w = market.weights(spec).to_numpy()
        step = huginn.REBALANCE[spec[4]]
        moved = np.flatnonzero(np.abs(np.diff(w, axis=0)).sum(axis=1) > 0) + 1
        assert np.all(moved % step == 0), f"{huginn.label(spec)} traded off its schedule"


@check
def momentum_buys_winners():
    """Cross-sectional momentum goes long the best performer and short the worst."""
    days = pd.bdate_range("2010-01-01", periods=400)
    growth = {a: 1.0 for a in huginn.ASSETS}
    growth["SPY"], growth["TLT"] = 1.002, 0.998
    prices = pd.DataFrame({a: g ** np.arange(400) for a, g in growth.items()}, index=days)
    indicators = pd.DataFrame({"^VIX": 20.0, "^TNX": 4.0, "^IRX": 2.0}, index=days)
    w = huginn.Market(prices, indicators).weights(("cross_section", {"n": 63, "k": 1}, "all", False, "daily"))
    last = w.iloc[-1]
    assert last["SPY"] > 0 and last["TLT"] < 0, "momentum bought the loser"


@check
def counting_independent_tries():
    """Five groups of twenty identical or sign-flipped flights are five independent tries.

    The participation ratio says five. Li and Ji count each group once plus the fractional
    part of its eigenvalue, so they say at least five and fewer than ten.
    """
    rng = np.random.default_rng(3)
    base = rng.standard_normal((1500, 5))
    flights = np.repeat(base, 20, axis=1) * np.tile([1, -1], 50)
    li_ji, participation = stats.effective_trials(flights)
    assert abs(participation - 5) < 0.5 and 5 <= li_ji < 10, f"Li-Ji {li_ji:.1f}, participation {participation:.1f}"


@check
def overfitting_probability_tells_luck_from_skill():
    """PBO is near one half on pure noise and near zero when one flight has a real edge."""
    rng = np.random.default_rng(4)
    noise = rng.standard_normal((1600, 40)) * 0.01
    p_noise, _ = stats.pbo(noise)
    skill = noise.copy()
    skill[:, 0] += 0.003
    p_skill, _ = stats.pbo(skill)
    assert 0.25 < p_noise < 0.75 and p_skill < 0.1, f"noise {p_noise:.2f}, skill {p_skill:.2f}"


@check
def the_sample_is_split_cleanly():
    """In-sample and out-of-sample days do not overlap, and the out-of-sample starts after the split."""
    prices, _ = panel()
    book = muninn.Ledger(prices.pct_change().fillna(0.0), split="2016-12-31", rotations=2)
    assert book.rows_in[-1] < book.rows_out[0], "the samples overlap"
    assert book.index[book.rows_out[0]] > pd.Timestamp("2016-12-31") >= book.index[book.rows_in[-1]], "the split is in the wrong place"
    assert book.shifts.min() >= 252 and book.shifts.max() <= len(prices) - 252, "a rotation is shorter than a year"


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"ok    {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"{len(CHECKS) - failed} of {len(CHECKS)} checks pass")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
