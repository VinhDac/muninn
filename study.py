"""The study: Huginn flies the whole grammar, Muninn remembers every flight.

Writes results.json, which the page and the README are built from. Also re-runs the search
with the in-sample period ending in 2014 and in 2018, and with costs doubled, to see
whether the findings depend on those choices.

Run: python study.py   (a few minutes; the Yahoo data is cached in data/)
"""
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import data
import huginn
import muninn
import stats

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results.json")
SPLIT = "2016-12-31"
VARIANTS = {"split 2014": ("2014-12-31", 1.0), "split 2018": ("2018-12-31", 1.0), "costs doubled": (SPLIT, 2.0)}
GHOSTS = 16

FAMILY_NAMES = {
    "trend": "Trend",
    "ma_cross": "Moving-average cross",
    "regime": "Trend, switched by a regime",
    "cross_section": "Cross-sectional momentum",
    "reversal": "Reversal",
    "lead_lag": "One market leads another",
    "macro": "A macro indicator times a market",
}


def r2(x):
    return round(float(x), 2)


def r3(x):
    return round(float(x), 3)


def weekly(dates, values):
    s = pd.Series(np.asarray(values, dtype=float), index=dates).resample("W-FRI").last().dropna()
    return [d.strftime("%Y-%m-%d") for d in s.index], [round(float(v), 2) for v in s.values]


def curve(net, scale_part):
    """Cumulative P&L in percent, scaled to 10% annual volatility over scale_part, not compounded."""
    sd = float(np.std(scale_part, ddof=1)) * math.sqrt(stats.DAYS)
    return np.cumsum(np.asarray(net) * (0.10 / sd)) * 100


def summary(book):
    sr_in, sr_out = np.array(book.sr_in), np.array(book.sr_out)
    best = int(np.argmax(sr_in))
    fam = book.families()
    return {
        "kept": len(sr_in),
        "best": book.names[best], "best_family": book.family_of[best],
        "best_in": r2(sr_in[best]), "best_out": r2(sr_out[best]),
        "null_mean_best": r2(book.best_null().mean()), "bar95": r2(book.bar(0.95)),
        "p_value": r2(book.p_value(sr_in[best])),
        "rank_corr_variants": r2(spearmanr(sr_in, sr_out).statistic),
        "rank_corr_family_timing": r2(spearmanr(fam["timing"], fam["real_out"]).statistic),
        "families": {f: {"timing": r2(row.timing), "out": r2(row.real_out)} for f, row in fam.iterrows()},
        "years_in": round(book.years_in, 1), "years_out": round(book.years_out, 1),
    }


def main():
    t0 = time.time()
    prices, indicators = data.panel()
    market = huginn.Market(prices, indicators)
    flights = huginn.flights()
    book = muninn.Ledger(market.r, split=SPLIT)
    others = {k: muninn.Ledger(market.r, split=s, cost=huginn.COST * c, keep_returns=False) for k, (s, c) in VARIANTS.items()}
    ledger_seconds = 0.0
    for i, spec in enumerate(flights):
        w = market.weights(spec)
        start = time.perf_counter()
        book.record(huginn.label(spec), spec[0], w, tag=i)
        ledger_seconds += time.perf_counter() - start
        for other in others.values():
            other.record(huginn.label(spec), spec[0], w, tag=i)
        if i % 1000 == 0:
            print(f"  {i} flights, {time.time() - t0:.0f}s", flush=True)

    sr_in, sr_out = np.array(book.sr_in), np.array(book.sr_out)
    n = len(sr_in)
    order = np.argsort(-sr_in)
    best = int(order[0])
    best_null = book.best_null()
    M = book.matrix()
    null_mean = book.matrix().mean(axis=1)
    fam = book.families()
    family_order = list(fam.index)

    # The winner, and the best flight of a spread of rotated calendars, as P&L curves.
    rows_all = np.arange(book.rows_in[0], len(book.r))
    w_best = market.weights(flights[book.tags[best]])
    net_best = book.net(w_best)
    dates, values = weekly(book.index[rows_all], curve(net_best[rows_all], net_best[book.rows_in]))
    picks = np.argsort(best_null)[np.linspace(3, len(best_null) - 4, GHOSTS).astype(int)]
    ghosts = []
    for k in picks:
        j = int(np.argmax(M[:, k]))
        fake = book.rotated_net(market.weights(flights[book.tags[j]]), int(k))
        gd, gv = weekly(book.index[book.rows_in], curve(fake, fake))
        ghosts.append({"name": book.names[j], "sharpe": r2(book.null[j][k]), "values": gv})

    # How many independent flights the search amounts to, three ways.
    ins = np.column_stack(book.net_in)
    li_ji, participation = stats.effective_trials(ins)
    null_sd = 1 / math.sqrt(book.years_in)
    shortcuts = {}
    for label, k in (("every flight", n), ("participation ratio", participation), ("Li and Ji", li_ji), ("rotated calendars", book.implied_tries())):
        bar = stats.expected_max(k, null_sd)
        shortcuts[label] = {"tries": round(k), "bar": r2(bar), "deflated": r3(stats.psr(ins[:, best], bar / math.sqrt(stats.DAYS)))}
    pbo, _ = stats.pbo(ins)

    universes = [(flights[t][2] or "single") for t in book.tags]
    frame = pd.DataFrame({"family": book.family_of, "universe": universes, "timing": sr_in - null_mean, "out": sr_out})
    groups = frame.groupby(["family", "universe"]).agg(n=("out", "size"), timing=("timing", "mean"), out=("out", "mean")).reset_index()

    top = order[: n // 100]
    result = {
        "built": time.strftime("%Y-%m-%d"),
        "panel": {"start": str(prices.index[0].date()), "end": str(prices.index[-1].date()), "assets": list(prices.columns),
                  "indicators": list(indicators.columns)},
        "split": SPLIT, "in_start": str(book.index[book.rows_in[0]].date()),
        "in_end": str(book.index[book.rows_in[-1]].date()), "out_start": str(book.index[book.rows_out[0]].date()),
        "years_in": round(book.years_in, 1), "years_out": round(book.years_out, 1),
        "cost_bp": round(huginn.COST * 1e4, 1), "rotations": len(book.shifts), "ledger_seconds": round(ledger_seconds),
        "flights_seen": book.seen, "flights_kept": n,
        "family_names": FAMILY_NAMES, "family_order": family_order,
        "winner": {"name": book.names[best], "family": book.family_of[best], "sharpe_in": r2(sr_in[best]),
                   "sharpe_out": r2(sr_out[best]), "dates": dates, "values": values,
                   "split_index": int(np.searchsorted(dates, SPLIT, side="right"))},
        "ghosts": ghosts,
        "null_best": [r2(x) for x in best_null],
        "null_best_names": [book.names[int(np.argmax(M[:, k]))] for k in range(len(best_null))],
        "null_mean_best": r2(best_null.mean()), "null_median_best": r2(np.median(best_null)),
        "bar95": r2(np.quantile(best_null, 0.95)), "null_max_best": r2(best_null.max()),
        "p_value": r2(book.p_value(sr_in[best])),
        "spread_real": r2(sr_in.std(ddof=1)), "spread_null": r2(M.std(axis=0, ddof=1).mean()), "spread_luck": r2(null_sd),
        "mean_real_in": r3(sr_in.mean()), "mean_null_in": r3(null_mean.mean()),
        "top": {"count": len(top), "families": {k: int(v) for k, v in pd.Series([book.family_of[j] for j in top]).value_counts().items()},
                "mean_in": r2(sr_in[top].mean()), "mean_out": r2(sr_out[top].mean())},
        "topk_out": {str(k): r2(sr_out[order[:k]].mean()) for k in (1, 10, 45, 450)},
        "families": {f: {"n": int(row.n), "real_in": r2(row.real_in), "null_in": r2(row.null_in), "timing": r2(row.timing),
                         "real_out": r2(row.real_out),
                         "share_out_positive": r2(np.mean(sr_out[np.array(book.family_of) == f] > 0))} for f, row in fam.iterrows()},
        "groups": [{"family": g.family, "universe": g.universe, "n": int(g.n), "timing": r2(g.timing), "out": r2(g.out)} for g in groups.itertuples()],
        "rank_corr": {"variants": r2(spearmanr(sr_in, sr_out).statistic),
                      "variant_timing": r2(spearmanr(sr_in - null_mean, sr_out).statistic),
                      "families": r2(spearmanr(fam["real_in"], fam["real_out"]).statistic),
                      "family_timing": r2(spearmanr(fam["timing"], fam["real_out"]).statistic),
                      "groups": r2(spearmanr(groups["timing"], groups["out"]).statistic)},
        "shortcuts": shortcuts, "pbo": r2(pbo),
        "flock": {"family": [family_order.index(f) for f in book.family_of], "sr_in": [r3(x) for x in sr_in],
                  "sr_out": [r3(x) for x in sr_out], "names": book.names},
        "robustness": {k: summary(b) for k, b in others.items()},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
    print(f"\n{n} of {book.seen} flights kept; {book.years_in:.1f} years in sample, {book.years_out:.1f} out; {time.time() - t0:.0f}s")
    print(f"winner {book.names[best]}: {sr_in[best]:.2f} in, {sr_out[best]:.2f} out")
    print(f"rotated calendars: best mean {best_null.mean():.2f}, 95th {np.quantile(best_null, 0.95):.2f}; p {book.p_value(sr_in[best]):.2f}; implied tries {book.implied_tries():.0f}")
    print("shortcuts", shortcuts, "PBO", round(pbo, 2))
    print("top 1%:", result["top"])
    print(fam.round(2).to_string())
    for k, v in result["robustness"].items():
        print(k, {x: v[x] for x in ("best", "best_in", "best_out", "p_value", "rank_corr_family_timing")})
        print("   ", v["families"])


if __name__ == "__main__":
    main()
