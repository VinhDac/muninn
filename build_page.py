"""Build the note from results.json: every number on the page is read from it, and the
claims the text makes are checked against it before anything is written.

Run: python build_page.py   (writes site/index.html and README.md)
"""
import datetime as dt
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "https://github.com/VinhDac/muninn"
CHECKS, FAULTS = 14, 15  # checks.py, mutations.py
MINUS = "−"
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
ASSET = {"SPY": "US stocks", "EFA": "developed stocks outside the US", "EEM": "emerging-market stocks",
         "EWJ": "Japanese stocks", "TLT": "long Treasuries", "IEF": "7 to 10-year Treasuries",
         "LQD": "investment-grade credit", "HYG": "high yield", "GLD": "gold", "SLV": "silver",
         "DBC": "commodities", "USO": "oil", "UUP": "the dollar", "FXE": "the euro", "FXY": "the yen"}


def num(x, d=2):
    return (MINUS if x < 0 else "") + f"{abs(x):.{d}f}"


def signed(x, d=2):
    return (MINUS if x < 0 else "+") + f"{abs(x):.{d}f}"


def pct(x):
    return f"{round(100 * x)}%"


def day(iso):
    d = dt.date.fromisoformat(iso)
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def thousands(n):
    return f"{n:,}"


def plain(label):
    """A flight's label in words: lead_lag[lead=FXY target=SLV n=63] ... -> the yen leading silver."""
    m = re.match(r"(\w+)\[([^\]]*)\]", label)
    family, inside = m.group(1), dict(kv.split("=") for kv in m.group(2).split())
    if family == "lead_lag":
        return f"{ASSET[inside['lead']]} leading {ASSET[inside['target']]}"
    return label


def claims(r):
    """The sentences on the page, checked against the numbers before the page is written."""
    fam = r["families"]
    timing = [f for f in fam if fam[f]["timing"] >= 0.1]
    rest = [f for f in fam if fam[f]["timing"] < 0.1]
    assert r["winner"]["sharpe_in"] > 1.0 and r["winner"]["sharpe_out"] < 0.1, "the winner no longer fails"
    assert 0.1 <= r["p_value"] <= 0.5, "luck no longer beats the winner on a share of calendars"
    assert r["top"]["families"].get("lead_lag", 0) > r["top"]["count"] / 2, "lead-lag no longer fills the top"
    assert abs(fam["lead_lag"]["real_in"]) < 0.05, "lead-lag has an average edge after all"
    assert len(timing) == 4 and all(fam[f]["real_out"] > 0 for f in timing), "a family with timing lost"
    assert len(rest) == 3 and all(fam[f]["real_out"] < 0 for f in rest), "a family without timing earned"
    rob = r["robustness"]
    assert all(v["best_out"] <= 0.1 for v in rob.values()), "a winner survived in a robustness version"
    assert all(v["rank_corr_family_timing"] >= 0.7 for v in rob.values()), "family timing lost its order"
    sc = r["shortcuts"]
    assert sc["participation ratio"]["tries"] < sc["Li and Ji"]["tries"] < sc["every flight"]["tries"], "the counts changed order"
    assert r["rank_corr"]["family_timing"] >= 0.7, "family timing lost its order"
    return timing, rest


def tokens(r):
    timing, rest = claims(r)
    fam = r["families"]
    w = r["winner"]
    versions = [("To 2014", r["robustness"]["split 2014"]),
                ("To 2016, this note", {"best": w["name"], "best_in": w["sharpe_in"], "best_out": w["sharpe_out"],
                                                   "p_value": r["p_value"], "rank_corr_family_timing": r["rank_corr"]["family_timing"]}),
                ("To 2018", r["robustness"]["split 2018"]),
                ("To 2016, costs doubled", r["robustness"]["costs doubled"])]
    rows = "".join(
        f"<tr><td>{name}</td><td>{plain(v['best'])[0].upper() + plain(v['best'])[1:]}</td>"
        f"<td class=\"n\">{num(v['best_in'])} → {num(v['best_out'])}</td><td class=\"n\">{pct(v['p_value'])}</td>"
        f"<td class=\"n\">{num(v['rank_corr_family_timing'])}</td></tr>"
        for name, v in versions)
    outs = [v["best_out"] for _, v in versions]
    ps = [v["p_value"] for _, v in versions]
    rcs = [v["rank_corr_family_timing"] for _, v in versions]
    sc = r["shortcuts"]
    try_rows = "".join(f"<tr><td>{k[0].upper() + k[1:]}</td><td class=\"n\">{thousands(v['tries'])}</td><td class=\"n\">{num(v['bar'])}</td></tr>"
                       for k, v in sc.items())
    null_ll = sum(1 for n in r["null_best_names"] if n.startswith("lead_lag["))
    p_nights = sum(1 for v in r["null_best"] if v >= w["sharpe_in"])
    return {
        "MONTH": "November 2026",
        "FLIGHTS": thousands(r["flights_kept"]), "FLIGHTS_SEEN": thousands(r["flights_seen"]),
        "YEARS_IN": f"{r['years_in']:.1f}", "YEARS_OUT": f"{r['years_out']:.1f}",
        "WINNER_PRETTY": plain(w["name"]), "WINNER_PRETTY_CAP": plain(w["name"])[0].upper() + plain(w["name"])[1:],
        "WIN_IN": num(w["sharpe_in"]), "WIN_OUT": num(w["sharpe_out"]),
        "P_PCT": pct(r["p_value"]), "P_DEC": f"{r['p_value']:.2f}", "P_NIGHTS": str(p_nights),
        "TOP_LL": str(r["top"]["families"]["lead_lag"]), "TOP_N": str(r["top"]["count"]),
        "TOP_IN": num(r["top"]["mean_in"]), "TOP_OUT": num(r["top"]["mean_out"]),
        "LL_N": thousands(fam["lead_lag"]["n"]), "LL_IN": num(fam["lead_lag"]["real_in"]), "LL_OUT": num(fam["lead_lag"]["real_out"]),
        "NULL_LL": str(null_ll), "ROTATIONS": str(r["rotations"]),
        "NULL_MEAN": num(r["null_mean_best"]), "BAR95": num(r["bar95"]),
        "T_LO": signed(min(fam[f]["timing"] for f in timing)), "T_HI": signed(max(fam[f]["timing"] for f in timing)),
        "GOOD_LO": num(min(fam[f]["real_out"] for f in timing)), "GOOD_HI": num(max(fam[f]["real_out"] for f in timing)),
        "BAD_T": signed(max(fam[f]["timing"] for f in rest)),
        "BAD_LO": num(min(fam[f]["real_out"] for f in rest)), "BAD_HI": num(max(fam[f]["real_out"] for f in rest)),
        "ROB_OUT_LO": num(min(outs)), "ROB_OUT_HI": num(max(outs)), "ROB_P_LO": pct(min(ps)), "ROB_P_HI": pct(max(ps)),
        "ROB_RC_LO": num(min(rcs)), "ROB_RC_HI": num(max(rcs)), "ROB_ROWS": rows,
        "PR_TRIES": thousands(sc["participation ratio"]["tries"]), "LJ_TRIES": thousands(sc["Li and Ji"]["tries"]),
        "ROT_TRIES": thousands(sc["rotated calendars"]["tries"]), "PR_BAR": num(sc["participation ratio"]["bar"]),
        "ROT_BAR": num(sc["rotated calendars"]["bar"]), "TRY_ROWS": try_rows,
        "LEDGER_SECONDS": str(r["ledger_seconds"]),
        "ASSETS": ", ".join(r["panel"]["assets"]),
        "PANEL_START": day(r["panel"]["start"]), "PANEL_END": day(r["panel"]["end"]),
        "IN_START": day(r["in_start"]), "IN_END": day(r["in_end"]), "OUT_START": day(r["out_start"]),
        "COST": f"{r['cost_bp']:g}", "PBO": f"{r['pbo']:.2f}", "CHECKS": str(CHECKS), "FAULTS": str(FAULTS),
        "REPO": REPO, "REPO_SHORT": REPO.replace("https://", ""),
    }, p_nights


def page_data(r, p_nights):
    top = set(sorted(range(len(r["flock"]["sr_in"])), key=lambda i: -r["flock"]["sr_in"][i])[: r["top"]["count"]])
    keep = ("winner", "ghosts", "null_best", "null_best_names", "null_mean_best", "bar95", "flock", "family_order",
            "family_names", "families", "groups", "top")
    d = {k: r[k] for k in keep}
    d["top_set"] = [1 if i in top else 0 for i in range(len(r["flock"]["sr_in"]))]
    d["p_nights"] = p_nights
    return json.dumps(d, ensure_ascii=False, separators=(",", ":"))


def fill(template, values):
    out = template
    for k, v in values.items():
        out = out.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{[A-Z0-9_]+\}\}", out)
    assert not left, f"tokens left unfilled: {sorted(set(left))}"
    return out


def main():
    with open(os.path.join(HERE, "results.json"), encoding="utf-8") as f:
        r = json.load(f)
    values, p_nights = tokens(r)
    with open(os.path.join(HERE, "page", "note.tpl.html"), encoding="utf-8") as f:
        html = fill(f.read(), values).replace("/*DATA*/", page_data(r, p_nights))
    for word in (chr(0x2014),):
        assert word not in html, "an em dash on the page"
    os.makedirs(os.path.join(HERE, "site"), exist_ok=True)
    with open(os.path.join(HERE, "site", "index.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    with open(os.path.join(HERE, "page", "README.tpl.md"), encoding="utf-8") as f:
        readme = fill(f.read(), values)
    assert chr(0x2014) not in readme, "an em dash in the README"
    with open(os.path.join(HERE, "README.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(readme)
    print(f"site/index.html {len(html) // 1024} KB; README.md written; every claim holds")


if __name__ == "__main__":
    main()
