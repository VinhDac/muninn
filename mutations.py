"""Break each rule on a copy of this folder, and see a check fail.

A check nobody has seen fail proves little. Run: python mutations.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))

MUTATIONS = [
    ("trend reads tomorrow's price", "huginn.py",
     'lambda: np.sign(self.p / self.p.shift(n) - 1)', 'lambda: np.sign(self.p.shift(-1) / self.p.shift(n) - 1)'),
    ("the VIX filter reads tomorrow", "huginn.py",
     'vix = indicators["^VIX"]', 'vix = indicators["^VIX"].shift(-1)'),
    ("momentum buys the losers", "huginn.py",
     "rank = past.rank(axis=1, ascending=False)", "rank = past.rank(axis=1, ascending=True)"),
    ("rebalancing a day late", "huginn.py",
     "keep = np.arange(len(w)) % step == 0", "keep = np.arange(len(w)) % step == 1"),
    ("positions earn the same day", "muninn.py",
     "held = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])", "held = w"),
    ("costs are paid back", "muninn.py",
     "cost = self.cost * np.abs(", "cost = -self.cost * np.abs("),
    ("calendars are not rotated", "muninn.py",
     "self.shifts = rng.integers(252, n - 252, size=rotations)", "self.shifts = np.zeros(rotations, dtype=int)"),
    ("rotations of a few days", "muninn.py",
     "self.shifts = rng.integers(252, n - 252, size=rotations)", "self.shifts = rng.integers(1, 5, size=rotations)"),
    ("out of sample starts a year early", "muninn.py",
     'np.datetime64(split), side="right"))', 'np.datetime64(split), side="right")) - 250'),
    ("Sharpe annualised by calendar days", "stats.py", "DAYS = 252", "DAYS = 365"),
    ("the bar forgets e", "stats.py",
     "NORMAL.inv_cdf(1 - 1 / (n * math.e))", "NORMAL.inv_cdf(1 - 1 / n)"),
    ("deflated Sharpe ignores the sample size", "stats.py",
     "(sr - bar_daily) * math.sqrt(n - 1)", "(sr - bar_daily)"),
    ("overfitting probability flips its sign", "stats.py",
     "logits.append(math.log(omega / (1 - omega)))", "logits.append(math.log((1 - omega) / omega))"),
    ("participation ratio not squared", "stats.py",
     "participation = float(eig.sum() ** 2 / (eig ** 2).sum())", "participation = float(eig.sum() / (eig ** 2).sum())"),
    ("the panel loses the yen", "data.py", '"UUP": "fx", "FXE": "fx", "FXY": "fx",', '"UUP": "fx", "FXE": "fx",'),
]


def run(item):
    name, fname, old, new = item
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        copy = os.path.join(tmp, "study")
        shutil.copytree(HERE, copy, ignore=shutil.ignore_patterns("__pycache__", ".git", "site", "results.json"))
        path = os.path.join(copy, fname)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if text.count(old) != 1:
            return False, f"NOT APPLIED  {name} ({text.count(old)} matches)"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text.replace(old, new))
        done = subprocess.run([sys.executable, "-X", "utf8", "checks.py"], cwd=copy, capture_output=True, text=True, encoding="utf-8")
        failed = [line.split()[1].rstrip(":") for line in done.stdout.splitlines() if line.startswith("FAIL")]
        crashed = done.returncode != 0 and not failed
        if done.returncode == 0:
            return False, f"MISSED  {name}"
        return True, f"caught  {name}: " + (", ".join(failed) if failed else "checks crashed")


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, MUTATIONS))
    for _, line in results:
        print(line)
    caught = sum(ok for ok, _ in results)
    print(f"{caught} of {len(MUTATIONS)} planted faults caught")
    sys.exit(0 if caught == len(MUTATIONS) else 1)


if __name__ == "__main__":
    main()
