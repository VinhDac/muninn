# Muninn

A ledger for agent-run backtests. It remembers every flight a search makes, and flies each one again on calendars where nothing can be predicted, so the bar a winner must clear is measured rather than assumed.

The note: https://vinhdac.github.io/notes/muninn/

## What it found

A search flew 4,536 macro ideas, seven families over fifteen ETFs, in sample from 18 Jun 2008 to 30 Dec 2016 (8.5 years).

- The best, the yen leading silver, had a Sharpe of 1.20 in sample and −0.07 over the next 9.9 years.
- On 200 rotated calendars, where no signal can know the next day, the same search finds a best of 1.10 on average, and beats 1.20 on 22% of them.
- 28 of the top 45 flights came from one market leading another, a family of 1,260 flights whose average Sharpe was 0.01 in sample and −0.06 after.
- The four families whose Sharpe beat their own rotated calendars by +0.14 to +0.17 all earned after 2016 (0.06 to 0.26). The three that did not all lost (−0.32 to −0.06).
- The same holds with the split in 2014 or 2018 and with costs doubled: the best single flight fails afterwards (−0.10 to 0.08) and family timing keeps its order (rank correlation 0.71 to 0.82).

## Run it

    pip install numpy pandas scipy
    python study.py        # fetches daily data from Yahoo into data/, writes results.json
    python checks.py       # 14 checks of the rules the study depends on
    python mutations.py    # plants 15 faults on copies and expects a check to catch each
    python build_page.py   # writes site/index.html and this README from results.json

## Files

- `huginn.py`: the search, a grammar of seven families of macro signals.
- `muninn.py`: the ledger. It records flights, flies them on rotated calendars, and gives the bar and each family's timing.
- `stats.py`: Sharpe, the expected best of N tries, the deflated Sharpe ratio, the effective number of tries, the probability of backtest overfitting.
- `data.py`: daily adjusted closes from Yahoo, cached in `data/`, which is not redistributed.
- `study.py`, `checks.py`, `mutations.py`, `build_page.py`, `page/`.

## Limits

A grid of 4,728 flights stands in for an agent. There is no carry or value, which need futures curves, and ETFs stand in for futures. Costs are 3 basis points per unit of turnover.

Dac Vinh Nguyen, November 2026. Built with Claude Code as the coding agent.
