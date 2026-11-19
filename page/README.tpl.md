# Muninn

A ledger for agent-run backtests. It remembers every flight a search makes, and flies each one again on calendars where nothing can be predicted, so the bar a winner must clear is measured rather than assumed.

The note: https://vinhdac.github.io/notes/muninn/

## What it found

A search flew {{FLIGHTS}} macro ideas, seven families over fifteen ETFs, in sample from {{IN_START}} to {{IN_END}} ({{YEARS_IN}} years).

- The best, {{WINNER_PRETTY}}, had a Sharpe of {{WIN_IN}} in sample and {{WIN_OUT}} over the next {{YEARS_OUT}} years.
- On {{ROTATIONS}} rotated calendars, where no signal can know the next day, the same search finds a best of {{NULL_MEAN}} on average, and beats {{WIN_IN}} on {{P_PCT}} of them.
- {{TOP_LL}} of the top {{TOP_N}} flights came from one market leading another, a family of {{LL_N}} flights whose average Sharpe was {{LL_IN}} in sample and {{LL_OUT}} after.
- The four families whose Sharpe beat their own rotated calendars by {{T_LO}} to {{T_HI}} all earned after 2016 ({{GOOD_LO}} to {{GOOD_HI}}). The three that did not all lost ({{BAD_LO}} to {{BAD_HI}}).
- The same holds with the split in 2014 or 2018 and with costs doubled: the best single flight fails afterwards ({{ROB_OUT_LO}} to {{ROB_OUT_HI}}) and family timing keeps its order (rank correlation {{ROB_RC_LO}} to {{ROB_RC_HI}}).

## Run it

    pip install numpy pandas scipy
    python study.py        # fetches daily data from Yahoo into data/, writes results.json
    python checks.py       # {{CHECKS}} checks of the rules the study depends on
    python mutations.py    # plants {{FAULTS}} faults on copies and expects a check to catch each
    python build_page.py   # writes site/index.html and this README from results.json

## Files

- `huginn.py`: the search, a grammar of seven families of macro signals.
- `muninn.py`: the ledger. It records flights, flies them on rotated calendars, and gives the bar and each family's timing.
- `stats.py`: Sharpe, the expected best of N tries, the deflated Sharpe ratio, the effective number of tries, the probability of backtest overfitting.
- `data.py`: daily adjusted closes from Yahoo, cached in `data/`, which is not redistributed.
- `study.py`, `checks.py`, `mutations.py`, `build_page.py`, `page/`.

## Limits

A grid of {{FLIGHTS_SEEN}} flights stands in for an agent. There is no carry or value, which need futures curves, and ETFs stand in for futures. Costs are {{COST}} basis points per unit of turnover.

Dac Vinh Nguyen, {{MONTH}}. Built with Claude Code as the coding agent.
