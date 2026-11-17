[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/hyrenk722/gta6-twitch-ccu-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/hyrenk722/gta6-twitch-ccu-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# Midnight Unlock — GTA VI Launch-Day Twitch CCU Forecaster

Entry for the make.fun competition **gta6-twitch-ccu-forecast-01**:
"Forecast GTA VI's launch-day Twitch viewers hour by hour".

## What it does

`main.py` reads one case file per run — a JSON record of a game category's
audience history up to the end of hour 1 of its release day, plus a set of
comparable past launches — and forecasts the Twitch category audience
(CCU) for hours 2–24 of release day. For each hour it writes five
quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95). Scoring is mean pinball loss
on the log scale, so the quantiles are calibrated for that loss, not for
a point forecast.

## Method

Each comparable launch is summarised by the *shape* of its release day
relative to its own first hour: `lr[h] = ln(y_h) − ln(y_1)`. The target's
forecast is the distribution of those shapes pooled from the launches
most similar to it, scaled back by the target's own hour-1 audience.

Similarity is a product of Gaussian kernels over descriptors computed
identically for the target and for every comparable:

- **Audience scale** — log10 of the first-hour audience.
- **Pre-release peak** — log10 of the busiest hour in the 72 h before
  release day.
- **Ramp** — log10 of (first-hour audience / mean of the last 6 hours
  before release day): how hard the audience was still climbing into
  the launch.
- **Unlock pattern** — early access, midnight-local rolling, or one
  simultaneous global unlock, classified from the case notes and each
  comparable's unlock text; launches whose audience was partly spent
  before release day have different shapes.

For each hour, two quantile estimates of the pooled shapes — a weighted
Gaussian fit and a weighted empirical quantile curve — are blended in
log space, re-centred on the blended median, and widened or narrowed by
one calibration factor fitted by leave-one-out on the practice
comparables. Small launches (first hour under 50k) use their own
parameters (a tighter pre-release-peak kernel, Gaussian fit only),
averaged over two neighbouring peak widths as a small ensemble.

Two safety properties are built in:

- **ESS-collapse guard** — when the multi-dimension kernel concentrates
  an hour's pool onto one or two launches (tiny effective sample size),
  the estimate is stabilised by admixing the broader scale-only kernel
  weights, just enough to restore a minimum effective sample size.
- **Persistence fallback** — any hour whose pool is effectively empty,
  and any unexpected error anywhere, degrades to a persistence forecast
  (first hour held flat, spread growing like a random walk). The
  program never crashes and always writes a valid forecast file.

Validation on the practice set (leave-one-out over 37 comparables):
mean pinball loss 0.3212 versus 0.9297 for the persistence baseline;
0.1896 on the large-launch slice (first hour ≥ 50k) that GTA VI will
fall into. The final parameters also survived a nested holdout that
re-selected them out-of-sample.

## Running

Standard library only; no dependencies to install.

```sh
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=/path/to/output python3 main.py
```

- Input: every `*.in` file in `MAKEFUN_INPUT_DIR` (default `/input`),
  one JSON case per file.
- Output: one `<case>.out` JSON file per case in `MAKEFUN_OUTPUT_DIR`
  (default `/output`), with the five quantiles for each hour 2–24.
- The program is deterministic: the same input always produces
  byte-identical output.

## Repository layout

- `main.py` — the forecaster (the entry program).
- `LICENSE` — MIT license.
- `.makefun/owner.json` — entrant claim file naming the payout wallet
  for this entry.
