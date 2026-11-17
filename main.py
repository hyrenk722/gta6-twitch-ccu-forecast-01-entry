"""Launch-day Twitch audience forecaster -- rebuild v2.

Competition: gta6-twitch-ccu-forecast-01 (make.fun). The program reads one
case file per run from the input directory and writes five quantiles
(0.05/0.25/0.5/0.75/0.95) of the Twitch category audience for each hour
2..24 of the release day.

Approach
--------
Every past launch in the case's `comparables` is summarised by the shape
of its release day relative to its own first hour: for hour h the value
x = ln(viewers_h) - ln(viewers_1). A new launch is forecast by pooling
those shapes from the launches most similar to it, where similarity is a
product of Gaussian kernels over launch descriptors computed identically
for the target and for each comparable:

  * audience scale      -- log10 of the first-hour audience;
  * pre-release peak    -- log10 of the busiest hour in the 72 h before
                           the release day;
  * ramp                -- log10 of (first-hour audience / mean of the
                           last 6 hours before the release day): how
                           hard the audience was still climbing into
                           the launch.
(Two further descriptors -- a front-loading ratio and viewers per
channel -- are computed by the same machinery but their kernels are
disabled in the shipped parameters: leave-one-out showed they hurt.)

For each hour, the pooled shapes give two quantile estimates of x -- a
weighted Gaussian fit and a weighted empirical quantile curve -- which
are blended in log space, re-centred on the blended median, and widened
or narrowed by one calibration factor fitted by leave-one-out on the
practice comparables. The case's own first-hour print turns the ratios
back into viewers.

One guard against a failure the leave-one-out runs exposed: when the
multi-dimension kernel concentrates an hour's pool onto one or two
launches (a tiny effective sample size), the estimate for that hour is
stabilised by admixing the broader scale-only kernel weights, just
enough to restore a minimum effective sample size.

Safety: any hour whose pool is effectively empty, and any unexpected
error anywhere, degrades to a persistence forecast (first hour held
flat, spread growing like a random walk) -- the house baseline's shape.
The program is deterministic, standard library only, and always writes
a valid forecast file for every case file it finds.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_LEVELS = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)
FORECAST_HOURS = tuple(range(2, 25))

# Model parameters. The defaults are the values selected by leave-one-out
# over the practice comparables (see REBUILD-REPORT.md); the tuning script
# overrides them through forecast_case's `params` argument.
PARAMS = {
    "bw_scale": 0.5,      # kernel width on log10(first-hour audience)
    "bw_peak": 0.6,       # kernel width on log10(pre-72h peak); None = off
    "bw_front": None,     # kernel width on log10(anchor / pre-peak)
    "bw_ramp": 0.3,       # kernel width on log10(anchor / trailing mean)
    "trail_hours": 6,     # window of the trailing mean behind the ramp
    "bw_vpc": None,       # kernel width on log10(viewers per channel, h1)
    "empirical_share": 0.5,   # blend weight of the empirical quantiles
    "spread": 0.95,       # calibration multiplier on the log-spread
    "ess_target": 2.0,    # stabilise an hour's pool up to this eff. size
    "ess_mix_max": 0.8,   # cap on the scale-only weight admixture
    "ess_ref": 6.0,       # below this eff. size an hour's spread widens
    "ess_widen": 0.0,     # strength of that widening (0 = off)
    "min_anchor": 100.0,  # a comparable needs this hour-1 audience to train
    "sd_min": 0.05,       # floor on the fitted log-spread of an hour
    "base_sigma": 0.15,   # persistence fallback spread per sqrt(hour - 1)
    "cap": 1e9,
}


def _epoch(text):
    """Epoch seconds for an ISO-8601 string, else None."""
    if not isinstance(text, str):
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def _clean(value):
    """Round to 2 decimals (half up); integral values become ints.
    Non-finite values are clamped into the valid output range first."""
    if isinstance(value, float) and math.isnan(value):
        value = 0.01
    elif value == math.inf:
        value = 1e9
    elif value == -math.inf:
        value = 0.01
    scaled = math.floor(value * 100.0 + 0.5) / 100.0
    return int(scaled) if float(scaled).is_integer() else scaled


def _finite(value):
    """float(value) when it is a finite number, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _hourly_curve(day_start_epoch, points):
    """Release-day curve: hour (1..24) -> mean viewers of that hour."""
    buckets = {}
    for point in points or []:
        if not isinstance(point, dict):
            continue
        start = _epoch(point.get("from"))
        viewers = _finite(point.get("viewers"))
        if start is None or viewers is None:
            continue
        offset = start - day_start_epoch
        if 0.0 <= offset < 86400.0:
            buckets.setdefault(int(offset // 3600) + 1, []).append(viewers)
    return {h: mean for h, vals in buckets.items()
            if math.isfinite(mean := sum(vals) / len(vals))}


def _peak_before(points, key, lo, hi):
    """Largest viewers among points timestamped in [lo, hi), else None."""
    best = None
    for point in points or []:
        if not isinstance(point, dict):
            continue
        start = _epoch(point.get(key))
        viewers = _finite(point.get("viewers"))
        if start is None or viewers is None:
            continue
        if lo <= start < hi and (best is None or viewers > best):
            best = viewers
    return best


def _descriptors(anchor, pre_peak, trail_mean=None, channels=None):
    """The kernel coordinates of a launch (target or comparable)."""
    desc = {"scale": math.log10(anchor)}
    if pre_peak and pre_peak > 0:
        desc["peak"] = math.log10(pre_peak)
        desc["front"] = math.log10(anchor / pre_peak)
    if trail_mean and trail_mean > 0:
        desc["ramp"] = math.log10(anchor / trail_mean)
    if channels and channels > 0:
        desc["vpc"] = math.log10(anchor / channels)
    return desc


def _trailing_mean(points, key, lo, hi):
    """Mean viewers among points timestamped in [lo, hi), else None."""
    vals = []
    for point in points or []:
        if not isinstance(point, dict):
            continue
        start = _epoch(point.get(key))
        viewers = _finite(point.get("viewers"))
        if start is None or viewers is None:
            continue
        if lo <= start < hi:
            vals.append(viewers)
    return sum(vals) / len(vals) if vals else None


def _case_anchor(case, day_start_epoch=0.0):
    # The output cap is 1e9, so an anchor above it can only do harm
    # (overflow in the persistence fallback); clamp it on the way in.
    for row in case.get("known") or []:
        if isinstance(row, dict) and row.get("hour") == 1:
            viewers = _finite(row.get("viewers"))
            if viewers:
                return min(max(1.0, viewers), 1e9)
    history = [p for p in case.get("history") or [] if isinstance(p, dict)]
    # Prefer the last point inside the release day's first hour; only a
    # case with no such point falls back to the last point of any age.
    for point in reversed(history):
        viewers = _finite(point.get("viewers"))
        start = _epoch(point.get("at"))
        if viewers and start is not None and start >= day_start_epoch:
            return min(max(1.0, viewers), 1e9)
    for point in reversed(history):
        viewers = _finite(point.get("viewers"))
        if viewers:
            return min(max(1.0, viewers), 1e9)
    return 1.0


def _case_pre_peak(case, day_start_epoch):
    peak = _peak_before(case.get("history_hourly"), "from",
                        day_start_epoch - 72 * 3600, day_start_epoch)
    if peak is None:
        peak = _peak_before(case.get("history"), "at",
                            day_start_epoch - 72 * 3600, day_start_epoch)
    return peak


def _hour1_channels(points, day_start_epoch):
    """Mean channels over the release day's first hour, else None."""
    vals = []
    for point in points or []:
        if not isinstance(point, dict):
            continue
        start = _epoch(point.get("from"))
        channels = _finite(point.get("channels"))
        if start is None or channels is None:
            continue
        if day_start_epoch <= start < day_start_epoch + 3600:
            vals.append(channels)
    return sum(vals) / len(vals) if vals else None


def _effective_size(weights):
    total = sum(weights)
    if total <= 0.0:
        return 0.0
    return total * total / sum(w * w for w in weights)


def _pool_shapes(case, anchor, target_desc, par):
    """hour -> list of (log-ratio shape, weight) from the comparables.

    Each comparable carries two weights: the full multi-dimension kernel
    weight and the scale-only kernel weight. An hour whose full-weight
    pool has collapsed to a tiny effective sample size (one launch
    dominating) is stabilised by admixing the scale-only weights -- the
    smallest admixture from a fixed ladder that lifts the effective
    size to the target, capped so the local pool still leads.
    """
    own_name = str((case.get("game") or {}).get("name", "")).casefold()
    dims = (("scale", par["bw_scale"]), ("peak", par["bw_peak"]),
            ("front", par["bw_front"]), ("ramp", par["bw_ramp"]),
            ("vpc", par["bw_vpc"]))
    launches = []  # (per-hour log ratios, full weight, scale-only weight)
    for comp in sorted(case.get("comparables") or [],
                       key=lambda c: str(c.get("name", ""))
                       if isinstance(c, dict) else ""):
        if not isinstance(comp, dict):
            continue
        comp_name = str(comp.get("name", "")).casefold()
        if comp_name == own_name or comp_name.startswith(own_name + " "):
            continue  # the target's own category history is not a launch
        start = _epoch(comp.get("day_start"))
        if start is None:
            continue
        curve = _hourly_curve(start, comp.get("hourly"))
        own_anchor = curve.get(1)
        if own_anchor is None or own_anchor < par["min_anchor"]:
            continue
        own_peak = _peak_before(comp.get("hourly"), "from",
                                start - 72 * 3600, start)
        own_trail = _trailing_mean(comp.get("hourly"), "from",
                                   start - par["trail_hours"] * 3600, start)
        own_desc = _descriptors(own_anchor, own_peak, own_trail,
                                _hour1_channels(comp.get("hourly"), start))
        log_weight = 0.0
        log_scale = None
        usable = True
        for dim, width in dims:
            if width is None:
                continue
            if dim not in target_desc or dim not in own_desc:
                if dim == "scale":
                    usable = False
                continue
            pull = (own_desc[dim] - target_desc[dim]) / width
            log_weight -= 0.5 * pull * pull
            if dim == "scale":
                log_scale = -0.5 * pull * pull
        if not usable or log_scale is None:
            continue
        log_anchor = math.log(own_anchor)
        ratios = {hour: math.log(curve[hour]) - log_anchor
                  for hour in FORECAST_HOURS
                  if curve.get(hour) and curve[hour] > 0}
        launches.append((ratios, math.exp(log_weight), math.exp(log_scale)))
    pool = {}
    ladder = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, par["ess_mix_max"])
    for hour in FORECAST_HOURS:
        entries = [(ratios[hour], full, scale)
                   for ratios, full, scale in launches if hour in ratios]
        if not entries:
            pool[hour] = []
            continue
        chosen = ladder[-1]
        for mix in ladder:
            weights = [full + mix * scale for _, full, scale in entries]
            if _effective_size(weights) >= par["ess_target"]:
                chosen = mix
                break
        pool[hour] = [(value, full + chosen * scale)
                      for value, full, scale in entries]
    return pool


def _gaussian_quantiles(pairs, par):
    total = sum(w for _, w in pairs)
    mean = sum(v * w for v, w in pairs) / total
    var = sum(w * (v - mean) ** 2 for v, w in pairs) / total
    sd = max(math.sqrt(var), par["sd_min"])
    return [mean + z * sd for z in Z_LEVELS]


def _empirical_quantiles(pairs):
    """Weighted quantiles by linear interpolation over the cumulative
    weight midpoints of the sorted sample."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    centres = []
    walked = 0.0
    for value, weight in ordered:
        centres.append((value, (walked + 0.5 * weight) / total))
        walked += weight
    out = []
    for tau in LEVELS:
        if tau <= centres[0][1]:
            out.append(centres[0][0])
            continue
        if tau >= centres[-1][1]:
            out.append(centres[-1][0])
            continue
        for (v0, c0), (v1, c1) in zip(centres, centres[1:]):
            if c0 <= tau <= c1:
                share = 0.0 if c1 == c0 else (tau - c0) / (c1 - c0)
                out.append(v0 + (v1 - v0) * share)
                break
    return out


def _persistence(anchor, hour, par):
    width = par["base_sigma"] * math.sqrt(hour - 1)
    return [anchor * math.exp(z * width) for z in Z_LEVELS]


def _hour_row(pairs, anchor, hour, par):
    if not pairs or sum(w for _, w in pairs) <= 1e-12:
        raw = _persistence(anchor, hour, par)
    else:
        gauss = _gaussian_quantiles(pairs, par)
        empir = _empirical_quantiles(pairs)
        share = par["empirical_share"]
        blended = [share * e + (1.0 - share) * g
                   for e, g in zip(empir, gauss)]
        median = blended[2]
        spread = par["spread"]
        if par["ess_widen"] > 0.0:
            ess = _effective_size([w for _, w in pairs])
            if ess < par["ess_ref"]:
                spread *= 1.0 + par["ess_widen"] * (par["ess_ref"] - ess) \
                    / par["ess_ref"]
        if spread != 1.0:
            blended = [median + spread * (v - median) for v in blended]
        raw = [anchor * math.exp(v) for v in blended]
    row = [min(max(v, 0.01), par["cap"]) for v in raw]
    row.sort()
    return [_clean(v) for v in row]


def _wanted_hours(case):
    entries = case.get("forecast")
    if not isinstance(entries, list):
        entries = []
    hours = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            hour = int(entry["hour"])
        except (KeyError, TypeError, ValueError):
            continue
        if 2 <= hour <= 24 and hour not in hours:
            hours.append(hour)
    return hours if len(hours) == len(FORECAST_HOURS) else list(FORECAST_HOURS)


def forecast_case(case, params=None):
    par = dict(PARAMS)
    if params:
        par.update(params)
    day_start = _epoch(case.get("day_start")) or 0.0
    anchor = _case_anchor(case, day_start)
    pre_peak = _case_pre_peak(case, day_start)
    trail = _trailing_mean(case.get("history_hourly"), "from",
                           day_start - par["trail_hours"] * 3600, day_start)
    if trail is None:
        trail = _trailing_mean(case.get("history"), "at",
                               day_start - par["trail_hours"] * 3600, day_start)
    channels = None
    for row in case.get("known") or []:
        if isinstance(row, dict) and row.get("hour") == 1:
            channels = _finite(row.get("channels"))
    target_desc = _descriptors(anchor, pre_peak, trail, channels)
    pool = _pool_shapes(case, anchor, target_desc, par)
    return {"hours": [{"hour": h, "q": _hour_row(pool.get(h, []), anchor, h, par)}
                      for h in _wanted_hours(case)]}


def _fallback_case(case, params=None):
    par = dict(PARAMS)
    if params:
        par.update(params)
    anchor = _case_anchor(case, _epoch(case.get("day_start")) or 0.0)
    return {"hours": [{"hour": h,
                       "q": [_clean(min(max(v, 0.01), par["cap"]))
                             for v in _persistence(anchor, h, par)]}
                      for h in _wanted_hours(case)]}


def _blank_case():
    par = dict(PARAMS)
    return {"hours": [{"hour": h,
                       "q": [_clean(v) for v in _persistence(1.0, h, par)]}
                      for h in FORECAST_HOURS]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            case = None
        if not isinstance(case, dict):
            result = _blank_case()
        else:
            try:
                result = forecast_case(case)
            except Exception:
                try:
                    result = _fallback_case(case)
                except Exception:
                    result = _blank_case()
        text = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
