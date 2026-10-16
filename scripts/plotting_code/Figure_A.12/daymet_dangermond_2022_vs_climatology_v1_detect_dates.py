"""
Daymet V4 at Dangermond Preserve: 2022 vs 1981-2024 climatology
================================================================

Pulls the full daily Daymet record (1981-2024) for an AOI, derives VPD on every
daily image *before* averaging, builds a day-of-year climatology, and compares
the 2022 series against it.

Outputs
-------
  daymet_dangermond_daily_ALL.csv    raw daily series, 1981-2024 (cached)
  daymet_dangermond_climatology.csv  day-of-year climatology + 2022 anomalies
  daymet_dangermond_2022_vs_clim.png four-panel comparison figure
  (console)                          anomaly stats for the SHIFT window

Requirements
------------
  pip install earthengine-api pandas numpy matplotlib
  earthengine authenticate

Notes
-----
* Daymet bands are lowercase: vp, tmin, tmax, prcp.

* Daymet uses a 365-day calendar: Dec 31 is dropped in leap years, Feb 29 kept.
  Climatology is therefore grouped on month-day (not DOY) to avoid a one-day
  shift after February in leap years. Feb 29 is excluded from the climatology
  (2022 is not a leap year, so the target series is unaffected).

* VPD is computed per image, then averaged. Computing it from climatological
  mean temperatures would bias it low, because es(T) is convex (Jensen).

* Daymet is a 1 km gridded interpolation of station data, not observations.
  At Point Conception it will smooth coastal marine-layer gradients.
"""

import os
import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ----------------------------------------------------------------------
# 0. CONFIG
# ----------------------------------------------------------------------

EE_PROJECT = "p1-jacabenga"

SITE_LON, SITE_LAT = -120.41542, 34.52548     # Ramajal EC site, Dangermond
BUFFER_M = 5000                                # 0 = single containing cell

TARGET_YEAR = 2022
CLIM_FIRST, CLIM_LAST = 1981, 2024             # full Daymet availability

# SHIFT AVIRIS-NG acquisition window
SHIFT_START, SHIFT_END = f"{TARGET_YEAR}-02-24", f"{TARGET_YEAR}-05-29"

F_ALL = "daymet_dangermond_daily_ALL.csv"
F_CLIM = "daymet_dangermond_climatology.csv"
F_FIG = "daymet_dangermond_2022_vs_clim.png"

dirout_ = r'C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\dirout_meteo'

SMOOTH_WINDOW = 15        # days, centred rolling mean applied to climatology

# ----------------------------------------------------------------------
# 1. SERVER-SIDE: add VPD bands to each daily image
# ----------------------------------------------------------------------

BANDS = ["vp", "tmin", "tmax", "prcp"]


def add_vpd(img):
    """Add daily-mean and daytime VPD (kPa) to a Daymet image.

    es(T) = 0.6108 * exp(17.27 T / (T + 237.3))     [kPa, T in degC]
    ea    = vp / 1000                                [Pa -> kPa]
    """
    tmin = img.select("tmin")
    tmax = img.select("tmax")
    ea = img.select("vp").divide(1000.0)

    es_tmin = tmin.expression("0.6108 * exp(17.27 * T / (T + 237.3))", {"T": tmin})
    es_tmax = tmax.expression("0.6108 * exp(17.27 * T / (T + 237.3))", {"T": tmax})

    vpd_mean = es_tmin.add(es_tmax).divide(2).subtract(ea).max(0).rename("vpd_mean")
    vpd_max = es_tmax.subtract(ea).max(0).rename("vpd_max")
    tmean = tmin.add(tmax).divide(2).rename("tmean")

    return img.addBands([vpd_mean, vpd_max, tmean])


# ----------------------------------------------------------------------
# 2. DOWNLOAD (cached)
# ----------------------------------------------------------------------

def fetch_year(year, aoi):
    """Daily AOI-mean series for one calendar year."""
    col = (ee.ImageCollection("NASA/ORNL/DAYMET_V4")
           .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
           .select(BANDS)
           .map(add_vpd))

    def _reduce(img):
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=1000, maxPixels=1e9)
        return ee.Feature(None, stats).set("date", img.date().format("YYYY-MM-dd"))

    feats = ee.FeatureCollection(col.map(_reduce)).getInfo()["features"]
    return pd.DataFrame([f["properties"] for f in feats])


def load_all():
    if os.path.exists(F_ALL):
        print(f"Loading cached {F_ALL}")
        df = pd.read_csv(F_ALL, parse_dates=["date"])
        return df

    ee.Initialize(project=EE_PROJECT)
    pt = ee.Geometry.Point([SITE_LON, SITE_LAT])
    aoi = pt.buffer(BUFFER_M) if BUFFER_M > 0 else pt

    frames = []
    for yr in range(CLIM_FIRST, CLIM_LAST + 1):
        print(f"  fetching {yr} ...", flush=True)
        frames.append(fetch_year(yr, aoi))

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(os.path.join(dirout_, F_ALL), index=False)
    print(f"Wrote {F_ALL}  ({len(df)} rows)")
    return df


df = load_all()

df["year"] = df["date"].dt.year
df["md"] = df["date"].dt.strftime("%m-%d")

# California water year: 1 Oct - 30 Sep
df["wyear"] = np.where(df["date"].dt.month >= 10, df["year"] + 1, df["year"])
wy_start = pd.to_datetime(dict(year=df["wyear"] - 1, month=10, day=1))
df["wday"] = (df["date"] - wy_start).dt.days + 1

VARS = ["prcp", "tmin", "tmax", "tmean", "vpd_mean", "vpd_max"]

# ----------------------------------------------------------------------
# 3. DAY-OF-YEAR CLIMATOLOGY (month-day grouping, Feb 29 excluded)
# ----------------------------------------------------------------------

hist = df[df["md"] != "02-29"]

clim = hist.groupby("md")[VARS].agg(["mean", "std",
                                     lambda s: s.quantile(.10),
                                     lambda s: s.quantile(.90)])
clim.columns = [f"{v}_{s}" for v, s in clim.columns]
clim = clim.rename(columns=lambda c: c.replace("<lambda_0>", "p10")
                                      .replace("<lambda_1>", "p90"))
clim = clim.reset_index()

# Smooth the daily climatology (wrap-around at year end)
sm_cols = [c for c in clim.columns if c != "md"]
pad = pd.concat([clim.tail(SMOOTH_WINDOW), clim, clim.head(SMOOTH_WINDOW)])
pad[sm_cols] = pad[sm_cols].rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()
clim = pad.iloc[SMOOTH_WINDOW:-SMOOTH_WINDOW].reset_index(drop=True)

# ----------------------------------------------------------------------
# 4. JOIN 2022 TO CLIMATOLOGY
# ----------------------------------------------------------------------

tgt = df[df["year"] == TARGET_YEAR].copy()
cmp_ = tgt.merge(clim, on="md", how="left")

for v in VARS:
    cmp_[f"{v}_anom"] = cmp_[v] - cmp_[f"{v}_mean"]
    cmp_[f"{v}_z"] = cmp_[f"{v}_anom"] / cmp_[f"{v}_std"].replace(0, np.nan)

cmp_.to_csv(os.path.join(dirout_, F_CLIM), index=False)
print(f"Wrote {F_CLIM}")

# ----------------------------------------------------------------------
# 5. CUMULATIVE PRECIPITATION BY WATER YEAR
# ----------------------------------------------------------------------

wy = df[(df["wyear"] > CLIM_FIRST) & (df["wyear"] <= CLIM_LAST)].copy()
wy["cum"] = wy.groupby("wyear")["prcp"].cumsum()

cum_clim = wy[wy["wyear"] != TARGET_YEAR].groupby("wday")["cum"].agg(
    mean="mean", p10=lambda s: s.quantile(.10), p90=lambda s: s.quantile(.90))
cum_2022 = wy[wy["wyear"] == TARGET_YEAR].set_index("wday")["cum"]

wy_totals = wy.groupby("wyear")["prcp"].sum()
v = wy_totals.loc[TARGET_YEAR]
print(f"\nWater-year {TARGET_YEAR} precipitation")
print(f"  total:           {v:7.1f} mm")
print(f"  {CLIM_FIRST+1}-{CLIM_LAST} mean:  {wy_totals.mean():7.1f} mm")
print(f"  anomaly:         {v - wy_totals.mean():+7.1f} mm "
      f"({100 * (v / wy_totals.mean() - 1):+.1f} %)")
print(f"  z-score:         {(v - wy_totals.mean()) / wy_totals.std():+7.2f}")
print(f"  percentile rank: {(wy_totals < v).mean() * 100:.0f}th "
      f"(rank {int(wy_totals.rank().loc[TARGET_YEAR])} of {len(wy_totals)})")

# ----------------------------------------------------------------------
# 6. ANOMALIES WITHIN THE SHIFT ACQUISITION WINDOW
# ----------------------------------------------------------------------

win = cmp_[(cmp_["date"] >= SHIFT_START) & (cmp_["date"] <= SHIFT_END)]
print(f"\nSHIFT window ({SHIFT_START} to {SHIFT_END}), {len(win)} days")
print(f"{'variable':<12}{'2022':>9}{'clim':>9}{'anom':>9}{'mean z':>9}")
for var, unit in [("prcp", "mm/d"), ("tmax", "degC"), ("tmin", "degC"),
                  ("vpd_mean", "kPa"), ("vpd_max", "kPa")]:
    a, b = win[var].mean(), win[f"{var}_mean"].mean()
    print(f"{var:<12}{a:9.2f}{b:9.2f}{a - b:+9.2f}{win[f'{var}_z'].mean():+9.2f}"
          f"   {unit}")


# ----------------------------------------------------------------------
# 6b. APPENDIX: EXCEEDANCE FREQUENCY AND EVENT DURATION (>90th pct)
# ----------------------------------------------------------------------
# Rationale: a mean z-score across the SHIFT window can understate a signal
# that is concentrated in a handful of sustained events rather than spread
# evenly across all 95 days. This section instead asks: (i) what fraction
# of days/time exceeded the climatological 90th percentile for that
# calendar date, and (ii) how many distinct multi-day events made up that
# time, since a week-long event has a different physiological/phenological
# meaning than several isolated days.
#
# "p90" here is the same climatological 90th percentile already computed
# in the smoothed day-of-year climatology (Section 3), i.e. cmp_[f"{v}_p90"].
#
# THIS SECTION IS THE MOST RELIABLE WAY TO GET EXACT DATES for specific
# events (e.g. "the rain event", "the heatwave") -- read the printed
# "event list" below rather than eyeballing the figure, since it gives
# exact start/end dates directly from your own site-specific series.


def exceedance_events(flag_series, dates):
    """Given a boolean exceedance flag indexed to consecutive dates,
    return (events_df, summary_dict).

    events_df: one row per contiguous run of True values, with start date,
               end date, and length in days.
    """
    flag = flag_series.to_numpy()
    dates = pd.DatetimeIndex(dates)

    # Assumes `dates` is daily and contiguous (true for the 95-day SHIFT
    # window as constructed above -- no missing days expected).
    change = np.diff(flag.astype(int), prepend=0)
    run_start_idx = np.where(change == 1)[0]
    run_end_idx = np.where(np.diff(flag.astype(int), append=0) == -1)[0]

    events = pd.DataFrame({
        "start": dates[run_start_idx],
        "end": dates[run_end_idx],
    })
    events["length_days"] = (events["end"] - events["start"]).dt.days + 1

    n_days_total = len(flag)
    n_days_exceed = int(flag.sum())

    summary = {
        "n_days_total": n_days_total,
        "n_days_exceed": n_days_exceed,
        "pct_time_exceed": 100 * n_days_exceed / n_days_total,
        "n_events": len(events),
        "n_events_ge2d": int((events["length_days"] >= 2).sum()),
        "n_events_ge3d": int((events["length_days"] >= 3).sum()),
        "days_in_events_ge2d": int(events.loc[events["length_days"] >= 2, "length_days"].sum()),
        "days_in_events_ge3d": int(events.loc[events["length_days"] >= 3, "length_days"].sum()),
        "max_event_length_days": int(events["length_days"].max()) if len(events) else 0,
    }
    summary["pct_time_in_events_ge2d"] = 100 * summary["days_in_events_ge2d"] / n_days_total
    summary["pct_time_in_events_ge3d"] = 100 * summary["days_in_events_ge3d"] / n_days_total

    return events, summary


print(f"\nAppendix: exceedance of the climatological 90th percentile "
      f"within the SHIFT window ({SHIFT_START} to {SHIFT_END})")

exceed_events_by_var = {}

for var, unit in [("vpd_max", "kPa"), ("vpd_mean", "kPa"), ("prcp", "mm/d"), ("tmax", "degC")]:
    flag = win[var] > win[f"{var}_p90"]
    events, summ = exceedance_events(flag, win["date"])
    exceed_events_by_var[var] = events

    print(f"\n  {var} ({unit}), threshold = climatological 90th percentile")
    print(f"    days above p90:            {summ['n_days_exceed']} / {summ['n_days_total']} "
          f"({summ['pct_time_exceed']:.1f}% of window time)")
    print(f"    distinct events:           {summ['n_events']}")
    print(f"    events lasting >=2 days:   {summ['n_events_ge2d']}  "
          f"({summ['pct_time_in_events_ge2d']:.1f}% of window time)")
    print(f"    events lasting >=3 days:   {summ['n_events_ge3d']}  "
          f"({summ['pct_time_in_events_ge3d']:.1f}% of window time)")
    print(f"    longest single event:      {summ['max_event_length_days']} days")
    if summ["n_events"] > 0:
        ev_str = "; ".join(
            f"{r.start.strftime('%d %b')}-{r.end.strftime('%d %b')} "
            f"({r.length_days}d)"
            for r in events.itertuples())
        print(f"    event list:                {ev_str}")

# Save the VPD (daytime) event table specifically, since this is the
# variable most directly tied to the phenological-shift argument.
vpd_events_path = os.path.join(dirout_, "vpd_max_exceedance_events_SHIFT_window.csv")
exceed_events_by_var["vpd_max"].to_csv(vpd_events_path, index=False)
print(f"\nWrote {vpd_events_path}")

# ----------------------------------------------------------------------
# 7. FIGURE
# ----------------------------------------------------------------------
from matplotlib.transforms import blended_transform_factory

fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=False)
d = cmp_["date"]

panel_labels = ["A", "B", "C", "D"]

# (A) cumulative precipitation, water year
ax = axes[0]
ax.fill_between(cum_clim.index, cum_clim["p10"], cum_clim["p90"],
                color="steelblue", alpha=.20, label="10th-90th pct")

ax.plot(cum_clim.index[:-1], cum_clim["mean"][:-1], color="steelblue", ls="-.", lw=1.0,
        label=f"{CLIM_FIRST+1}-{CLIM_LAST} mean")

ax.plot(cum_2022.index, cum_2022.values, color="firebrick", lw=2,
        label=f"water year {TARGET_YEAR}")

ax.set_ylabel("Cumulative precip (mm)")
ax.set_xlabel("Day of water year (1 = 1 Oct)")
ax.legend(fontsize=8, frameon=False, loc="upper left")

ax.axvspan(147, 241, color="steelblue", alpha=.10, zorder=0)

# 5-day grid (panel A x-axis is an integer "day of water year", not a date)
ax.set_xticks(np.arange(0, cum_clim.index.max() + 1, 5), minor=True)
ax.grid(True, which="minor", axis="x", alpha=0.2, linestyle=":")
ax.grid(True, which="major", axis="y", alpha=0.3)

# add legend to span ('SHIFT campaign')
trans = blended_transform_factory(ax.transData, ax.transAxes)

ax.text(241 - 18, 0.1,
        "SHIFT campaign",
        transform=trans,
        fontsize=8,
        color="steelblue",
        ha="center",
        va="top",
        alpha=0.8)

ax.set_title("Dangermond Preserve — Daymet V4")

# Panel label
ax.text(0.97, 0.97, panel_labels[0], transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')

# (B-D) daily series vs climatology
panels = [("tmax", "Max air temp (°C)", "firebrick"),
          ("vpd_max", "Daytime VPD (kPa)", "darkorange"),
          ("vpd_mean", "Daily-mean VPD (kPa)", "grey")]

for i, (ax, (var, lab, col)) in enumerate(zip(axes[1:], panels), start=1):
    ax.fill_between(d, cmp_[f"{var}_p10"], cmp_[f"{var}_p90"],
                    color="grey", alpha=.20, label="10th-90th pct")
    ax.plot(d, cmp_[f"{var}_mean"], color="k", ls="-.", lw=1.0, alpha=.6,
            label=f"{CLIM_FIRST}-{CLIM_LAST} mean")
    ax.plot(d, cmp_[var], color=col, lw=1.2, label=str(TARGET_YEAR))
    ax.set_ylabel(lab)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    ax.axvspan(pd.Timestamp(SHIFT_START), pd.Timestamp(SHIFT_END),
               color="steelblue", alpha=.10, zorder=0)

    # add legend to span ('SHIFT campaign')
    trans = blended_transform_factory(ax.transData, ax.transAxes)

    mid = (pd.Timestamp(SHIFT_END) - pd.Timedelta(days=18))  # positioning of label text

    ax.text(mid, 0.1,   # positioning
            "SHIFT campaign",
            transform=trans,
            fontsize=8,
            color="steelblue",
            ha="center",
            va="top",
            alpha=0.8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())

    # 5-day grid: minor ticks every 5 days, unlabeled, just gridlines
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=5))
    ax.grid(True, which="minor", axis="x", alpha=0.2, linestyle=":")
    ax.grid(True, which="major", axis="y", alpha=0.3)

    # Panel label
    ax.text(0.97, 0.97, panel_labels[i], transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')

axes[-1].set_xlabel(f"{TARGET_YEAR}  (shaded = SHIFT acquisition window)")
plt.tight_layout()
plt.savefig(os.path.join(dirout_, F_FIG), dpi=200)
plt.close()
print(f"\nWrote {F_FIG}")
