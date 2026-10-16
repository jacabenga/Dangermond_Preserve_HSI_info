# RF_windowed_v3_fixed_split_v0.py
# date: 01.10.2026
"""
Spectrally windowed Random Forest classification -- v3 (reviewer comment 3)

Changes vs v2 (RF_shrub_ex_simple_2_for_loop.py, RF_tree_ex_simple_2.1_for_loop.py)
------------------------------------------------------------------------------
1. SAME PIXELS AND SAME SPLIT FOR EVERY WINDOW.
   v2 drew the 100k subsample INSIDE the window loop with an unseeded
   np.random.choice, so every window saw different pixels and a different
   train/test partition. Here the subsample and the split are drawn ONCE per
   (date, repetition) and reused for all windows.
2. SENSITIVITY TO RESAMPLING. The whole experiment is repeated N_REPEATS times
   with different seeds. We report mean +/- SD per window and the Spearman
   rank correlation of window scores between repetitions.
3. METRICS. Overall accuracy (OA), balanced accuracy (mean per-class recall),
   macro-F1, Cohen's kappa and per-class recall. Class counts are saved.
4. CLASS IMBALANCE. Handled identically in every task (class_weight below).
   v2 used it only in the Tree script.
5. DATE/FILE MATCHING by the date string in the filename, not by the position
   in os.listdir() (whose order is not guaranteed).
6. ONE SCRIPT FOR ALL PFTs: only TASK changes.
7. Window edges are half-open [lo, hi), so a band on an edge is assigned to
   exactly one window (v2 used closed intervals on both sides).
8. The reflectance stretch R' = (2 - R)^2 is OFF by default: it is monotonic
   per band, and tree splits are invariant to monotonic per-feature
   transforms, so it has no material effect on RF results. Set
   APPLY_STRETCH = True to reproduce v2 exactly.
"""

import os
import re
import gc
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             cohen_kappa_score, recall_score)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

DIR_PARQUET = r'D:\Jaime_C._Revenga_LandCRAFT_2025\Project_with_F.Schneider_S.Normand\data\mosaics_labeled\df_parquet_streamed'
DIR_OUT = r'C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\controlled_experiments_RF\results_RF_windowed_v3'
os.makedirs(DIR_OUT, exist_ok=True)

TASK = 'Herbaceous' #  'Tree' # 'Shrub'          # 'Shrub' | 'Herbaceous'  -> communities within the PFT
                        # 'Tree'                  -> Coast Live Oak Woodland vs the other 7 classes
N_SEGMENTS = 32         # as in the manuscript figures, for all three PFTs (~66.6 nm, 13-14 bands per window)
N_SUBSAMPLE = 100_000   # pixels per (date, repetition); all if fewer available
TEST_SIZE = 0.3
N_REPEATS = 10          # resampling repetitions (5 is enough if runtime is an issue)
BASE_SEED = 42
CLASS_WEIGHT = 'balanced'   # or None to reproduce v2-Shrub behaviour
N_ESTIMATORS = 100
APPLY_STRETCH = False
NODATA = -9999.0

DATES = ['2022-02-24', '2022-02-28', '2022-03-08', '2022-03-16', '2022-03-22',
         '2022-04-05', '2022-04-12', '2022-04-20', '2022-04-29',
         '2022-05-03', '2022-05-11', '2022-05-17', '2022-05-29']

PFT_MAPPING = {16: 'Tree',
               20: 'Shrub', 5: 'Shrub', 11: 'Shrub', 37: 'Shrub', 31: 'Shrub',
               33: 'Herbaceous', 43: 'Herbaceous'}

VEG_CLASS_NAMES = {5: 'Black Sage Scrub', 11: 'California Sagebrush Scrub',
                   16: 'Coast Live Oak Woodland', 20: 'Coyote Brush Scrub',
                   31: "Menzie's Golden Bush Scrub", 33: 'Non-native Grassland',
                   37: 'Purple Sage Scrub', 43: 'Semi-natural Herbaceous Stands'}

# ---- EXCLUSION RULE (keep identical to Figure_5_publication_ready_clean_code_only-plotting_v3.py) ----
# A window is flagged 'excluded_gray' if at least EXCLUDE_MIN_OVERLAP of its width
# lies inside GRAY_REGIONS_NM (water vapour / sensor limits). The flag does NOT
# change any model fit: all windows are still trained and evaluated; it only
# marks windows to hide in figures and leaves them out of the stability stats.
# With 32 windows this flags Seg01, Seg15, Seg16, Seg22, Seg23, Seg24, Seg32.
GRAY_REGIONS_NM = [(0, 450), (1325, 1450), (1775, 2000), (2450, 2510)]
EXCLUDE_MIN_OVERLAP = 0.5


def is_excluded(lo, hi):
    """Exclusion rule shared with the Figure 5 plotting script."""
    overlap = sum(max(0.0, min(hi, r1) - max(lo, r0)) for r0, r1 in GRAY_REGIONS_NM)
    return overlap / (hi - lo) >= EXCLUDE_MIN_OVERLAP


def wavelength_nm(band_num):
    """Same convention as v2: band_{i+1} -> 380 + 5.02 * i."""
    return 380.0 + 5.02 * (band_num - 1)


# ----------------------------------------------------------------------
# 1. Match files to dates by the date in the filename
# ----------------------------------------------------------------------

date_re = re.compile(r'(20\d{2})-?(\d{2})-?(\d{2})')
date_files = {}
for f in os.listdir(DIR_PARQUET):
    m = date_re.search(f)
    if m and f.endswith('.parquet'):
        date_files['-'.join(m.groups())] = os.path.join(DIR_PARQUET, f)

missing = [d for d in DATES if d not in date_files]
if missing:
    raise FileNotFoundError(f'No parquet file found for dates {missing}. '
                            f'Files present: {sorted(os.listdir(DIR_PARQUET))}')
print('Date -> file:')
for d in DATES:
    print(f'  {d}: {os.path.basename(date_files[d])}')


# ----------------------------------------------------------------------
# 2. Task definition (labels, class names)
# ----------------------------------------------------------------------

def build_task(df):
    """Return (df_filtered, y, label_names) for the chosen TASK."""
    df = df[df['veg_class'].isin(PFT_MAPPING.keys())]
    if TASK in ('Shrub', 'Herbaceous'):
        keep = [c for c, p in PFT_MAPPING.items() if p == TASK]
        df = df[df['veg_class'].isin(keep)]
        y = df['veg_class'].astype(int).to_numpy()
        names = {c: VEG_CLASS_NAMES[c] for c in sorted(keep)}
    elif TASK == 'Tree':
        y = (df['veg_class'].astype(int) == 16).astype(int).to_numpy()
        names = {0: 'Other (7 classes)', 1: VEG_CLASS_NAMES[16]}
    else:
        raise ValueError(f'Unknown TASK {TASK}')
    return df, y, names


# ----------------------------------------------------------------------
# 3. Spectral windows (computed once from the band columns)
# ----------------------------------------------------------------------

def build_windows(band_cols):
    nums = np.array([int(c.split('_')[1]) for c in band_cols])
    wl = wavelength_nm(nums)
    edges = np.linspace(380, 2510, N_SEGMENTS + 1)
    windows = []
    for k in range(N_SEGMENTS):
        lo, hi = edges[k], edges[k + 1]
        last = (k == N_SEGMENTS - 1)
        mask = (wl >= lo) & ((wl <= hi) if last else (wl < hi))
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        windows.append(dict(window=f'Seg{k + 1:02d} ({lo:.0f}-{hi:.0f} nm)',
                            lo_nm=lo, hi_nm=hi, idx=idx, excluded=is_excluded(lo, hi)))
    full = np.flatnonzero((wl >= 400) & (wl <= 2500))
    windows.append(dict(window='Full (400-2500 nm)', lo_nm=400.0, hi_nm=2500.0,
                        idx=full, excluded=False))
    return windows


# ----------------------------------------------------------------------
# 4. Main loop: date -> repetition (fixed pixels + split) -> windows
# ----------------------------------------------------------------------

rows = []
class_count_rows = []
windows = None

for date in DATES:
    print('=' * 70)
    print(f'{date}  |  task = {TASK}')

    df = pd.read_parquet(date_files[date])
    band_cols = sorted([c for c in df.columns if c.startswith('band_')],
                       key=lambda c: int(c.split('_')[1]))
    df = df[~(df == NODATA).any(axis=1)]
    df, y, names = build_task(df)
    labels = sorted(names)

    X_full = df[band_cols].to_numpy(dtype=np.float32)
    if APPLY_STRETCH:
        X_full = (2.0 - X_full) ** 2
    del df
    gc.collect()

    if windows is None:
        windows = build_windows(band_cols)
        print(f'{len(windows)} windows; bands per window: '
              f'{[len(w["idx"]) for w in windows]}')

    counts_all = {names[c]: int((y == c).sum()) for c in labels}
    print('Class counts (all valid pixels):', counts_all)

    for rep in range(N_REPEATS):
        seed = BASE_SEED + rep
        rng = np.random.default_rng(seed)

        # --- ONE subsample and ONE split, shared by all windows ---
        n = len(y)
        sub = rng.choice(n, N_SUBSAMPLE, replace=False) if n > N_SUBSAMPLE else np.arange(n)
        tr, te = train_test_split(sub, test_size=TEST_SIZE, random_state=seed,
                                  stratify=y[sub])
        X_tr_full, X_te_full = X_full[tr], X_full[te]
        y_tr, y_te = y[tr], y[te]

        class_count_rows.append(dict(
            date=date, rep=rep, seed=seed,
            counts_all=json.dumps(counts_all),
            counts_train=json.dumps({names[c]: int((y_tr == c).sum()) for c in labels}),
            counts_test=json.dumps({names[c]: int((y_te == c).sum()) for c in labels})))

        for w in windows:
            rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=None,
                                        n_jobs=-1, random_state=seed,
                                        class_weight=CLASS_WEIGHT)
            rf.fit(X_tr_full[:, w['idx']], y_tr)
            y_pred = rf.predict(X_te_full[:, w['idx']])

            rec = recall_score(y_te, y_pred, labels=labels, average=None, zero_division=0)
            row = dict(date=date, rep=rep, seed=seed, window=w['window'],
                       lo_nm=w['lo_nm'], hi_nm=w['hi_nm'], n_bands=len(w['idx']),
                       excluded_gray=w['excluded'],
                       n_train=len(tr), n_test=len(te),
                       OA=accuracy_score(y_te, y_pred),
                       balanced_acc=balanced_accuracy_score(y_te, y_pred),
                       macro_F1=f1_score(y_te, y_pred, average='macro', zero_division=0),
                       kappa=cohen_kappa_score(y_te, y_pred))
            for c, r in zip(labels, rec):
                row[f'recall_{names[c]}'] = r
            rows.append(row)
            del rf

        print(f'  rep {rep + 1}/{N_REPEATS} done (seed {seed})')
        del X_tr_full, X_te_full
        gc.collect()

    # checkpoint after each date
    pd.DataFrame(rows).to_csv(os.path.join(DIR_OUT, f'RF_windowed_{TASK}_all_runs.csv'), index=False)
    del X_full
    gc.collect()

res = pd.DataFrame(rows)
res.to_csv(os.path.join(DIR_OUT, f'RF_windowed_{TASK}_all_runs.csv'), index=False)
pd.DataFrame(class_count_rows).to_csv(
    os.path.join(DIR_OUT, f'RF_windowed_{TASK}_class_counts.csv'), index=False)


# ----------------------------------------------------------------------
# 5. Summaries: mean +/- SD per window, and sensitivity to resampling
# ----------------------------------------------------------------------

metrics = ['OA', 'balanced_acc', 'macro_F1', 'kappa']
summary = (res.groupby(['date', 'window', 'lo_nm', 'n_bands', 'excluded_gray'])[metrics]
              .agg(['mean', 'std']))
summary.columns = [f'{m}_{s}' for m, s in summary.columns]
summary = summary.reset_index().sort_values(['date', 'lo_nm'])
summary.to_csv(os.path.join(DIR_OUT, f'RF_windowed_{TASK}_summary.csv'), index=False)

stab_rows = []
# stability over the windows shown in the figure (not Full, not excluded_gray)
for date, g in res[~res['window'].str.startswith('Full') & ~res['excluded_gray']].groupby('date'):
    piv = g.pivot(index='window', columns='rep', values='balanced_acc')
    rho = piv.corr(method='spearman').to_numpy()
    off = rho[~np.eye(len(rho), dtype=bool)]
    stab_rows.append(dict(
        date=date,
        spearman_rank_corr_mean=np.nanmean(off) if off.size else np.nan,
        spearman_rank_corr_min=np.nanmin(off) if off.size else np.nan,
        sd_within_window_mean=piv.std(axis=1).mean(),   # resampling noise
        sd_between_windows=piv.mean(axis=1).std()))     # signal across windows
stab = pd.DataFrame(stab_rows)
stab['signal_to_resampling_ratio'] = stab['sd_between_windows'] / stab['sd_within_window_mean']
stab.to_csv(os.path.join(DIR_OUT, f'RF_windowed_{TASK}_resampling_sensitivity.csv'), index=False)
print('\nSensitivity to resampling (balanced accuracy):')
print(stab.to_string(index=False))


# ----------------------------------------------------------------------
# 6. Plot per date: mean +/- SD balanced accuracy and OA, in wavelength order
# ----------------------------------------------------------------------

for date, g in summary.groupby('date'):
    seg = g[~g['window'].str.startswith('Full')].sort_values('lo_nm')
    full = g[g['window'].str.startswith('Full')]
    centres = seg['lo_nm'] + (2510 - 380) / N_SEGMENTS / 2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(centres, seg['balanced_acc_mean'], yerr=seg['balanced_acc_std'],
                fmt='o-', color='black', capsize=3, label='Balanced accuracy (mean ± SD)')
    ax.plot(centres, seg['OA_mean'], 's--', color='steelblue', alpha=0.7,
            label='Overall accuracy (mean)')
    if len(full):
        ax.axhline(full['balanced_acc_mean'].iloc[0], color='black', lw=0.8, ls=':',
                   label='Full spectrum, balanced accuracy')
    for a_lo, a_hi in GRAY_REGIONS_NM:
        ax.axvspan(a_lo, a_hi, color='grey', alpha=0.2)
    ax.set_xlabel('Window centre (nm)')
    ax.set_ylabel('Score')
    ax.set_title(f'RF by spectral window — {TASK}, {date} '
                 f'({N_REPEATS} repetitions, same pixels/split across windows)')
    ax.set_xlim(380, 2510)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_OUT, f'RF_windowed_{TASK}_{date}.png'), dpi=200)
    plt.close(fig)

print(f'\nDone. Results in {DIR_OUT}')

