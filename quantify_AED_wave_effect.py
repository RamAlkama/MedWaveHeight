#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantification robuste de l'effet de l'AED sur les vagues ERA5.

Sorties
-------
- aed_wave_quantification.csv
- aed_wave_yearly_DJF.csv
- aed_wave_quantification.png
- console: phrases directement réutilisables dans l'article

Analyse principale
------------------
1) EOF3 de SLP en DJF, calculée comme dans plot_Fig_Med.py.
2) Signe de PC3 orienté pour que les valeurs positives correspondent
   à un gradient de pression AED positif (Atlantique NE - Europe centrale).
3) Agrégation DJF correcte: décembre est affecté à l'année suivante.
4) Région fixée a priori (pas de sélection du pixel de corrélation maximale).
5) Régression par +1 écart-type d'AED, composites AED > +1 sigma et AED < -1 sigma.
6) IC 95 % par moving-block bootstrap afin de limiter l'effet de l'autocorrélation.
7) Sensibilités: série brute et série linéairement détrendée; Spearman en complément.

IMPORTANT
---------
Cette analyse quantifie l'association historique ERA5. Elle ne transforme pas
automatiquement la projection CMIP6 de l'AED en une projection déterministe de Hs.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.stats import linregress, spearmanr

# ============================================================
# PARAMETRES
# ============================================================
ERA5_FILE = "era5_195001_202412_all.nc"
OUTDIR = Path("figures")
OUTDIR.mkdir(exist_ok=True)

# Domaine EOF identique au manuscrit
EOF_LON = (-25.5, 20.0)
EOF_LAT = (20.0, 65.0)

# Centres d'action de l'AED, identiques aux scripts fournis
WEST_BOX = {"lon": (-24.5, -20.0), "lat": (45.0, 62.0)}
EAST_BOX = {"lon": (10.0, 15.0), "lat": (45.0, 62.0)}

# Région d'analyse fixée a priori.
# Ajuster seulement si une région explicitement définie dans le manuscrit est préférée.
WAVE_BOX = {"lon": (4.0, 10.0), "lat": (41.0, 44.5)}
REGION_NAME = "northwestern Mediterranean / Corsica–Gulf of Lion"

# Fichiers des quantiles mensuels déjà utilisés dans plot_Fig_Med.py
PERCENTILE_FILES = {
    "Hs_P90": "p90_swh_195001_202412_grid.nc",
    "Hs_P95": "p95_swh_195001_202412_grid.nc",
    "Hs_P99": "p99_swh_195001_202412_grid.nc",
}

N_BOOT = 10000
BLOCK_LENGTH = 5
RANDOM_SEED = 20260803
MIN_COMPLETE_MONTHS = 3

# ============================================================
# OUTILS
# ============================================================
def normalize_lon(da: xr.DataArray) -> xr.DataArray:
    if float(da.lon.max()) > 180:
        lon = ((da.lon + 180) % 360) - 180
        da = da.assign_coords(lon=lon).sortby("lon")
    return da


def spatial_mean(da: xr.DataArray, box: dict) -> xr.DataArray:
    """Moyenne surfacique pondérée par cos(latitude)."""
    sub = da.sel(
        lon=slice(box["lon"][0], box["lon"][1]),
        lat=slice(box["lat"][0], box["lat"][1]),
    )
    weights = np.cos(np.deg2rad(sub.lat))
    return sub.weighted(weights).mean(("lat", "lon"))


def compute_eof(slp_anom: xr.DataArray, n_modes: int = 3):
    """EOF pondérée en latitude, cohérente avec le script original."""
    weights = np.sqrt(np.cos(np.deg2rad(slp_anom.lat)))
    slp_w = slp_anom * weights
    slp_2d = slp_w.stack(space=("lat", "lon")).dropna("space")
    A = slp_2d.transpose("time", "space").values

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    var_frac = (s ** 2) / np.sum(s ** 2) * 100.0

    eof = xr.DataArray(
        Vt[:n_modes] * s[:n_modes, None],
        dims=("mode", "space"),
        coords={"mode": np.arange(1, n_modes + 1), "space": slp_2d.space},
    ).unstack("space")
    eof = eof / weights

    pc = xr.DataArray(
        U[:, :n_modes] * s[:n_modes],
        dims=("time", "mode"),
        coords={"time": slp_anom.time, "mode": np.arange(1, n_modes + 1)},
    )
    return eof, pc, var_frac[:n_modes]


def to_djf_yearly(da: xr.DataArray, how: str = "mean") -> xr.DataArray:
    """
    Agrège DJF correctement: décembre appartient à l'hiver de l'année suivante.
    Exclut les saisons incomplètes.
    """
    djf = da.where(da.time.dt.month.isin([12, 1, 2]), drop=True)
    season_year = xr.where(djf.time.dt.month == 12,
                           djf.time.dt.year + 1,
                           djf.time.dt.year)
    djf = djf.assign_coords(season_year=("time", season_year.values))

    count = djf.groupby("season_year").count("time")
    if how == "mean":
        out = djf.groupby("season_year").mean("time")
    elif how == "max":
        out = djf.groupby("season_year").max("time")
    else:
        raise ValueError("how doit être 'mean' ou 'max'")

    return out.where(count >= MIN_COMPLETE_MONTHS, drop=True)


def standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - np.nanmean(values)) / np.nanstd(values, ddof=1)


def linear_detrend(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    out = np.full(values.shape, np.nan, dtype=float)
    if mask.sum() >= 3:
        out[mask] = detrend(values[mask], type="linear")
    return out


def moving_block_indices(n: int, block_length: int, rng) -> np.ndarray:
    """Bootstrap circulaire par blocs, puis tronqué à n observations."""
    starts = rng.integers(0, n, size=int(np.ceil(n / block_length)))
    idx = np.concatenate([
        (np.arange(s, s + block_length) % n) for s in starts
    ])
    return idx[:n]


def statistic_bundle(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Statistiques interprétables:
    - pente en m par +1 sigma d'AED
    - différence positive vs négative
    - différence positive vs climatologie
    - pourcentages correspondants
    """
    xz = standardize(x)
    reg = linregress(xz, y)

    pos = y[xz > 1.0]
    neg = y[xz < -1.0]
    clim = np.nanmean(y)

    if len(pos) < 3 or len(neg) < 3:
        raise RuntimeError(
            "Trop peu de saisons dans les composites ±1 sigma. "
            "Vérifier la série ou utiliser temporairement ±0.75 sigma."
        )

    diff_pos_neg = np.nanmean(pos) - np.nanmean(neg)
    rel_pos_neg = 100.0 * diff_pos_neg / np.nanmean(neg)
    diff_pos_clim = np.nanmean(pos) - clim
    rel_pos_clim = 100.0 * diff_pos_clim / clim

    rho, rho_p = spearmanr(xz, y, nan_policy="omit")

    return {
        "n": int(np.isfinite(xz * y).sum()),
        "n_positive": int(len(pos)),
        "n_negative": int(len(neg)),
        "slope_m_per_1sd": float(reg.slope),
        "r": float(reg.rvalue),
        "p_ols": float(reg.pvalue),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "mean_positive_m": float(np.nanmean(pos)),
        "mean_negative_m": float(np.nanmean(neg)),
        "mean_all_m": float(clim),
        "diff_positive_negative_m": float(diff_pos_neg),
        "increase_positive_vs_negative_pct": float(rel_pos_neg),
        "diff_positive_climatology_m": float(diff_pos_clim),
        "increase_positive_vs_climatology_pct": float(rel_pos_clim),
    }


def bootstrap_ci(x, y, n_boot=N_BOOT, block_length=BLOCK_LENGTH, seed=RANDOM_SEED):
    """IC 95 % par moving-block bootstrap des années appariées."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    rng = np.random.default_rng(seed)

    keys = [
        "slope_m_per_1sd",
        "diff_positive_negative_m",
        "increase_positive_vs_negative_pct",
        "diff_positive_climatology_m",
        "increase_positive_vs_climatology_pct",
    ]
    draws = {k: [] for k in keys}

    for _ in range(n_boot):
        idx = moving_block_indices(n, block_length, rng)
        xb, yb = x[idx], y[idx]
        try:
            stat = statistic_bundle(xb, yb)
        except (RuntimeError, ValueError):
            continue
        for k in keys:
            if np.isfinite(stat[k]):
                draws[k].append(stat[k])

    ci = {}
    for k in keys:
        arr = np.asarray(draws[k], dtype=float)
        if len(arr) < n_boot * 0.5:
            raise RuntimeError(f"Bootstrap instable pour {k}: seulement {len(arr)} tirages valides.")
        ci[k + "_ci_low"], ci[k + "_ci_high"] = np.percentile(arr, [2.5, 97.5])
    ci["bootstrap_valid"] = min(len(v) for v in draws.values())
    return ci


def analyse_metric(years, aed, values, metric, detrended=False):
    mask = np.isfinite(aed) & np.isfinite(values)
    x = np.asarray(aed)[mask]
    y = np.asarray(values)[mask]
    yr = np.asarray(years)[mask]

    if detrended:
        x = linear_detrend(x)
        y = linear_detrend(y)

    stat = statistic_bundle(x, y)
    ci = bootstrap_ci(x, y)
    stat.update(ci)
    stat.update({
        "metric": metric,
        "detrended": detrended,
        "start_year": int(np.min(yr)),
        "end_year": int(np.max(yr)),
        "region": REGION_NAME,
        "block_length_years": BLOCK_LENGTH,
        "n_boot": N_BOOT,
    })
    return stat


# ============================================================
# LECTURE ET CALCUL AED
# ============================================================
ds = xr.open_dataset(ERA5_FILE, decode_times=True)

slp = normalize_lon((ds["slp"] / 100.0).where(ds["slp"] > -1e20))
hs = normalize_lon(ds["swh"].where(ds["swh"] > -1e20))

slp = slp.sel(
    lon=slice(EOF_LON[0], EOF_LON[1]),
    lat=slice(EOF_LAT[0], EOF_LAT[1]),
)

slp_djf = slp.where(slp.time.dt.month.isin([12, 1, 2]), drop=True)
clim_month = slp_djf.groupby("time.month").mean("time")
slp_anom = slp_djf.groupby("time.month") - clim_month
slp_anom = slp_anom.interpolate_na("time").fillna(0)

eof, pc, var_frac = compute_eof(slp_anom, n_modes=3)
pc3_monthly = pc.sel(mode=3)

# Orientation du signe par le gradient physique AED
west = spatial_mean(slp_anom, WEST_BOX)
east = spatial_mean(slp_anom, EAST_BOX)
dslp_monthly = west - east
sign_corr = np.corrcoef(pc3_monthly.values, dslp_monthly.values)[0, 1]
if sign_corr < 0:
    pc3_monthly = -pc3_monthly
    eof.loc[dict(mode=3)] = -eof.sel(mode=3)

aed_djf = to_djf_yearly(pc3_monthly, how="mean")
aed_djf = xr.DataArray(
    standardize(aed_djf.values),
    dims=("season_year",),
    coords={"season_year": aed_djf.season_year},
    name="AED_standardized",
)

# ============================================================
# VARIABLES DE VAGUES
# ============================================================
metrics = {}

# Hs moyen régional mensuel, puis moyen DJF
hs_regional = spatial_mean(hs, WAVE_BOX)
metrics["Hs_mean"] = to_djf_yearly(hs_regional, how="mean")

# Quantiles de Hs, déjà calculés dans les fichiers fournis
for metric, filename in PERCENTILE_FILES.items():
    path = Path(filename)
    if not path.exists():
        print(f"AVERTISSEMENT: {filename} absent; {metric} sera ignoré.")
        continue
    dqp = xr.open_dataset(path, decode_times=True)
    q = normalize_lon(dqp["swh"].where(dqp["swh"] > -1e20))
    q_regional = spatial_mean(q, WAVE_BOX)
    metrics[metric] = to_djf_yearly(q_regional, how="mean")

# ============================================================
# ALIGNEMENT, ANALYSES ET EXPORT
# ============================================================
series = [aed_djf.rename("AED")] + [v.rename(k) for k, v in metrics.items()]
aligned = xr.align(*series, join="inner")
aed_aligned = aligned[0]

yearly = pd.DataFrame({
    "year": aed_aligned.season_year.values.astype(int),
    "AED": aed_aligned.values,
})
for i, key in enumerate(metrics, start=1):
    yearly[key] = aligned[i].values

yearly.to_csv(OUTDIR / "aed_wave_yearly_DJF.csv", index=False)

rows = []
for metric in metrics:
    for detrended_flag in (False, True):
        rows.append(
            analyse_metric(
                yearly["year"].values,
                yearly["AED"].values,
                yearly[metric].values,
                metric,
                detrended=detrended_flag,
            )
        )

results = pd.DataFrame(rows)
results.to_csv(OUTDIR / "aed_wave_quantification.csv", index=False)

# ============================================================
# FIGURE DE SYNTHESE
# ============================================================
plot_metrics = list(metrics.keys())
fig, axes = plt.subplots(len(plot_metrics), 2,
                         figsize=(11, 4 * len(plot_metrics)),
                         squeeze=False)

for i, metric in enumerate(plot_metrics):
    x = yearly["AED"].values
    y = yearly[metric].values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    # Régression
    ax = axes[i, 0]
    ax.scatter(x, y, s=28, alpha=0.75)
    reg = linregress(x, y)
    xx = np.linspace(np.min(x), np.max(x), 100)
    ax.plot(xx, reg.intercept + reg.slope * xx, linewidth=2)
    ax.axvline(-1, linestyle="--", linewidth=1)
    ax.axvline(1, linestyle="--", linewidth=1)
    ax.set_xlabel("AED (standard deviations)")
    ax.set_ylabel(f"{metric} (m)")
    ax.set_title(f"{metric}: slope = {reg.slope:.3f} m per +1σ, r = {reg.rvalue:.2f}")

    # Composites
    ax = axes[i, 1]
    groups = [
        y[x < -1],
        y[np.abs(x) <= 0.5],
        y[x > 1],
    ]
    ax.boxplot(groups, labels=["AED < -1σ", "|AED| ≤ 0.5σ", "AED > +1σ"],
               showmeans=True)
    ax.set_ylabel(f"{metric} (m)")
    ax.set_title(f"{metric}: phase composites")

plt.tight_layout()
plt.savefig(OUTDIR / "aed_wave_quantification.png", dpi=300)
plt.close()

# ============================================================
# RESUME TEXTE
# ============================================================
print("\n=== RESULTATS PRINCIPAUX, SERIE BRUTE ===")
raw = results[results["detrended"] == False]
for _, r in raw.iterrows():
    print(
        f"{r['metric']}: +1 SD d'AED = {r['slope_m_per_1sd']:.3f} m "
        f"[IC95% {r['slope_m_per_1sd_ci_low']:.3f}, "
        f"{r['slope_m_per_1sd_ci_high']:.3f}]; "
        f"AED > +1σ vs AED < -1σ = "
        f"{r['diff_positive_negative_m']:.3f} m, "
        f"soit {r['increase_positive_vs_negative_pct']:.1f}% "
        f"[IC95% {r['increase_positive_vs_negative_pct_ci_low']:.1f}, "
        f"{r['increase_positive_vs_negative_pct_ci_high']:.1f}]."
    )

print("\nFichiers produits:")
print(OUTDIR / "aed_wave_yearly_DJF.csv")
print(OUTDIR / "aed_wave_quantification.csv")
print(OUTDIR / "aed_wave_quantification.png")
print(f"\nEOF3 explique {var_frac[2]:.2f}% de la variance DJF.")
print(f"Corrélation PC3–ΔSLP mensuelle après orientation: {abs(sign_corr):.3f}")
