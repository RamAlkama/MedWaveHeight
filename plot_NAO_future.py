import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.stats import gaussian_kde

# ==============================
# 1. PARAMÈTRES GÉNÉRAUX
# ==============================
SSP = "245" # "585"  "370"  "245"
index = "AED" #"NAO"  "AED" "WEPA" "Med"
smooth = 1

ref_start, ref_end = "1980", "2010"   # période de référence P90 FIXE

path = f"/data/ralkama/CMIP6/psl_hist_ssp585/SSP{SSP}/*.nc"
files = sorted(glob.glob(path))

varname = "psl"
time_start, time_end = '1971','2100'#"1850", "2100"   #'1971','2100'

# ==============================
# 2. DOMAINES SPATIAUX
# ==============================
if index=='NAO': 
    lat_Azores = slice(36, 40) 
    lon_Azores = slice(-28, -20) 
    lat_Iceland = slice(63, 70) 
    lon_Iceland = slice(-25, -16) 
elif index=='WEPA': 
    lat_Azores = slice(22, 36) 
    lon_Azores = slice(-27, 0) 
    lat_Iceland = slice(47, 61) 
    lon_Iceland = slice(-21, 21) 
elif index=='Med': 
    lat_Azores = slice(46, 54) 
    lon_Azores = slice(16, 24) 
    lat_Iceland = slice(46, 54) 
    lon_Iceland = slice(-29, -21) 
elif index=='AED': 
    lat_Azores = slice(45, 65) 
    lon_Azores = slice(-25, -20)#(-30, -25) 
    lat_Iceland = slice(45, 65) 
    lon_Iceland = slice(10, 15) 
else: 
    raise ValueError("Index non défini ici")

time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)

# ==============================
# 3. FONCTION INDEX
# ==============================
def compute_index(ds, var):
    A = ds[var].sel(lat=lat_Azores, lon=lon_Azores).mean(dim=("lat", "lon"))
    B = ds[var].sel(lat=lat_Iceland, lon=lon_Iceland).mean(dim=("lat", "lon"))
    return A - B

# ==============================
# 4. LECTURE DES MODÈLES
# ==============================
nao_list, model_names = [], []
ntime_ref = None

for f in files:
    ds = xr.open_dataset(f, decode_times=time_coder)
    ds = ds.sel(time=slice(time_start, time_end))
    dsref = ds.sel(time=slice(time_start, ref_end))

    idx = compute_index(ds, varname)
    idx_ref = compute_index(dsref, varname).mean()
    idx = idx - idx_ref

    if ntime_ref is None:
        ntime_ref = idx.sizes["time"]

    if idx.sizes["time"] == ntime_ref:
        nao_list.append(idx)
        model_names.append(f.split("/")[-1])
    else:
        print (f, 'corrupted, not used')

nao_all = xr.concat(nao_list, dim="model")
nao_all = nao_all.assign_coords(model=("model", model_names))


file_era5 = "/data/ralkama/CMIP6/era5_195001_202412.nc"
var_era5 = "slp"

ds_era5 = xr.open_dataset(file_era5, decode_times=time_coder)
nao_era5_mon = compute_index(ds_era5, var_era5)
month = nao_era5_mon["time.month"]
year = nao_era5_mon["time.year"]

year_djfm = xr.where(month == 12, year + 1, year)
djfm_mask = month.isin([12, 1, 2, 3])

nao_era5_djfm = (
    nao_era5_mon
    .where(djfm_mask, drop=True)
    .assign_coords(
        year=("time", year_djfm[djfm_mask].data)  # 
    )
    .groupby("year")
    .mean(dim="time")
)

nao_era5_djfm = nao_era5_djfm.sel(year=slice(max(1951,int(time_start)), min(2024,int(time_end))))
nao_era5_roll = nao_era5_djfm.rolling(year=smooth, center=True).mean()/100.
nao_era5_roll= (nao_era5_roll -nao_era5_roll.mean(dim="year"))/np.std(nao_era5_roll)

years_era5 = nao_era5_roll.year.values

# ==============================
# 5. LISSAGE + hPa
# ==============================
nao = nao_all.rolling(time=smooth, center=True).mean() / 100.0

# ==============================
# 6. STATISTIQUES
# ==============================
nao_mean = nao.mean("model")
nao_q1 = nao.quantile(0.25, "model")
nao_q3 = nao.quantile(0.75, "model")
years = np.array([t.year for t in nao.time.values])

# ==============================
# 7. PÉRIODES D’ANALYSE
# ==============================
periods = {
    "1980-2010": ("black", "1980", "2010"),
    "2010-2040": ("orange", "2010", "2040"),
    "2040-2070": ("red", "2040", "2070"),
    "2070-2100": ("brown", "2070", "2100"),
}

# ==============================
# 8. SEUIL FIXE P90 (RÉFÉRENCE)
# ==============================
ref = nao.sel(time=slice(ref_start, ref_end))
p90_ref = ref.quantile(0.90, dim="time")

# ==============================
# 9. CONSTRUCTION DES DIAGNOSTICS
# ==============================
pdf_all = {}
pdf_ext = {}
freq_ext = {}
box_ext = {}

for label, (color, t1, t2) in periods.items():
    data = nao.sel(time=slice(t1, t2))

    # --- PDF totale ---
    vals = data.values.flatten()
    pdf_all[label] = (color, vals[~np.isnan(vals)])

    # --- Extrêmes seuil FIXE ---
    ext_vals, freq = [], []

    for m in data.model.values:
        dm = data.sel(model=m)
        thr = p90_ref.sel(model=m)
        v = dm.where(dm > thr)
        ext_vals.extend(v.values[~np.isnan(v.values)])
        freq.append(float(v.count() / dm.count() * 100))

    pdf_ext[label] = (color, np.array(ext_vals))
    freq_ext[label] = (color, np.mean(freq))
    box_ext[label] = np.array(ext_vals)

# ==============================
# 10. FIGURE PAPIER-READY
# ==============================
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10})

fig = plt.figure(figsize=(10, 12))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

ax_ts   = fig.add_subplot(gs[0, :])
ax_pdf  = fig.add_subplot(gs[1, 0])
ax_ext  = fig.add_subplot(gs[1, 1])
ax_freq = fig.add_subplot(gs[2, :])
#ax_box  = fig.add_subplot(gs[3:, :])

# --- (A) Série temporelle ---
ax_ts.plot(years, nao_mean, lw=2, label=f"{len(nao_list)} CMIP6 members mean")
ax_ts.plot(years_era5, nao_era5_roll, color="green", lw=1, label="ERA5")

ax_ts.fill_between(years, nao_q1, nao_q3, alpha=0.3, label="Q1-Q3 spread")
ax_ts.axhline(0, ls="--", c="gray")
ax_ts.set_ylabel(f"{index} (hPa)")
ax_ts.set_title("(a) CMIP6 multi-model mean")
ax_ts.legend()

# --- (B) PDF totale ---
x = np.linspace(-5, 5, 500)
for label, (c, d) in pdf_all.items():
    ax_pdf.plot(x, gaussian_kde(d)(x), c=c, label=label)
ax_pdf.set_title("(b) PDF - all values")
ax_pdf.axvline(0, ls="--", c="gray")
#ax_pdf.legend()
ax_pdf.set_xlabel(f"{index} (hPa)")
ax_pdf.set_ylabel("Density")

# --- (C) PDF extrêmes ---
# --- (C) Boxplot des extrêmes > P90 ---
labels = list(box_ext.keys())
data_box = [box_ext[k] for k in labels]
colors = [periods[k][0] for k in labels]

bp = ax_ext.boxplot(
    data_box,
    patch_artist=True,
    showfliers=True,
    medianprops=dict(color="black"),
    boxprops=dict(linewidth=1.2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2)
)

# Couleur des boîtes
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)

ax_ext.set_xticklabels(labels, rotation=20)
ax_ext.set_ylabel(f"{index} (hPa)")
ax_ext.set_title("(c) Extremes > P90 (ref 1980–2010)")
ax_ext.axhline(0, ls="--", c="gray", lw=0.8)
ax_ext.grid(True)

# --- (D) Fréquence ---
labels = list(freq_ext.keys())
vals = [freq_ext[k][1] for k in labels]
cols = [freq_ext[k][0] for k in labels]
ax_freq.bar(labels, vals, color=cols)
ax_freq.set_ylabel("Frequency (%)")
ax_freq.set_title("(d) Frequency of extremes")

plt.tight_layout()
plt.show()

