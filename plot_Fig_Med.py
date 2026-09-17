#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
from scipy.signal import detrend
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import pandas as pd
from matplotlib.patches import Rectangle

# ============================================================
# PARAMÈTRES
# ============================================================
infile = "era5_195001_202412_all.nc"
outdir = "figures"
os.makedirs(outdir, exist_ok=True)

#=== BOITES PRESSION (EOF3 DJF) ===
#MAX : lon=-34.88, lat=52.38
lonmax,latmax=(-24.5,-20),(45,62)#(-30,-25),(45,62)
#MIN : lon=12.12, lat=54.38
lonmin,latmin=(10,15),(45,62)


lon_corse = (8, 10)
lat_corse = (41, 44)

lon_bounds = (-25.5, 20)#(-30.5,20)#(-25, 15)
lat_bounds = (20, 65)# (20,65)

n_modes = 3
seasons = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}
season_order = ["DJF", "MAM", "JJA", "SON"]
seas_list={"DJF":"Winter","MAM":"Spring","JJA":"Summer","SON":"Autumn"}
# ============================================================
# FONCTIONS UTILES
# ============================================================
def normalize_lon(da):
    lon = da["lon"]
    if lon.max() > 180:
        lon = ((lon + 180) % 360) - 180
        da = da.assign_coords(lon=lon).sortby("lon")
    return da

def safe_detrend(ts):
    ts = np.asarray(ts)
    mask = np.isfinite(ts)
    if mask.sum() < 10:
        return None
    out = np.full_like(ts, np.nan, dtype=float)
    out[mask] = detrend(ts[mask])
    return out

def compute_eof(slp):
    weights = np.sqrt(np.cos(np.deg2rad(slp.lat)))
    slp_w = slp * weights

    slp_2d = slp_w.stack(space=("lat", "lon")).dropna("space")
    A = slp_2d.transpose("time", "space").values

    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    # Variance expliquée
    var_frac = (s**2) / np.sum(s**2) * 100

    eof = xr.DataArray(
        (Vt[:n_modes] * s[:n_modes, None]),
        dims=("mode", "space"),
        coords={"mode": np.arange(1, n_modes + 1), "space": slp_2d.space},
    ).unstack("space")

    eof = eof / weights
    pc = xr.DataArray(
        U[:, :n_modes] * s[:n_modes],
        dims=("time", "mode"),
        coords={"time": slp.time, "mode": np.arange(1, n_modes + 1)},
    )

    return eof, pc, var_frac[:n_modes]

def enforce_same_sign(eof_ref, pc_ref, eof, pc):
    r = pearsonr(
        eof_ref.values.flatten(),
        eof.values.flatten()
    )[0]
    if r < 0:
        eof *= -1
        pc *= -1
    return eof, pc

def eof_to_wind(eof_field):
    """
    Convertit un champ EOF de SLP en vent géostrophique.
    eof_field : xr.DataArray [lat, lon]
    """
    Omega = 7.2921e-5
    R = 6371000

    lat = eof_field.lat.values
    lon = eof_field.lon.values

    phi = np.deg2rad(lat)  # 1D
    f = 2 * Omega * np.sin(phi)  # 1D [lat]

    # dx et dy en mètres
    dlat = np.gradient(lat) * np.pi / 180 * R  # 1D [lat]
    dlon = np.gradient(lon) * np.pi / 180 * R  # 1D [lon]

    # Créer des grilles 2D pour le calcul
    dlon2d = np.tile(dlon, (len(lat), 1))          # shape [lat, lon]
    dlat2d = np.tile(dlat[:, None], (1, len(lon))) # shape [lat, lon]
    f2d = np.tile(f[:, None], (1, len(lon)))       # shape [lat, lon]

    # Calcul des gradients SLP
    dpdy, dpdx = np.gradient(eof_field.values)  # dpdy, dpdx = [lat, lon]

    # Convertir en vent géostrophique
    u = -dpdy / (dlat2d * f2d)  # zonal
    v = dpdx / (dlon2d * f2d)   # meridional

    return xr.DataArray(u, coords=eof_field.coords, dims=eof_field.dims), \
           xr.DataArray(v, coords=eof_field.coords, dims=eof_field.dims)

# ============================================================
# LECTURE DONNÉES
# ============================================================
ds = xr.open_dataset(infile, decode_times=True)

slp = ds["slp"] / 100  # Pa → hPa
u10 = ds["u_wind"]
v10 = ds["v_wind"]
Hs = ds["swh"]  # ["shts" ["swh"]
wind = ds["wind"]
pr=ds["tp"]
tas=ds["tas"]

# Nettoyage FillValue
slp = slp.where(slp > -1e20)
Hs = Hs.where(Hs > -1e20)
wind = wind.where(wind > -1e20)
pr= pr.where(pr > -1e20)
tas= tas.where(tas > -1e20)

slp = normalize_lon(slp)
u10 = normalize_lon(u10)
v10 = normalize_lon(v10)
Hs = normalize_lon(Hs)
wind = normalize_lon(wind)
tas = normalize_lon(tas)
pr = normalize_lon(pr)

slp = slp.sel(lon=slice(*lon_bounds), lat=slice(*lat_bounds))
u10 = u10.sel(lon=slice(*lon_bounds), lat=slice(*lat_bounds))
v10 = v10.sel(lon=slice(*lon_bounds), lat=slice(*lat_bounds))
Hs = Hs.sel(lon=slice(*lon_bounds), lat=slice(*lat_bounds))
wind=wind.sel(lon=slice(*lon_bounds), lat=slice(*lat_bounds))
tas=tas.sel(lon=slice(*lon_bounds), lat=slice(*lat_bounds))
pr=pr.sel(lon=slice(*lon_bounds), lat=slice(*lat_bounds))

# ============================================================
# EOF PAR SAISON + COHÉRENCE DES SIGNES
# ============================================================
results = {}
ref_eof = {}
var_explained = {}

for seas in season_order:
    ds_seas = slp.sel(time=slp.time.dt.month.isin(seasons[seas]))
    clim = ds_seas.groupby("time.month").mean("time")
    anom = ds_seas.groupby("time.month") - clim
    anom = anom.interpolate_na("time").fillna(0)

    eof, pc, var_frac = compute_eof(anom)

    if seas == "DJF":
        ref_eof = eof.copy()
    else:
        for m in range(n_modes):
            eof[m], pc[:, m] = enforce_same_sign(
                ref_eof[m], None, eof[m], pc[:, m]
            )

    results[seas] = {"eof": eof, "pc": pc, "SLP":anom}
    var_explained[seas] = var_frac

# ============================================================
# FIGURE 1 : EOFs + VENTS GÉOSTROPHIQUES
# ============================================================
fig, axes = plt.subplots(
    4, 3, figsize=(16, 18),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

for i, seas in enumerate(season_order):
    eof = results[seas]["eof"]

    for j in range(3):
        ax = axes[i, j]
        field = eof.isel(mode=j)

        # Vent géostrophique associé à ce mode EOF
        u, v = eof_to_wind(field)
        vmax = np.nanmax(np.abs(field))
        cf = ax.contourf( field.lon, field.lat, field, levels=np.linspace(-vmax, vmax, 21), cmap="RdBu_r", extend="both" )
        ax.quiver( field.lon[::5], field.lat[::5], u.values[::5, ::5], v.values[::5, ::5], scale=10)#, width=0.002        )
        if j==2:
            ax.add_patch(Rectangle((lonmin[0] , latmin[0]), lonmin[1]-lonmin[0], latmin[1]-latmin[0], edgecolor="orange", facecolor="none", linewidth=2))
            ax.add_patch(Rectangle((lonmax[0] , latmax[0]), lonmax[1]-lonmax[0], latmax[1]-latmax[0], edgecolor="orange", facecolor="none", linewidth=2))
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)
        ax.set_title(f"{seas} – EOF{j+1} ({var_explained[seas][j]:.1f}%)", fontsize=16,  fontweight="bold")
        plt.colorbar(cf, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig(f"{outdir}/EOFs_3modes_4saisons_geostrophic.png", dpi=150)
plt.close()

# ============================================================
# FIGURE 2 : CORR PC–Hs
# ============================================================
lon_bounds = (-9, 15)
lat_bounds = (30, 45)
Hs = Hs.sel(lon=slice(*lon_bounds), lat=slice(*lat_bounds))

fig, axes = plt.subplots(
    4, 3, figsize=(12, 12),
    subplot_kw={"projection": ccrs.PlateCarree()})

best_point = None

for i, seas in enumerate(season_order):
    for j, mode in enumerate([1, 2, 3]):
        ax = axes[i, j]

        pc = results[seas]["pc"].sel(mode=mode)
        var = Hs.sel(time=Hs.time.dt.month.isin(seasons[seas]))

        corr = np.full((len(var.lat), len(var.lon)), np.nan)
        pval = corr.copy()

        for ii in range(len(var.lat)):
            for jj in range(len(var.lon)):
                ts = safe_detrend(var[:, ii, jj])
                if ts is None:
                    continue
                mask = np.isfinite(ts) & np.isfinite(pc)
                if mask.sum() < 10:
                    continue
                corr[ii, jj], pval[ii, jj] = pearsonr(ts[mask], pc.values[mask])

        if seas == "DJF" and mode==3:
            sub=xr.DataArray(corr, coords={"lat": var.lat, "lon": var.lon}, dims=("lat", "lon"),   )
            sub = sub.where((sub.lon >= lon_corse[0]) & (sub.lon <= lon_corse[1]) &
                            (sub.lat >= lat_corse[0]) & (sub.lat <= lat_corse[1]))
            iy, ix = np.unravel_index(np.nanargmax(np.abs(sub.values)), sub.shape)
            best_point = (float(sub.lon[ix]), float(sub.lat[iy]))


        cf = ax.contourf( var.lon, var.lat, corr, levels=np.linspace(-0.8, 0.8, 17), cmap="RdBu_r", extend="both" )

        #sig = np.where(pval < 0.05)
        #sig = (sig[0][::2], sig[1][::2])
        #ax.scatter(var.lon.values[sig[1]], var.lat.values[sig[0]], s=1, c="k", alpha=0.3 )
        # masque zones non significatives
        nonsig = pval >= 0.05
        ax.contourf(var.lon, var.lat, nonsig, levels=[0.5, 1],hatches=["///"],  
                    colors="lightgrey", alpha=0.3, transform=ccrs.PlateCarree())

        ax.coastlines()
        if i==0 :
            ax.set_title(f"EOF{mode}",  fontsize=15,  fontweight="bold")#{seas} – EOF{mode}")
        if j==0:
            season=seas_list[seas]
            ax.text( -0.07, 0.2,    f"{season}-{seas}", transform=ax.transAxes,  fontsize=15,  fontweight="bold" ,rotation=90)

        if seas == "DJF" and mode==3:
            ax.scatter( best_point[0], best_point[1], c="lime", s=70, edgecolor="k", zorder=10)

cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.02])
cb = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal', label='Correlation PC3-Hs')
cb.ax.tick_params(labelsize=15)
cb.set_label('Correlation PCs-Hs', fontsize=15, fontweight='bold')

plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig(f"{outdir}/Corr_PC_Hs_3modes_4saisons_horizontal.png", dpi=150)
plt.close()




# ============================================================
# FIGURE 2bis : CORR PC–Hs_percentile90 95 99
# ============================================================
lon_bounds = (-9, 15)
lat_bounds = (30, 45)

for perc in [90,95,99]:
    dsss = xr.open_dataset(f"p{perc}_swh_195001_202412_grid.nc", decode_times=True)
    print(dsss["time"].values)
    print("Début :", dsss["time"].values[0])
    print("Fin   :", dsss["time"].values[-1])
    print("Nombre:", len(dsss["time"]))
    Hss = dsss["swh"].sel(time=slice("1950-01-01", "2024-12-31"))  # ["shts" ["swh"]
    Hss = Hss.where(Hss > -1e20)
    Hss = normalize_lon(Hss)
    Hss = Hss.sel(lon=slice(*lon_bounds), lat=slice(*lat_bounds))

    fig, axes = plt.subplots(
        4, 3, figsize=(12, 12),
        subplot_kw={"projection": ccrs.PlateCarree()})

    best_point = None

    for i, seas in enumerate(season_order):
        for j, mode in enumerate([1, 2, 3]):
            ax = axes[i, j]

            pc = results[seas]["pc"].sel(mode=mode)
            var = Hss.sel(time=Hss.time.dt.month.isin(seasons[seas]))

            corr = np.full((len(var.lat), len(var.lon)), np.nan)
            pval = corr.copy()

            for ii in range(len(var.lat)):
                for jj in range(len(var.lon)):
                    ts = safe_detrend(var[:, ii, jj])
                    if ts is None:
                        continue
                    mask = np.isfinite(ts) & np.isfinite(pc)
                    if mask.sum() < 10:
                        continue
                    corr[ii, jj], pval[ii, jj] = pearsonr(ts[mask], pc.values[mask])

            if False: #seas == "DJF" and mode==3:
                sub=xr.DataArray(corr, coords={"lat": var.lat, "lon": var.lon}, dims=("lat", "lon"),   )
                sub = sub.where((sub.lon >= lon_corse[0]) & (sub.lon <= lon_corse[1]) &
                            (sub.lat >= lat_corse[0]) & (sub.lat <= lat_corse[1]))
                iy, ix = np.unravel_index(np.nanargmax(np.abs(sub.values)), sub.shape)
                best_point = (float(sub.lon[ix]), float(sub.lat[iy]))

            cf = ax.contourf( var.lon, var.lat, corr, levels=np.linspace(-0.8, 0.8, 17), cmap="RdBu_r", extend="both" )

            #sig = np.where(pval < 0.05)
            #sig = (sig[0][::2], sig[1][::2])
            #ax.scatter(var.lon.values[sig[1]], var.lat.values[sig[0]], s=1, c="k", alpha=0.3 )
            # masque zones non significatives
            nonsig = pval >= 0.05
            ax.contourf(var.lon, var.lat, nonsig, levels=[0.5, 1],hatches=["///"],  
                    colors="lightgrey", alpha=0.3, transform=ccrs.PlateCarree())

            ax.coastlines()
            if i==0 :
                ax.set_title(f"EOF{mode}",  fontsize=15,  fontweight="bold")#{seas} – EOF{mode}")
            if j==0:
                season=seas_list[seas]
                ax.text( -0.07, 0.2,    f"{season}-{seas}", transform=ax.transAxes,  fontsize=15,  fontweight="bold" ,rotation=90)

            if False :#seas == "DJF" and mode==3:
                ax.scatter( best_point[0], best_point[1], c="lime", s=70, edgecolor="k", zorder=10)

    cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.02])
    cb = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal', label=f'Correlation PC3-P{perc}(Hs)')
    cb.ax.tick_params(labelsize=15)
    cb.set_label(f'Correlation PCs-P{perc}(Hs)', fontsize=15, fontweight='bold')

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(f"{outdir}/Corr_PC_Hs_3modes_4saisons_horizontal_P{perc}.png", dpi=150)
    plt.close()

# ============================================================
# FIGURE 3 : CORR PC–wind 
# ============================================================
lon_bounds = (-9, 15)
lat_bounds = (30, 55)
wind = wind.sel(lon=slice(*lon_bounds), lat=slice(*lat_bounds))

fig, axes = plt.subplots(
    4, 3, figsize=(10, 12),
    subplot_kw={"projection": ccrs.PlateCarree()})

best_point = None

for i, seas in enumerate(season_order):
    for j, mode in enumerate([1, 2, 3]):
        ax = axes[i, j]

        pc = results[seas]["pc"].sel(mode=mode)
        var = wind.sel(time=wind.time.dt.month.isin(seasons[seas]))

        corr = np.full((len(var.lat), len(var.lon)), np.nan)
        pval = corr.copy()

        for ii in range(len(var.lat)):
            for jj in range(len(var.lon)):
                ts = safe_detrend(var[:, ii, jj])
                if ts is None:
                    continue
                mask = np.isfinite(ts) & np.isfinite(pc)
                if mask.sum() < 10:
                    continue
                corr[ii, jj], pval[ii, jj] = pearsonr(ts[mask], pc.values[mask])

        if seas == "DJF" and mode==3:
            sub=xr.DataArray(corr, coords={"lat": var.lat, "lon": var.lon}, dims=("lat", "lon"),   )
            sub = sub.where((sub.lon >= lon_corse[0]) & (sub.lon <= lon_corse[1]) &
                            (sub.lat >= lat_corse[0]) & (sub.lat <= lat_corse[1]))
            iy, ix = np.unravel_index(np.nanargmax(np.abs(sub.values)), sub.shape)
            best_point = (float(sub.lon[ix]), float(sub.lat[iy]))


        cf = ax.contourf( var.lon, var.lat, corr, levels=np.linspace(-0.8, 0.8, 17), cmap="RdBu_r", extend="both" )

        #sig = np.where(pval < 0.05)
        #sig = (sig[0][::4], sig[1][::4])
        #ax.scatter(var.lon.values[sig[1]], var.lat.values[sig[0]], s=1, c="k", alpha=0.3 )
        nonsig = pval >= 0.05
        ax.contourf(var.lon, var.lat, nonsig, levels=[0.5, 1],hatches=["///"],  
                    colors="lightgrey", alpha=0.3, transform=ccrs.PlateCarree())

        ax.coastlines()
        #ax.set_title(f"{seas} – EOF{mode}")
        if i==0 :
            ax.set_title(f"PC{mode}",  fontsize=15,  fontweight="bold")#{seas} – EOF{mode}")
        if j==0:
            season=seas_list[seas]
            ax.text( -0.15, 0.2,    f"{season}-{seas}", transform=ax.transAxes,  fontsize=15,  fontweight="bold" ,rotation=90)

cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.02])
#norm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-0.8, vmax=0.8))
#norm.set_array([])
fig.colorbar(cf, cax=cbar_ax, orientation='horizontal', label='Correlation')
cb = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal', label='Correlation PC3-Wind speed')
cb.ax.tick_params(labelsize=15)
cb.set_label('Correlation PCs-wind speed', fontsize=15, fontweight='bold')

plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig(f"{outdir}/Corr_PC_Wind_3modes_4saisons_horizontal.png", dpi=150)
plt.close()

# ============================================================
# FIGURE 4 : CORR PC3–(tas-pr-wind) 
# ============================================================
lon_bounds = (-9, 15)
lat_bounds = (30, 55)
seas_list={"DJF":"Winter","MAM":"Spring","JJA":"Summer","SON":"Autumn"}
varss={0:"Temperature",1:"Precipitation",2:"Wind"}
wind = wind.sel(lon=slice(*lon_bounds), lat=slice(*lat_bounds))
pr = pr.sel(lon=slice(*lon_bounds), lat=slice(*lat_bounds))
tas = tas.sel(lon=slice(*lon_bounds), lat=slice(*lat_bounds))

fig, axes = plt.subplots(
    4, 3, figsize=(10.05, 12),
    subplot_kw={"projection": ccrs.PlateCarree()})

best_point = None

for i, seas in enumerate(season_order):
    mode=3
    for j, vartmp in enumerate([tas, pr, wind]):
        ax = axes[i, j]

        pc = results[seas]["pc"].sel(mode=mode)
        var = vartmp.sel(time=wind.time.dt.month.isin(seasons[seas]))

        corr = np.full((len(var.lat), len(var.lon)), np.nan)
        pval = corr.copy()

        for ii in range(len(var.lat)):
            for jj in range(len(var.lon)):
                ts = safe_detrend(var[:, ii, jj])
                if ts is None:
                    continue
                mask = np.isfinite(ts) & np.isfinite(pc)
                if mask.sum() < 10:
                    continue
                corr[ii, jj], pval[ii, jj] = pearsonr(ts[mask], pc.values[mask])

        if seas == "DJF" and mode==3:
            sub=xr.DataArray(corr, coords={"lat": var.lat, "lon": var.lon}, dims=("lat", "lon"),   )
            sub = sub.where((sub.lon >= lon_corse[0]) & (sub.lon <= lon_corse[1]) &
                            (sub.lat >= lat_corse[0]) & (sub.lat <= lat_corse[1]))
            iy, ix = np.unravel_index(np.nanargmax(np.abs(sub.values)), sub.shape)
            best_point = (float(sub.lon[ix]), float(sub.lat[iy]))

        cf = ax.contourf( var.lon, var.lat, corr, levels=np.linspace(-0.8, 0.8, 17), cmap="RdBu_r", extend="both" )

        #sig = np.where(pval < 0.05)
        #sig = (sig[0][1::4], sig[1][1::4])
        #ax.scatter(var.lon.values[sig[1]], var.lat.values[sig[0]], s=1, c="k", alpha=0.3 )
        nonsig = pval >= 0.05
        ax.contourf(var.lon, var.lat, nonsig, levels=[0.5, 1],hatches=["///"],  
                    colors="lightgrey", alpha=0.3, transform=ccrs.PlateCarree())
        
        ax.coastlines()
        if i==0 :
            variable=varss[j]
            ax.set_title(f"{variable}",  fontsize=15,  fontweight="bold")#{seas} – EOF{mode}")
        if j==0:
            season=seas_list[seas]
            ax.text( -0.15, 0.2,    f"{season}-{seas}", transform=ax.transAxes,  fontsize=15,  fontweight="bold" ,rotation=90)

cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.02])
#norm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-0.8, vmax=0.8))
#norm.set_array([])
cb = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal', label='Correlation PC3-variable')
cb.ax.tick_params(labelsize=15)
cb.set_label('Correlation PC3-variable', fontsize=15, fontweight='bold')


plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig(f"{outdir}/Corr_PC3_Tas_Pr_Wind_4saisons.png", dpi=150)
plt.close()

# ============================================================
# FIGURE 5 — Séries temporelles Corse
# ============================================================
lon0, lat0 = best_point

def standardize(x):
    return (x - x.mean()) / x.std()

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
label = ord("a")

for i, seas in enumerate(season_order):
    ax = axes[i]

    pc = results[seas]["pc"].sel(mode=3)
    ts = Hs.sel(lon=lon0, lat=lat0, method="nearest")
    ts = ts.sel(time=ts.time.dt.month.isin(seasons[seas]))
    pc_s = pc.sel(time=ts.time)

    SLP = results[seas]["SLP"]

    pmax = SLP.sel(
        lon=slice(lonmax[0], lonmax[1]),
        lat=slice(latmax[0], latmax[1])).mean(dim=("lat", "lon"))

    pmin = SLP.sel(
        lon=slice(lonmin[0], lonmin[1]),
        lat=slice(latmin[0], latmin[1])).mean(dim=("lat", "lon"))

    dp = standardize(pmax.resample(time="YE").mean() - pmin.resample(time="YE").mean())

    # --- dataframe ---
    df = pd.DataFrame({
            "time": pd.to_datetime(results[seas]["pc"].time.values),
            "PC": pc.values,
            "Hs": ts.values     }).set_index("time")

    # --- annual mean ---
    df_annual = df.resample("YE").mean()
    years = df_annual.index.year
    pc_plot = standardize(df_annual["PC"])
    # --- colors ---
    colors = np.where(pc_plot.values >= 0, "olive", "green")

    # --- bars (PC) ---
    ax.bar( years,   pc_plot.values, color=colors,    width=0.8,  label=f"PC3"  ,alpha=0.3  )
    ax.set_ylabel("PC3 and ΔSLP", color="tab:green",fontsize=15,  fontweight="bold")
    #ax.plot(pc_s.time, pc_s, "o", color="tab:green", label="PC3 (hPa)")
    #ax.plot(years,df_annual["PC"].values, "o", color="tab:green", label="PC3 (hPa)")
    #ax.set_ylabel("PC3 (hPa)", color="tab:green")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.plot(years, dp.values,"o", color="tab:green", label="ΔSLP")
    ax2 = ax.twinx()
    #ax2.plot(ts.time, ts, "o", color="tab:blue", label="Hs")
    ax2.plot(years, df_annual["Hs"].values, color="tab:blue", label="Hs")
    #ax2.plot(years, df_annual["Hs"].values, "o", color="tab:blue", label="Hs")
    ax2.set_ylabel("Hs (m)", color="tab:blue",fontsize=15,  fontweight="bold")
    r_slp = pearsonr(df_annual["PC"].values, dp.values)[0]
    r = pearsonr(df_annual["PC"].values, df_annual["Hs"].values)[0]
    ax.text( 0.15, 0.85,    f"r = {r_slp:.2f}", transform=ax.transAxes,  fontsize=15,  fontweight="bold" ,color="green" )
    ax.text( 0.85, 0.85,    f"{seas}  r = {r:.2f}", transform=ax.transAxes,  fontsize=15,  fontweight="bold"  )

    ax.text( 0.02, 1.05, f"({chr(label)})", transform=ax.transAxes,    fontsize=12, fontweight="bold"    )
    label += 1

plt.tight_layout()
plt.savefig(f"{outdir}/Figure3_TimeSeries_Corse.png", dpi=300)
plt.close()


print("Finito !!!")

