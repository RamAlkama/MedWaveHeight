#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
future_AED_wave_revision.py

Version robuste, rapide et portable pour quantifier les changements futurs
conditionnels de vagues associés à l'AED.

Cette version :
- trouve automatiquement les CSV d'entrée, même s'ils sont dans un sous-dossier ;
- accepte aussi des chemins explicites en ligne de commande ;
- ne relit pas les NetCDF CMIP6 ;
- ne stocke pas tous les tirages bootstrap en mémoire ;
- applique un bootstrap hiérarchique à poids égal par modèle ;
- produit tableaux, figures et phrases pour le manuscrit.

Entrées requises
----------------
1) future_AED_member_diagnostics.csv
2) future_AED_equal_model_diagnostics.csv
3) aed_wave_quantification.csv

Exécution simple
----------------
python future_AED_wave_revision.py

Exécution avec chemins explicites
---------------------------------
python future_AED_wave_revision.py \
  --member-csv /chemin/future_AED_member_diagnostics.csv \
  --model-csv /chemin/future_AED_equal_model_diagnostics.csv \
  --era5-csv /chemin/aed_wave_quantification.csv \
  --outdir figures/future_AED_wave_revision

Important
---------
Les résultats de vagues sont des projections conditionnelles :
    Delta vague = beta_ERA5 x Delta AED_CMIP6

Elles supposent que la relation historique AED-vagues reste stationnaire.
Ce ne sont pas des simulations dynamiques directes de vagues.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIO_ORDER = ["SSP2-4.5", "SSP3-7.0", "SSP5-8.5"]
PERIOD_ORDER = ["2011–2040", "2041–2070", "2071–2100"]
METRIC_ORDER = ["Hs_mean", "Hs_P90", "Hs_P95", "Hs_P99"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Projection conditionnelle future AED-vagues."
    )
    parser.add_argument(
        "--member-csv",
        type=Path,
        default=None,
        help="Chemin vers future_AED_member_diagnostics.csv",
    )
    parser.add_argument(
        "--model-csv",
        type=Path,
        default=None,
        help="Chemin vers future_AED_equal_model_diagnostics.csv",
    )
    parser.add_argument(
        "--era5-csv",
        type=Path,
        default=None,
        help="Chemin vers aed_wave_quantification.csv",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path("."),
        help="Dossier racine pour la recherche automatique récursive.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("figures/future_AED_wave_revision"),
        help="Dossier de sortie.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=10000,
        help="Nombre de tirages bootstrap hiérarchiques.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260803,
        help="Graine aléatoire.",
    )
    return parser.parse_args()


def resolve_input(
    explicit: Path | None,
    expected_name: str,
    search_root: Path,
) -> Path:
    """
    Utilise un chemin explicite s'il est fourni.
    Sinon cherche récursivement le fichier sous search_root.
    """
    if explicit is not None:
        explicit = explicit.expanduser().resolve()
        if not explicit.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {explicit}"
            )
        return explicit

    direct = (search_root / expected_name).expanduser().resolve()
    if direct.exists():
        return direct

    matches = list(search_root.expanduser().resolve().rglob(expected_name))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Impossible de trouver '{expected_name}' sous "
            f"{search_root.resolve()}.\n"
            f"Utiliser l'option explicite correspondante."
        )

    if len(matches) > 1:
        print(
            f"Attention : plusieurs fichiers '{expected_name}' trouvés. "
            f"Le premier sera utilisé : {matches[0]}",
            file=sys.stderr,
        )

    return matches[0]


def check_columns(
    df: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(
            f"{label}: colonnes manquantes : {sorted(missing)}"
        )


def ci95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return (
        float(np.nanquantile(values, 0.025)),
        float(np.nanquantile(values, 0.975)),
    )


def hierarchical_bootstrap(
    subset: pd.DataFrame,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Bootstrap hiérarchique :
    1) tirage des modèles avec remise ;
    2) tirage d'un membre dans chaque modèle sélectionné ;
    3) moyenne à poids égal par modèle.
    """
    models = np.array(sorted(subset["model"].dropna().unique()))
    n_models = len(models)

    if n_models < 2:
        raise RuntimeError(
            "Au moins deux modèles sont requis pour le bootstrap."
        )

    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model in models:
        group = subset[subset["model"] == model]
        aed = group["delta_AED_z"].to_numpy(float)
        freq = group["future_extreme_frequency_pct"].to_numpy(float)

        mask = np.isfinite(aed) & np.isfinite(freq)
        aed = aed[mask]
        freq = freq[mask]

        if len(aed) == 0:
            raise RuntimeError(
                f"Aucune donnée valide pour le modèle {model}."
            )

        cache[str(model)] = (aed, freq)

    draws_aed = np.empty(n_boot, dtype=np.float64)
    draws_freq = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        sampled_models = rng.choice(
            models,
            size=n_models,
            replace=True,
        )

        aed_values = np.empty(n_models, dtype=np.float64)
        freq_values = np.empty(n_models, dtype=np.float64)

        for j, model in enumerate(sampled_models):
            aed, freq = cache[str(model)]
            idx = rng.integers(0, len(aed))
            aed_values[j] = aed[idx]
            freq_values[j] = freq[idx]

        draws_aed[b] = aed_values.mean()
        draws_freq[b] = freq_values.mean()

    return draws_aed, draws_freq, n_models


def format_ci(value: float, low: float, high: float, decimals: int) -> str:
    return (
        f"{value:.{decimals}f} "
        f"({low:.{decimals}f}–{high:.{decimals}f})"
    )


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    member_path = resolve_input(
        args.member_csv,
        "future_AED_member_diagnostics.csv",
        args.search_root,
    )
    model_path = resolve_input(
        args.model_csv,
        "future_AED_equal_model_diagnostics.csv",
        args.search_root,
    )
    era5_path = resolve_input(
        args.era5_csv,
        "aed_wave_quantification.csv",
        args.search_root,
    )

    print("Fichiers utilisés :")
    print(f"  Membres CMIP6 : {member_path}")
    print(f"  Modèles CMIP6 : {model_path}")
    print(f"  Relation ERA5 : {era5_path}")
    print(f"  Sorties        : {args.outdir.resolve()}")

    member_df = pd.read_csv(member_path)
    model_df = pd.read_csv(model_path)
    relation_df = pd.read_csv(era5_path)

    check_columns(
        member_df,
        {
            "scenario",
            "model",
            "member",
            "period",
            "delta_AED_z",
            "future_extreme_frequency_pct",
        },
        "member CSV",
    )
    check_columns(
        model_df,
        {
            "scenario",
            "model",
            "period",
            "delta_AED_z",
            "future_extreme_frequency_pct",
        },
        "model CSV",
    )
    check_columns(
        relation_df,
        {
            "metric",
            "detrended",
            "slope_m_per_1sd",
            "slope_m_per_1sd_ci_low",
            "slope_m_per_1sd_ci_high",
            "mean_all_m",
        },
        "ERA5 CSV",
    )

    relation_df = relation_df[
        (relation_df["detrended"] == False)
        & (relation_df["metric"].isin(METRIC_ORDER))
    ].copy()

    if relation_df.empty:
        raise RuntimeError(
            "Aucune relation ERA5 brute trouvée."
        )

    rng = np.random.default_rng(args.seed)
    summary_rows: list[dict] = []

    available_scenarios = [
        scenario
        for scenario in SCENARIO_ORDER
        if scenario in member_df["scenario"].unique()
    ]

    for scenario in available_scenarios:
        for period in PERIOD_ORDER:
            sub_members = member_df[
                (member_df["scenario"] == scenario)
                & (member_df["period"] == period)
            ].copy()

            sub_models = model_df[
                (model_df["scenario"] == scenario)
                & (model_df["period"] == period)
            ].copy()

            if sub_members.empty or sub_models.empty:
                continue

            draws_aed, draws_freq, n_models = hierarchical_bootstrap(
                sub_members,
                args.n_boot,
                rng,
            )

            central_aed = float(
                sub_models["delta_AED_z"].mean()
            )
            central_freq = float(
                sub_models["future_extreme_frequency_pct"].mean()
            )

            aed_low, aed_high = ci95(draws_aed)
            freq_low, freq_high = ci95(draws_freq)

            for row in relation_df.itertuples(index=False):
                beta = float(row.slope_m_per_1sd)
                beta_low = float(row.slope_m_per_1sd_ci_low)
                beta_high = float(row.slope_m_per_1sd_ci_high)
                historical_mean = float(row.mean_all_m)

                beta_se = (beta_high - beta_low) / (2.0 * 1.96)
                beta_draws = rng.normal(
                    beta,
                    beta_se,
                    size=args.n_boot,
                )
                beta_draws = np.maximum(beta_draws, 0.0)

                wave_draws = beta_draws * draws_aed
                pct_draws = (
                    100.0 * wave_draws / historical_mean
                )

                wave_low, wave_high = ci95(wave_draws)
                pct_low, pct_high = ci95(pct_draws)

                central_wave = beta * central_aed
                central_pct = (
                    100.0 * central_wave / historical_mean
                )

                summary_rows.append(
                    {
                        "scenario": scenario,
                        "period": period,
                        "metric": row.metric,
                        "n_models": n_models,
                        "n_members": int(
                            sub_members["member"].nunique()
                        ),
                        "delta_AED_sigma": central_aed,
                        "delta_AED_sigma_ci_low": aed_low,
                        "delta_AED_sigma_ci_high": aed_high,
                        "conditional_wave_change_m": central_wave,
                        "conditional_wave_change_m_ci_low": wave_low,
                        "conditional_wave_change_m_ci_high": wave_high,
                        "conditional_wave_change_pct": central_pct,
                        "conditional_wave_change_pct_ci_low": pct_low,
                        "conditional_wave_change_pct_ci_high": pct_high,
                        "future_AED_extreme_frequency_pct": central_freq,
                        "future_AED_extreme_frequency_pct_ci_low": freq_low,
                        "future_AED_extreme_frequency_pct_ci_high": freq_high,
                        "ERA5_beta_m_per_sigma": beta,
                        "ERA5_beta_ci_low": beta_low,
                        "ERA5_beta_ci_high": beta_high,
                    }
                )

    summary = pd.DataFrame(summary_rows)

    if summary.empty:
        raise RuntimeError(
            "Aucun résultat produit. Vérifier les scénarios et périodes."
        )

    summary.to_csv(
        args.outdir / "future_AED_wave_projection_full.csv",
        index=False,
    )

    # Tableau fin de siècle
    end = summary[summary["period"] == "2071–2100"].copy()

    end["Delta AED, sigma (95% CI)"] = end.apply(
        lambda r: format_ci(
            r["delta_AED_sigma"],
            r["delta_AED_sigma_ci_low"],
            r["delta_AED_sigma_ci_high"],
            2,
        ),
        axis=1,
    )
    end["Conditional change, m (95% CI)"] = end.apply(
        lambda r: format_ci(
            r["conditional_wave_change_m"],
            r["conditional_wave_change_m_ci_low"],
            r["conditional_wave_change_m_ci_high"],
            3,
        ),
        axis=1,
    )
    end["Conditional change, % (95% CI)"] = end.apply(
        lambda r: format_ci(
            r["conditional_wave_change_pct"],
            r["conditional_wave_change_pct_ci_low"],
            r["conditional_wave_change_pct_ci_high"],
            1,
        ),
        axis=1,
    )

    manuscript_table = end[
        [
            "scenario",
            "metric",
            "Delta AED, sigma (95% CI)",
            "Conditional change, m (95% CI)",
            "Conditional change, % (95% CI)",
            "n_models",
            "n_members",
        ]
    ].rename(
        columns={
            "scenario": "Scenario",
            "metric": "Wave metric",
            "n_models": "Models",
            "n_members": "Members",
        }
    )

    manuscript_table.to_csv(
        args.outdir / "future_AED_wave_table_2071_2100.csv",
        index=False,
    )

    # Figures séparées
    def plot_metric(
        metric: str,
        value_col: str,
        low_col: str,
        high_col: str,
        ylabel: str,
        title: str,
        filename: str,
        reference_line: float = 0.0,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        x = np.arange(len(PERIOD_ORDER))

        for scenario in available_scenarios:
            sub = summary[
                (summary["scenario"] == scenario)
                & (summary["metric"] == metric)
            ].set_index("period").reindex(PERIOD_ORDER)

            y = sub[value_col].to_numpy(float)
            low = sub[low_col].to_numpy(float)
            high = sub[high_col].to_numpy(float)

            valid = np.isfinite(y) & np.isfinite(low) & np.isfinite(high)
            if not valid.any():
                continue

            yerr = np.vstack([y - low, high - y])

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                linewidth=1.8,
                capsize=4,
                label=scenario,
            )

        ax.axhline(reference_line, linestyle="--", linewidth=1)
        ax.set_xticks(x, PERIOD_ORDER)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(
            args.outdir / filename,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    plot_metric(
        "Hs_mean",
        "delta_AED_sigma",
        "delta_AED_sigma_ci_low",
        "delta_AED_sigma_ci_high",
        r"$\Delta$AED ($\sigma$)",
        "Projected change in the Atlantic–Europe Dipole",
        "future_AED_change.png",
    )
    plot_metric(
        "Hs_mean",
        "conditional_wave_change_m",
        "conditional_wave_change_m_ci_low",
        "conditional_wave_change_m_ci_high",
        "Conditional change in mean Hs (m)",
        "Conditional circulation-driven change in winter mean Hs",
        "future_Hs_mean_change.png",
    )
    plot_metric(
        "Hs_P95",
        "conditional_wave_change_m",
        "conditional_wave_change_m_ci_low",
        "conditional_wave_change_m_ci_high",
        "Conditional change in P95 Hs (m)",
        "Conditional circulation-driven change in winter P95 Hs",
        "future_Hs_P95_change.png",
    )
    plot_metric(
        "Hs_P99",
        "conditional_wave_change_m",
        "conditional_wave_change_m_ci_low",
        "conditional_wave_change_m_ci_high",
        "Conditional change in P99 Hs (m)",
        "Conditional circulation-driven change in winter P99 Hs",
        "future_Hs_P99_change.png",
    )
    plot_metric(
        "Hs_mean",
        "future_AED_extreme_frequency_pct",
        "future_AED_extreme_frequency_pct_ci_low",
        "future_AED_extreme_frequency_pct_ci_high",
        "Frequency of AED > historical P90 (%)",
        "Projected frequency of extreme positive AED phases",
        "future_AED_extreme_frequency.png",
        reference_line=10.0,
    )

    # Phrases manuscrit
    lines = [
        "These values are conditional circulation-driven estimates, "
        "not direct dynamical wave projections.",
        "",
    ]

    for scenario in available_scenarios:
        sub = summary[
            (summary["scenario"] == scenario)
            & (summary["period"] == "2071–2100")
        ]

        if sub.empty:
            continue

        aed_row = sub[sub["metric"] == "Hs_mean"].iloc[0]

        lines.append(
            f"{scenario}: By 2071–2100, the equal-weighted "
            f"multi-model mean AED change is "
            f"{aed_row['delta_AED_sigma']:.2f} sigma "
            f"(95% CI: "
            f"{aed_row['delta_AED_sigma_ci_low']:.2f}–"
            f"{aed_row['delta_AED_sigma_ci_high']:.2f})."
        )

        for metric in METRIC_ORDER:
            metric_sub = sub[sub["metric"] == metric]
            if metric_sub.empty:
                continue

            row = metric_sub.iloc[0]

            lines.append(
                f"  {metric}: conditional change = "
                f"{row['conditional_wave_change_m']:.3f} m "
                f"(95% CI: "
                f"{row['conditional_wave_change_m_ci_low']:.3f}–"
                f"{row['conditional_wave_change_m_ci_high']:.3f} m), "
                f"equivalent to "
                f"{row['conditional_wave_change_pct']:.1f}% "
                f"(95% CI: "
                f"{row['conditional_wave_change_pct_ci_low']:.1f}–"
                f"{row['conditional_wave_change_pct_ci_high']:.1f}%)."
            )

        lines.append(
            f"  Frequency of AED values exceeding the historical "
            f"P90 = "
            f"{aed_row['future_AED_extreme_frequency_pct']:.1f}% "
            f"(95% CI: "
            f"{aed_row['future_AED_extreme_frequency_pct_ci_low']:.1f}–"
            f"{aed_row['future_AED_extreme_frequency_pct_ci_high']:.1f}%)."
        )
        lines.append("")

    (
        args.outdir
        / "future_AED_wave_manuscript_sentences.txt"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\nAnalyse terminée.")
    print("Fichiers produits :")
    for output in sorted(args.outdir.glob("*")):
        print(f"  {output}")


if __name__ == "__main__":
    main()

