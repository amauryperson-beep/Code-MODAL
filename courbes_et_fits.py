from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def load_two_col_csv(path_csv: Path) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Load a 2-column CSV file (x, y)."""
    with path_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames) < 2:
            raise ValueError(f"{path_csv}: header invalide.")
        rows = list(reader)
        if not rows:
            raise ValueError(f"{path_csv}: vide.")
        x_name = reader.fieldnames[0]
        y_name = reader.fieldnames[1]

    x_vals: list[float] = []
    y_vals: list[float] = []
    for i, row in enumerate(rows, start=2):
        xs = (row.get(x_name) or "").strip()
        ys = (row.get(y_name) or "").strip()
        if not xs and not ys:
            continue
        if not xs or not ys:
            raise ValueError(f"{path_csv}: ligne incomplete {i}.")
        x_vals.append(float(xs))
        y_vals.append(float(ys))
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    idx = np.argsort(x)
    return x[idx], y[idx], x_name, y_name


def multi_gaussian_2(
    x: np.ndarray,
    baseline: float,
    a1: float,
    mu1: float,
    s1: float,
    a2: float,
    mu2: float,
    s2: float,
) -> np.ndarray:
    """Double Gaussian model."""
    g1 = a1 * np.exp(-0.5 * ((x - mu1) / s1) ** 2)
    g2 = a2 * np.exp(-0.5 * ((x - mu2) / s2) ** 2)
    return baseline + g1 + g2


def initial_guess_2g(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Generate initial guess for double Gaussian fit."""
    y_min = float(np.min(y))
    y_span = max(float(np.max(y) - y_min), 1e-6)
    peaks, _ = find_peaks(y, prominence=0.05 * y_span, distance=max(1, len(x) // 3))
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(y))], dtype=int)
    peaks = peaks[np.argsort(y[peaks])[::-1]]

    selected: list[int] = []
    for p in peaks.tolist():
        if p not in selected:
            selected.append(p)
        if len(selected) == 2:
            break
    if len(selected) < 2:
        for p in np.linspace(0, len(x) - 1, 3, dtype=int).tolist():
            if p not in selected:
                selected.append(p)
            if len(selected) == 2:
                break

    dx = np.diff(x)
    dx = dx[dx > 0]
    min_dx = float(np.min(dx)) if dx.size else 1.0
    span = max(float(np.max(x) - np.min(x)), min_dx)
    sigma0 = max(span / 10.0, min_dx)

    i1, i2 = selected[:2]
    p0 = np.array(
        [
            y_min,
            max(float(y[i1] - y_min), 0.1),
            float(x[i1]),
            sigma0,
            max(float(y[i2] - y_min), 0.1),
            float(x[i2]),
            sigma0,
        ],
        dtype=float,
    )
    return p0


def sort_by_mu(params: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort Gaussian parameters so that mu1 < mu2."""
    if params[2] <= params[5]:
        permutation = np.arange(len(params), dtype=int)
    else:
        permutation = np.array([0, 4, 5, 6, 1, 2, 3], dtype=int)
    params_tries = params[permutation]
    covariance_tries = covariance[np.ix_(permutation, permutation)]
    return params_tries, covariance_tries


def fit_double_gaussian(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit a double Gaussian to the data."""
    p0 = initial_guess_2g(x, y)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_span = max(y_max - y_min, 1e-6)
    span = max(x_max - x_min, 1e-6)
    min_dx = max(float(np.min(np.diff(x))) if len(x) > 1 else 1e-3, 1e-3)

    bounds_low = np.array(
        [y_min - 2.0 * y_span, 0.0, x_min, 0.5 * min_dx, 0.0, x_min, 0.5 * min_dx],
        dtype=float,
    )
    bounds_up = np.array(
        [y_max + 2.0 * y_span, 10.0 * y_span, x_max, 1.5 * span, 10.0 * y_span, x_max, 1.5 * span],
        dtype=float,
    )

    sigma = np.full_like(y, fill_value=max(float(np.std(y, ddof=0)), 1e-3), dtype=float)

    popt, pcov = curve_fit(
        multi_gaussian_2,
        x,
        y,
        p0=p0,
        bounds=(bounds_low, bounds_up),
        sigma=sigma,
        absolute_sigma=True,
        maxfev=250000,
    )
    params, covariance = sort_by_mu(popt, pcov)
    params_err = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    y_hat = multi_gaussian_2(x, *params)
    resid = y - y_hat
    chi2 = float(np.sum((resid / sigma) ** 2))
    dof = len(x) - len(params)
    chi2_red = chi2 / dof if dof > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(resid**2)))
    return params, params_err, chi2_red, rmse


def plot_single_curve_with_fit(csv_path: Path) -> None:
    """Load a single CSV file and display it with a double Gaussian fit."""
    x, y, x_name, y_name = load_two_col_csv(csv_path)
    
    if len(x) < 6:
        raise ValueError("Pas assez de points pour ajuster (minimum 6).")
    
    params, params_err, chi2_red, rmse = fit_double_gaussian(x, y)
    
    # Generate fitted curve
    x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 500)
    y_fit = multi_gaussian_2(x_fit, *params)
    
    baseline, a1, mu1, s1, a2, mu2, s2 = params
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, "o-", markersize=6, label="Données")
    plt.plot(x_fit, y_fit, "-", linewidth=2, label="Fit double gaussienne")
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(f"Fit double gaussienne - {csv_path.name}\nχ² réduit = {chi2_red:.4g}, RMSE = {rmse:.4g}")
    plt.grid(True, alpha=0.35)
    
    # Add fit parameters to the plot
    info_text = (
        f"Baseline: {baseline:.4g} ± {params_err[0]:.2g}\n"
        f"Gauss 1: A={a1:.4g}±{params_err[1]:.2g}, μ={mu1:.4g}±{params_err[2]:.2g}, σ={s1:.4g}±{params_err[3]:.2g}\n"
        f"Gauss 2: A={a2:.4g}±{params_err[4]:.2g}, μ={mu2:.4g}±{params_err[5]:.2g}, σ={s2:.4g}±{params_err[6]:.2g}"
    )
    plt.text(0.98, 0.05, info_text, transform=plt.gca().transAxes, fontsize=8,
             verticalalignment="bottom", horizontalalignment="right",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.legend()
    plt.tight_layout()
    plt.show()



def _load_extractions(path_csv: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            i_max_key = "i_max_fit" if "i_max_fit" in row else "i_max_mean"
            i_max_err_key = "i_max_fit_err" if "i_max_fit_err" in row else "i_max_err_utilisee"
            resolution_key = "resolution_fit" if "resolution_fit" in row else ("resolution_mean" if "resolution_mean" in row else None)
            resolution_err_key = (
                "resolution_fit_err"
                if "resolution_fit_err" in row
                else ("resolution_err_utilisee" if "resolution_err_utilisee" in row else None)
            )
            # Support legacy largeur_mh naming
            if resolution_key is None:
                resolution_key = "largeur_mh_fit" if "largeur_mh_fit" in row else ("largeur_mh_mean" if "largeur_mh_mean" in row else None)
            if resolution_err_key is None:
                resolution_err_key = (
                    "largeur_mh_fit_err"
                    if "largeur_mh_fit_err" in row
                    else ("largeur_mh_err_utilisee" if "largeur_mh_err_utilisee" in row else None)
                )
            resolution_mean = float(row[resolution_key]) if resolution_key is not None else float("nan")
            resolution_err = float(row[resolution_err_key]) if resolution_err_key is not None else float("nan")
            rows.append(
                {
                    "tension": float(row["tension"]),
                    "debit": float(row["debit"]),
                    "i_max_mean": float(row[i_max_key]),
                    "i_max_err": float(row[i_max_err_key]),
                    "resolution_mean": resolution_mean,
                    "resolution_err": resolution_err,
                    "sigma1": float(row["sigma1"]) if "sigma1" in row else float("nan"),
                    "sigma1_err": float(row["sigma1_err"]) if "sigma1_err" in row else float("nan"),
                    "sigma2": float(row["sigma2"]) if "sigma2" in row else float("nan"),
                    "sigma2_err": float(row["sigma2_err"]) if "sigma2_err" in row else float("nan"),
                }
            )
    return rows


def _fit_lineaire_pondere(x: np.ndarray, y: np.ndarray, yerr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    def linear(xx: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * xx + b

    sigma = np.asarray(yerr, dtype=float)
    fallback = max(float(np.median(np.abs(y - np.median(y)))), 1e-6)
    sigma = np.where(sigma <= 0, fallback, sigma)
    popt, pcov = curve_fit(linear, x, y, sigma=sigma, absolute_sigma=True)
    perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
    y_hat = linear(x, *popt)
    chi2 = float(np.sum(((y - y_hat) / sigma) ** 2))
    dof = len(x) - 2
    chi2_red = chi2 / dof if dof > 0 else float("nan")
    return popt, perr, chi2_red


def _plot_imageur(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    x_label: str,
    y_label: str,
    titre: str,
    label_donnees: str,
) -> None:
    ordre = np.argsort(x)
    x = x[ordre]
    y = y[ordre]
    yerr = yerr[ordre]

    if len(x) < 2:
        raise ValueError("Pas assez de points pour tracer + fitter (minimum 2).")

    (a, b), (ea, eb), chi2_red = _fit_lineaire_pondere(x, y, yerr)
    x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 400)
    y_fit = (a * x_fit) + b
    label_fit = f"Fit lineaire: y=a*x+b\na={a:.4g} +/- {ea:.2g}, b={b:.4g} +/- {eb:.2g}, chi2_red={chi2_red:.3g}"

    plt.figure(figsize=(8, 5))
    plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=4, label=label_donnees)
    plt.plot(x_fit, y_fit, "-", linewidth=2, label=label_fit)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(titre)
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_i_max_vs_tension(extractions_csv: Path, debit: float) -> None:
    rows = [r for r in _load_extractions(extractions_csv) if np.isclose(r["debit"], debit, rtol=0.0, atol=1e-9)]
    if len(rows) < 2:
        raise ValueError(f"Pas assez de points pour debit={debit:g} (minimum 2 tensions).")
    x = np.array([r["tension"] for r in rows], dtype=float)
    y = np.array([r["i_max_mean"] for r in rows], dtype=float)
    yerr = np.array([r["i_max_err"] for r in rows], dtype=float)
    _plot_imageur(
        x=x,
        y=y,
        yerr=yerr,
        x_label="Tension",
        y_label="Intensite maximale",
        titre=f"I_max en fonction de la tension (debit={debit:g})",
        label_donnees=f"I_max moyen (debit={debit:g})",
    )


def plot_resolution_vs_tension(extractions_csv: Path, debit: float, metrique: str | int = "default", a: float = 0.5, alpha: float = 0.5, beta: float = 0.5) -> None:
    rows = [r for r in _load_extractions(extractions_csv) if np.isclose(r["debit"], debit, rtol=0.0, atol=1e-9)]
    if len(rows) < 2:
        raise ValueError(f"Pas assez de points pour debit={debit:g} (minimum 2 tensions).")
    
    x = np.array([r["tension"] for r in rows], dtype=float)
    
    if isinstance(metrique, str) and metrique == "default":
        # Utiliser la colonne resolution directement
        y = np.array([r["resolution_mean"] for r in rows], dtype=float)
        yerr = np.array([r["resolution_err"] for r in rows], dtype=float)
        y_label = "Resolution (pixels)"
        titre_suffix = ""
    elif isinstance(metrique, int) and metrique in [1, 2, 3, 4]:
        # Utiliser les sigma_metrics numériques
        if not (0.0 <= a <= 1.0):
            raise ValueError(f"Le parametre a doit etre entre 0 et 1 (recu: {a:g}).")
        sigma1 = np.array([r["sigma1"] for r in rows], dtype=float)
        sigma1_err = np.array([r["sigma1_err"] for r in rows], dtype=float)
        sigma2 = np.array([r["sigma2"] for r in rows], dtype=float)
        sigma2_err = np.array([r["sigma2_err"] for r in rows], dtype=float)
        if np.isnan(sigma1).all() or np.isnan(sigma2).all():
            raise ValueError("Les colonnes sigma1/sigma2 ne sont pas presentes dans le CSV d'extraction.")
        y, yerr, description = _compute_sigma_metric(sigma1, sigma1_err, sigma2, sigma2_err, metrique, a)
        y_label = "Valeur de metrique (pixels)"
        titre_suffix = f" ({description})"
    else:
        raise ValueError(f"Metrique inconnue: {metrique}. Utilisez 'default', 'sigma1', 'sigma2' ou 1, 2, 3, 4.")
    
    if np.isnan(y).all():
        raise ValueError(f"Les donnees de resolution ne sont pas presentes dans le CSV d'extraction.")
    
    _plot_imageur(
        x=x,
        y=y,
        yerr=yerr,
        x_label="Tension",
        y_label=y_label,
        titre=f"Resolution en fonction de la tension (debit={debit:g}){titre_suffix}",
        label_donnees=f"Resolution moyenne (debit={debit:g}){titre_suffix}",
    )


def plot_i_max_vs_debit(extractions_csv: Path, tension: float) -> None:
    rows = [r for r in _load_extractions(extractions_csv) if np.isclose(r["tension"], tension, rtol=0.0, atol=1e-9)]
    if len(rows) < 2:
        raise ValueError(f"Pas assez de points pour tension={tension:g} (minimum 2 debits).")
    x = np.array([r["debit"] for r in rows], dtype=float)
    y = np.array([r["i_max_mean"] for r in rows], dtype=float)
    yerr = np.array([r["i_max_err"] for r in rows], dtype=float)
    _plot_imageur(
        x=x,
        y=y,
        yerr=yerr,
        x_label="Debit",
        y_label="Intensite maximale",
        titre=f"I_max en fonction du debit (tension={tension:g})",
        label_donnees=f"I_max moyen (tension={tension:g})",
    )


def plot_resolution_vs_debit(extractions_csv: Path, tension: float, metrique: str | int = "default", a: float = 0.5) -> None:
    rows = [r for r in _load_extractions(extractions_csv) if np.isclose(r["tension"], tension, rtol=0.0, atol=1e-9)]
    if len(rows) < 2:
        raise ValueError(f"Pas assez de points pour tension={tension:g} (minimum 2 debits).")
    
    x = np.array([r["debit"] for r in rows], dtype=float)
    
    if isinstance(metrique, str) and metrique == "default":
        # Utiliser la colonne resolution directement
        y = np.array([r["resolution_mean"] for r in rows], dtype=float)
        yerr = np.array([r["resolution_err"] for r in rows], dtype=float)
        y_label = "Resolution (pixels)"
        titre_suffix = ""
    elif isinstance(metrique, str) and metrique in ["sigma1", "sigma2"]:
        # Utiliser sigma1 ou sigma2 comme métrique de résolution
        sigma_key = metrique
        sigma_err_key = f"{metrique}_err"
        y = np.array([r[sigma_key] for r in rows], dtype=float)
        yerr = np.array([r[sigma_err_key] for r in rows], dtype=float)
        y_label = f"{metrique} resolution (pixels)"
        titre_suffix = f" ({metrique})"
    elif isinstance(metrique, int) and metrique in [1, 2, 3, 4]:
        # Utiliser les sigma_metrics numériques
        if not (0.0 <= a <= 1.0):
            raise ValueError(f"Le parametre a doit etre entre 0 et 1 (recu: {a:g}).")
        sigma1 = np.array([r["sigma1"] for r in rows], dtype=float)
        sigma1_err = np.array([r["sigma1_err"] for r in rows], dtype=float)
        sigma2 = np.array([r["sigma2"] for r in rows], dtype=float)
        sigma2_err = np.array([r["sigma2_err"] for r in rows], dtype=float)
        if np.isnan(sigma1).all() or np.isnan(sigma2).all():
            raise ValueError("Les colonnes sigma1/sigma2 ne sont pas presentes dans le CSV d'extraction.")
        y, yerr, description = _compute_sigma_metric(sigma1, sigma1_err, sigma2, sigma2_err, metrique, a)
        y_label = "Valeur de metrique (pixels)"
        titre_suffix = f" ({description})"
    else:
        raise ValueError(f"Metrique inconnue: {metrique}. Utilisez 'default', 'sigma1', 'sigma2' ou 1, 2, 3, 4.")
    
    if np.isnan(y).all():
        raise ValueError(f"Les donnees de resolution ne sont pas presentes dans le CSV d'extraction.")
    
    _plot_imageur(
        x=x,
        y=y,
        yerr=yerr,
        x_label="Debit",
        y_label=y_label,
        titre=f"Resolution en fonction du debit (tension={tension:g}){titre_suffix}",
        label_donnees=f"Resolution moyenne (tension={tension:g}){titre_suffix}",
    )


def _compute_sigma_metric(sigma1: np.ndarray, sigma1_err: np.ndarray, sigma2: np.ndarray, sigma2_err: np.ndarray, metrique: int, a: float = 0.5, alpha: float = 0.5, beta: float = 0.5) -> tuple[np.ndarray, np.ndarray, str]:

    if metrique == 1:
        return sigma1, sigma1_err, "sigma_1"
    elif metrique == 2:
        return sigma2, sigma2_err, "sigma_2"
    elif metrique == 3:
        values = (a * sigma1) + ((1.0 - a) * sigma2)
        errs = np.sqrt(((a * sigma1_err) ** 2) + (((1.0 - a) * sigma2_err) ** 2))
        return values, errs, f"a*sigma_1 + (1-a)*sigma_2 (a={a:g})"
    elif metrique == 4:
        values = np.full_like(sigma1, np.nan, dtype=float)
        errs = np.full_like(sigma1, np.nan, dtype=float)
        masque_pos = (sigma1 > 0.0) & (sigma2 > 0.0)
        if np.any(masque_pos):
            s1p = sigma1[masque_pos]
            s2p = sigma2[masque_pos]
            es1p = sigma1_err[masque_pos]
            es2p = sigma2_err[masque_pos]
            m4 = (s1p**a) * (s2p ** (1.0 - a))
            rel = np.sqrt(((a * es1p / s1p) ** 2) + ((((1.0 - a) * es2p / s2p) ** 2)))
            values[masque_pos] = m4
            errs[masque_pos] = m4 * rel
        return values, errs, f"sigma_1^a * sigma_2^(1-a) (a={a:g})"
    elif metrique == 5:
        values = (a * sigma1) + ((1.0 - a) * sigma2)
        errs = np.sqrt(((a * sigma1_err) ** 2) + (((1.0 - a) * sigma2_err) ** 2))
        return values, errs, f"a*sigma_1 + (1-a)*sigma_2 (a={a:g})"
    else:
        raise ValueError(f"Metrique inconnue: {metrique}. Utilisez 1, 2, 3 ou 4.")




def _parse_metrique(value: str) -> str | int:
    """Parse metrique argument (accepts 'default', 'sigma1', 'sigma2', '1', '2', '3', '4')."""
    if value in ["default", "sigma1", "sigma2"]:
        return value
    try:
        m = int(value)
        if m in [1, 2, 3, 4]:
            return m
        raise ValueError()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Metrique invalide: '{value}'. Doit etre 'default', 'sigma1', 'sigma2' ou 1, 2, 3, 4."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Affichage des courbes (Geiger/Imageur beta).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("single")
    p0.add_argument("csv", type=Path, help="Chemin du fichier CSV à afficher")

    p1 = sub.add_parser("imageur-i-max-vs-tension")
    p1.add_argument("--debit", type=float, required=True)
    p1.add_argument("--csv", type=Path, default=Path("Mesures/Imageur_beta/extractions_gaussiennes.csv"))

    p2 = sub.add_parser("imageur-resolution-vs-tension")
    p2.add_argument("--debit", type=float, required=True)
    p2.add_argument("--metrique", type=_parse_metrique, default="default", help="Metrique: 'default', 'sigma1', 'sigma2' ou 1, 2, 3, 4")
    p2.add_argument("--a", type=float, default=0.5, help="Parametre a pour les metriques 3 et 4 (0 <= a <= 1)")
    p2.add_argument("--csv", type=Path, default=Path("Mesures/Imageur_beta/extractions_gaussiennes.csv"))

    p3 = sub.add_parser("imageur-i-max-vs-debit")
    p3.add_argument("--tension", type=float, required=True)
    p3.add_argument("--csv", type=Path, default=Path("Mesures/Imageur_beta/extractions_gaussiennes.csv"))

    p4 = sub.add_parser("imageur-resolution-vs-debit")
    p4.add_argument("--tension", type=float, required=True)
    p4.add_argument("--metrique", type=_parse_metrique, default="default", help="Metrique: 'default', 'sigma1', 'sigma2' ou 1, 2, 3, 4")
    p4.add_argument("--a", type=float, default=0.5, help="Parametre a pour les metriques 3 et 4 (0 <= a <= 1)")
    p4.add_argument("--csv", type=Path, default=Path("Mesures/Imageur_beta/extractions_gaussiennes.csv"))

    p5 = sub.add_parser("imageur-metrique-vs-tension")
    p5.add_argument("--debit", type=float, required=True)
    p5.add_argument("--metrique", type=int, required=True, choices=[1, 2, 3, 4], help="Numero de la metrique (1, 2, 3 ou 4)")
    p5.add_argument("--a", type=float, default=0.5, help="Parametre a pour les metriques 3 et 4 (0 <= a <= 1)")
    p5.add_argument("--csv", type=Path, default=Path("Mesures/Imageur_beta/extractions_gaussiennes.csv"))

    p6 = sub.add_parser("imageur-metrique-vs-debit")
    p6.add_argument("--tension", type=float, required=True)
    p6.add_argument("--metrique", type=int, required=True, choices=[1, 2, 3, 4], help="Numero de la metrique (1, 2, 3 ou 4)")
    p6.add_argument("--a", type=float, default=0.5, help="Parametre a pour les metriques 3 et 4 (0 <= a <= 1)")
    p6.add_argument("--csv", type=Path, default=Path("Mesures/Imageur_beta/extractions_gaussiennes.csv"))

    args = parser.parse_args()
    try:
        if args.cmd == "single":
            plot_single_curve_with_fit(args.csv)
        elif args.cmd == "imageur-i-max-vs-tension":
            plot_i_max_vs_tension(args.csv, args.debit)
        elif args.cmd == "imageur-resolution-vs-tension":
            plot_resolution_vs_tension(args.csv, args.debit, args.metrique, args.a)
        elif args.cmd == "imageur-i-max-vs-debit":
            plot_i_max_vs_debit(args.csv, args.tension)
        elif args.cmd == "imageur-resolution-vs-debit":
            plot_resolution_vs_debit(args.csv, args.tension, args.metrique, args.a)
        elif args.cmd == "imageur-metrique-vs-tension":
            plot_resolution_vs_tension(args.csv, args.debit, args.metrique, args.a)
        elif args.cmd == "imageur-metrique-vs-debit":
            plot_resolution_vs_debit(args.csv, args.tension, args.metrique, args.a)
    except ValueError as exc:
        raise SystemExit(f"Erreur: {exc}") from exc


if __name__ == "__main__":
    main()
