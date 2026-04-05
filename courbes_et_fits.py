from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Zone de recadrage en pixels pour l'analyse Imageur beta.
# Modifier ces 2 valeurs si vous voulez couper la courbe entre 2 pixels.
# Mettre None pour ne pas couper de ce cote.
PIXEL_MIN: float | None = None
PIXEL_MAX: float | None = None
FWHM_FACTOR = 2.3548200450309493


def plot_grille_vs_distance(
    distances: np.ndarray,
    tensions: np.ndarray,
    moyenne: np.ndarray,
    ecart_type: np.ndarray,
    titre: str,
    log_x: bool = False,
    log_y: bool = False,
    y_label: str = "Compteur",
) -> None:
    plt.figure()
    for i, tension in enumerate(tensions):
        x = np.asarray(distances, dtype=float)
        y = np.asarray(moyenne[:, i], dtype=float)
        yerr = np.asarray(ecart_type[:, i], dtype=float)
        mask = np.ones_like(x, dtype=bool)
        if log_x:
            mask &= x > 0
        if log_y:
            x = x[mask]
            y = y[mask]
            yerr = yerr[mask]
            mask = y > 0
        x = x[mask]
        y = y[mask]
        yerr = yerr[mask]
        if x.size == 0:
            continue
        plt.errorbar(x, y, yerr, marker="o", linestyle="-", capsize=5, label=f"U = {tension:g} V")
    plt.xlabel("Distance (m)")
    plt.ylabel(y_label)
    plt.title(titre)
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_grille_vs_tension(
    distances: np.ndarray,
    tensions: np.ndarray,
    moyenne: np.ndarray,
    ecart_type: np.ndarray,
    titre: str,
    log_y: bool = False,
    y_label: str = "Compteur",
) -> None:
    plt.figure()
    for i, distance in enumerate(distances):
        x = np.asarray(tensions, dtype=float)
        y = np.asarray(moyenne[i, :], dtype=float)
        yerr = np.asarray(ecart_type[i, :], dtype=float)
        if log_y:
            mask = y > 0
            x = x[mask]
            y = y[mask]
            yerr = yerr[mask]
            if x.size == 0:
                continue
        plt.errorbar(x, y, yerr, marker="o", linestyle="-", capsize=5, label=f"D = {distance:g} m")
    plt.xlabel("Tension (V)")
    plt.ylabel(y_label)
    plt.title(titre)
    if log_y:
        plt.yscale("log")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_serie_vs_distance(
    distances: np.ndarray,
    moyenne: np.ndarray,
    ecart_type: np.ndarray,
    titre: str,
    label: str,
    y_label: str = "Compteur",
) -> None:
    plt.figure()
    plt.errorbar(distances, moyenne, ecart_type, marker="o", linestyle="-", capsize=5, label=label)
    plt.xlabel("Distance (m)")
    plt.ylabel(y_label)
    plt.title(titre)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_multi_serie_vs_distance(
    distances: np.ndarray,
    categories: np.ndarray,
    moyenne: np.ndarray,
    ecart_type: np.ndarray,
    titre: str,
    y_label: str = "Compteur",
) -> None:
    plt.figure()
    for i, categorie in enumerate(categories):
        plt.errorbar(
            distances,
            moyenne[i, :],
            ecart_type[i, :],
            marker="o",
            linestyle="-",
            capsize=5,
            label=f"{int(categorie)} plaque(s)",
        )
    plt.xlabel("Distance (m)")
    plt.ylabel(y_label)
    plt.title(titre)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_courbe_ajustee(
    x: np.ndarray,
    y: np.ndarray,
    y_erreur: np.ndarray,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    titre: str,
    etiquette: str,
    y_label: str = "Compteur",
) -> None:
    plt.figure()
    plt.errorbar(x, y, yerr=y_erreur, fmt="o", capsize=4, label="Données")
    plt.plot(x_fit, y_fit, label=etiquette)
    plt.title(titre)
    plt.xlabel("Distance (m)")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()


@dataclass(frozen=True)
class FitResult:
    baseline: float
    baseline_err: float
    params: np.ndarray
    params_err: np.ndarray
    sigma_x: float
    chi2: float
    dof: int
    chi2_reduced: float
    rmse: float


def load_two_col_csv(path_csv: Path) -> tuple[np.ndarray, np.ndarray, str, str]:
    with path_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames) < 2:
            raise ValueError(f"{path_csv}: header with 2 columns required.")
        rows = list(reader)
        if not rows:
            raise ValueError(f"{path_csv}: empty data.")

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
            raise ValueError(f"{path_csv}: incomplete row line {i}.")
        x_vals.append(float(xs))
        y_vals.append(float(ys))

    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    order = np.argsort(x)
    return x[order], y[order], x_name, y_name


def crop_curve(x: np.ndarray, y: np.ndarray, x_min: float | None, x_max: float | None) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones_like(x, dtype=bool)
    if x_min is not None:
        mask &= x >= float(x_min)
    if x_max is not None:
        mask &= x <= float(x_max)
    xc = x[mask]
    yc = y[mask]
    if xc.size < 6:
        raise ValueError("Pas assez de points apres recadrage pixel.")
    return xc, yc


def infer_sigma_x(x: np.ndarray) -> float:
    dx = np.diff(np.sort(x))
    dx = dx[dx > 0]
    if dx.size == 0:
        return 1.0
    return float(np.median(dx))


def multi_gaussian(x: np.ndarray, baseline: float, *params: float) -> np.ndarray:
    y = np.full_like(x, baseline, dtype=float)
    for i in range(0, len(params), 3):
        a, mu, sigma = params[i : i + 3]
        y += a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return y


def d_multi_gaussian_dx(x: np.ndarray, *params: float) -> np.ndarray:
    dy = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 3):
        a, mu, sigma = params[i : i + 3]
        z = (x - mu) / sigma
        dy += a * np.exp(-0.5 * z**2) * (-(x - mu) / (sigma**2))
    return dy


def initial_guess_2g(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    y_min = float(np.min(y))
    y_span = max(float(np.max(y) - y_min), 1e-6)
    peaks, _ = find_peaks(y, prominence=0.05 * y_span, distance=max(1, len(x) // 3))
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(y))], dtype=int)
    peaks = peaks[np.argsort(y[peaks])[::-1]]
    chosen: list[int] = []
    for p in peaks.tolist():
        if p not in chosen:
            chosen.append(int(p))
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        for p in np.linspace(0, len(x) - 1, 3, dtype=int).tolist():
            if p not in chosen:
                chosen.append(p)
            if len(chosen) == 2:
                break

    dx = np.diff(x)
    dx = dx[dx > 0]
    min_dx = float(np.min(dx)) if dx.size else 1.0
    span = max(float(np.max(x) - np.min(x)), min_dx)
    sigma0 = max(span / 10.0, min_dx)

    p0 = [y_min]
    for idx in chosen[:2]:
        p0.extend([max(float(y[idx] - y_min), 0.1), float(x[idx]), sigma0])
    return np.asarray(p0, dtype=float)


def fit_double_gaussian(x: np.ndarray, y: np.ndarray, sigma_x: float | None = None) -> FitResult:
    sx = float(sigma_x) if sigma_x is not None else infer_sigma_x(x)
    p0 = initial_guess_2g(x, y)

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_span = max(y_max - y_min, 1e-6)
    span = max(x_max - x_min, 1e-6)
    min_dx = max(float(np.min(np.diff(x))) if len(x) > 1 else 1e-3, 1e-3)

    lower = np.array([y_min - 2.0 * y_span, 0.0, x_min, 0.5 * min_dx, 0.0, x_min, 0.5 * min_dx], dtype=float)
    upper = np.array([y_max + 2.0 * y_span, 10.0 * y_span, x_max, 1.5 * span, 10.0 * y_span, x_max, 1.5 * span], dtype=float)

    popt, pcov = curve_fit(
        multi_gaussian,
        x,
        y,
        p0=p0,
        bounds=(lower, upper),
        maxfev=250000,
    )
    perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))

    y_hat = multi_gaussian(x, *popt)
    resid = y - y_hat
    rmse = float(np.sqrt(np.mean(resid**2)))

    dy_dx = d_multi_gaussian_dx(x, *popt[1:])
    sigma_y = np.abs(dy_dx) * sx
    y_scale = max(float(np.max(y) - np.min(y)), 1.0)
    floor = max(1e-6 * y_scale, float(np.percentile(sigma_y[sigma_y > 0], 10)) * 0.1) if np.any(sigma_y > 0) else 1e-3 * y_scale
    sigma_y = np.maximum(sigma_y, floor)
    chi2 = float(np.sum((resid / sigma_y) ** 2))
    dof = len(x) - len(popt)
    chi2_red = chi2 / dof if dof > 0 else float("nan")

    return FitResult(
        baseline=float(popt[0]),
        baseline_err=float(perr[0]),
        params=np.asarray(popt[1:], dtype=float),
        params_err=np.asarray(perr[1:], dtype=float),
        sigma_x=sx,
        chi2=chi2,
        dof=dof,
        chi2_reduced=chi2_red,
        rmse=rmse,
    )


def parse_measure_filename(path_csv: Path) -> tuple[float, float, int]:
    name = path_csv.name
    rgx_3 = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\-([0-9]+(?:\.[0-9]+)?)\-([0-9]+)\.csv$")
    rgx_2 = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\-([0-9]+(?:\.[0-9]+)?)\.csv$")
    m3 = rgx_3.match(name)
    if m3:
        return float(m3.group(1)), float(m3.group(2)), int(m3.group(3))
    m2 = rgx_2.match(name)
    if m2:
        # Compatibilite ancien format sans numero de repetition.
        return float(m2.group(1)), float(m2.group(2)), 1
    raise ValueError(f"Nom de fichier non reconnu: {name} (attendu: tension-debit-numero.csv)")


def sort_gaussians_by_mu(fit: FitResult) -> tuple[np.ndarray, np.ndarray]:
    pairs: list[tuple[float, np.ndarray, np.ndarray]] = []
    for i in range(2):
        p = fit.params[3 * i : 3 * i + 3]
        e = fit.params_err[3 * i : 3 * i + 3]
        pairs.append((float(p[1]), p.copy(), e.copy()))
    pairs.sort(key=lambda t: t[0])
    params_sorted = np.concatenate([pairs[0][1], pairs[1][1]])
    errs_sorted = np.concatenate([pairs[0][2], pairs[1][2]])
    return params_sorted, errs_sorted


def dominant_sigma_and_error(params_sorted: np.ndarray, errs_sorted: np.ndarray) -> tuple[float, float]:
    a1, _, s1 = params_sorted[0:3]
    a2, _, s2 = params_sorted[3:6]
    es1 = errs_sorted[2]
    es2 = errs_sorted[5]
    if a1 >= a2:
        return float(s1), float(es1)
    return float(s2), float(es2)


def save_fit_params(path_csv: Path, fit: FitResult) -> None:
    with path_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "value", "error"])
        w.writerow(["baseline", fit.baseline, fit.baseline_err])
        for i in range(2):
            a, mu, sigma = fit.params[3 * i : 3 * i + 3]
            ea, emu, es = fit.params_err[3 * i : 3 * i + 3]
            w.writerow([f"gauss_{i+1}_amplitude", a, ea])
            w.writerow([f"gauss_{i+1}_mu", mu, emu])
            w.writerow([f"gauss_{i+1}_sigma", sigma, es])
        w.writerow(["sigma_x", fit.sigma_x, 0.0])
        w.writerow(["chi2", fit.chi2, 0.0])
        w.writerow(["dof", fit.dof, 0.0])
        w.writerow(["chi2_reduced", fit.chi2_reduced, 0.0])
        w.writerow(["rmse", fit.rmse, 0.0])


def plot_single_curve(x: np.ndarray, y: np.ndarray, fit: FitResult, x_name: str, y_name: str, title: str) -> None:
    x_dense = np.linspace(float(np.min(x)), float(np.max(x)), 1200)
    y_dense = multi_gaussian(x_dense, fit.baseline, *fit.params)

    a1, mu1, s1 = fit.params[0:3]
    a2, mu2, s2 = fit.params[3:6]
    ea1, emu1, es1 = fit.params_err[0:3]
    ea2, emu2, es2 = fit.params_err[3:6]
    label = (
        f"Fit double gaussiennes\n"
        f"G1: A={a1:.3g} +/- {ea1:.2g}, mu={mu1:.3g} +/- {emu1:.2g}, s={s1:.3g} +/- {es1:.2g}\n"
        f"G2: A={a2:.3g} +/- {ea2:.2g}, mu={mu2:.3g} +/- {emu2:.2g}, s={s2:.3g} +/- {es2:.2g}\n"
        f"chi2_red={fit.chi2_reduced:.3g}"
    )

    plt.figure(figsize=(9, 5))
    plt.errorbar(x, y, xerr=fit.sigma_x, yerr=np.zeros_like(y), fmt="o", capsize=3, label="Donnees")
    plt.plot(x_dense, y_dense, "-", linewidth=2.0, label=label)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def list_measure_files(base_dir: Path) -> list[Path]:
    files: list[tuple[float, float, int, Path]] = []
    for p in base_dir.glob("*.csv"):
        try:
            tension, debit, numero = parse_measure_filename(p)
        except ValueError:
            continue
        files.append((tension, debit, numero, p))
    files.sort(key=lambda t: (t[0], t[1], t[2], t[3].name))
    return [p for _, _, _, p in files]


def extract_and_aggregate_imageur(
    base_dir: Path,
    out_csv: Path,
    out_params_dir: Path,
    sigma_x: float | None = None,
) -> Path:
    curves = list_measure_files(base_dir)
    if not curves:
        raise ValueError(f"Aucune mesure imageur detectee dans {base_dir}")
    out_params_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    details: list[dict[str, float | int | str]] = []
    for curve in curves:
        tension, debit, numero = parse_measure_filename(curve)
        x, y, _, _ = load_two_col_csv(curve)
        x, y = crop_curve(x, y, PIXEL_MIN, PIXEL_MAX)
        fit = fit_double_gaussian(x, y, sigma_x=sigma_x)

        params_sorted, errs_sorted = sort_gaussians_by_mu(fit)
        sigma_dom, sigma_dom_err = dominant_sigma_and_error(params_sorted, errs_sorted)
        fwhm = FWHM_FACTOR * sigma_dom
        fwhm_err = FWHM_FACTOR * sigma_dom_err

        save_fit_params(out_params_dir / f"{curve.stem}_fit_gaussiennes_parametres.csv", fit)

        imax_idx = int(np.argmax(y))
        x_max_data = float(x[imax_idx])
        max_data = float(y[imax_idx])

        x_dense = np.linspace(float(np.min(x)), float(np.max(x)), 3000)
        y_dense = multi_gaussian(x_dense, fit.baseline, *fit.params)
        i_fit = int(np.argmax(y_dense))
        x_max_fit = float(x_dense[i_fit])
        max_fit = float(y_dense[i_fit])

        details.append(
            {
                "tension": tension,
                "debit": debit,
                "numero": numero,
                "fichier": curve.name,
                "x_max_data_px": x_max_data,
                "max_data_gray": max_data,
                "x_max_fit_px": x_max_fit,
                "max_fit_gray": max_fit,
                "sigma_dominante_px": sigma_dom,
                "sigma_dominante_fit_error_px": sigma_dom_err,
                "fwhm_dominante_px": fwhm,
                "fwhm_dominante_fit_error_px": fwhm_err,
                "chi2_reduced": fit.chi2_reduced,
                "rmse": fit.rmse,
                "sigma_x_px": fit.sigma_x,
                "pixel_min": "" if PIXEL_MIN is None else PIXEL_MIN,
                "pixel_max": "" if PIXEL_MAX is None else PIXEL_MAX,
            }
        )

    grouped: dict[tuple[float, float], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in details:
        grouped[(float(row["tension"]), float(row["debit"]))].append(row)

    agg_rows: list[dict[str, float | int | str]] = []
    for (tension, debit), rows in grouped.items():
        max_fit_vals = np.array([float(r["max_fit_gray"]) for r in rows], dtype=float)
        fwhm_vals = np.array([float(r["fwhm_dominante_px"]) for r in rows], dtype=float)
        sigma_vals = np.array([float(r["sigma_dominante_px"]) for r in rows], dtype=float)
        sigma_fit_err_vals = np.array([float(r["sigma_dominante_fit_error_px"]) for r in rows], dtype=float)
        fwhm_fit_err_vals = np.array([float(r["fwhm_dominante_fit_error_px"]) for r in rows], dtype=float)
        rmse_vals = np.array([float(r["rmse"]) for r in rows], dtype=float)
        chi2r_vals = np.array([float(r["chi2_reduced"]) for r in rows], dtype=float)
        n = len(rows)

        max_std = float(np.std(max_fit_vals, ddof=1)) if n > 1 else 0.0
        fwhm_std = float(np.std(fwhm_vals, ddof=1)) if n > 1 else 0.0
        sigma_std = float(np.std(sigma_vals, ddof=1)) if n > 1 else 0.0
        max_err_used = max_std if n > 1 else max(float(np.mean(rmse_vals)), 1e-9)
        fwhm_err_used = fwhm_std if n > 1 else max(float(np.mean(fwhm_fit_err_vals)), 1e-9)

        agg_rows.append(
            {
                "tension": tension,
                "debit": debit,
                "n_mesures": n,
                "max_fit_gray_mean": float(np.mean(max_fit_vals)),
                "max_fit_gray_std": max_std,
                "max_fit_gray_err_utilisee": max_err_used,
                "sigma_dominante_px_mean": float(np.mean(sigma_vals)),
                "sigma_dominante_px_std": sigma_std,
                "sigma_dominante_fit_error_mean": float(np.mean(sigma_fit_err_vals)),
                "fwhm_dominante_px_mean": float(np.mean(fwhm_vals)),
                "fwhm_dominante_px_std": fwhm_std,
                "fwhm_dominante_fit_error_mean": float(np.mean(fwhm_fit_err_vals)),
                "fwhm_dominante_err_utilisee": fwhm_err_used,
                "chi2_reduced_mean": float(np.mean(chi2r_vals)),
                "chi2_reduced_std": float(np.std(chi2r_vals, ddof=1)) if n > 1 else 0.0,
                "rmse_mean": float(np.mean(rmse_vals)),
                "sigma_x_px_mean": float(np.mean([float(r["sigma_x_px"]) for r in rows])),
                "pixel_min": "" if PIXEL_MIN is None else PIXEL_MIN,
                "pixel_max": "" if PIXEL_MAX is None else PIXEL_MAX,
            }
        )

    agg_rows.sort(key=lambda r: (float(r["debit"]), float(r["tension"])))
    fields = [
        "tension",
        "debit",
        "n_mesures",
        "max_fit_gray_mean",
        "max_fit_gray_std",
        "max_fit_gray_err_utilisee",
        "sigma_dominante_px_mean",
        "sigma_dominante_px_std",
        "sigma_dominante_fit_error_mean",
        "fwhm_dominante_px_mean",
        "fwhm_dominante_px_std",
        "fwhm_dominante_fit_error_mean",
        "fwhm_dominante_err_utilisee",
        "chi2_reduced_mean",
        "chi2_reduced_std",
        "rmse_mean",
        "sigma_x_px_mean",
        "pixel_min",
        "pixel_max",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in agg_rows:
            w.writerow(row)
    return out_csv


def fit_linear_weighted(x: np.ndarray, y: np.ndarray, yerr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int, float]:
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
    return popt, perr, chi2, dof, chi2_red


def _safe_name(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def _plot_series_with_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    xlabel: str,
    ylabel: str,
    titre: str,
    serie_label: str,
    out_png: Path | None,
) -> None:
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    yerr = yerr[order]
    if len(x) < 2:
        return

    (a, b), (ea, eb), _, _, chi2_red = fit_linear_weighted(x, y, yerr)
    x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 400)
    y_fit = a * x_fit + b
    legend_fit = f"Fit lineaire: y=a*x+b\na={a:.4g} +/- {ea:.2g}, b={b:.4g} +/- {eb:.2g}, chi2_red={chi2_red:.3g}"

    plt.figure(figsize=(8, 5))
    plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=4, label=serie_label)
    plt.plot(x_fit, y_fit, "-", linewidth=2, label=legend_fit)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titre)
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=9)
    plt.tight_layout()
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=180)


def plot_aggregated_curves(extraction_csv: Path, out_dir: Path | None = None) -> None:
    rows: list[dict[str, float]] = []
    with extraction_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "tension": float(row["tension"]),
                    "debit": float(row["debit"]),
                    "max": float(row["max_fit_gray_mean"]),
                    "max_err": float(row["max_fit_gray_err_utilisee"]),
                    "fwhm": float(row["fwhm_dominante_px_mean"]),
                    "fwhm_err": float(row["fwhm_dominante_err_utilisee"]),
                }
            )
    if not rows:
        raise ValueError(f"Aucune donnee dans {extraction_csv}")

    by_debit: dict[float, list[dict[str, float]]] = defaultdict(list)
    by_tension: dict[float, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_debit[row["debit"]].append(row)
        by_tension[row["tension"]].append(row)

    nb_figures = 0
    for debit, serie in sorted(by_debit.items(), key=lambda t: t[0]):
        if len({s["tension"] for s in serie}) < 2:
            continue
        x = np.array([s["tension"] for s in serie], dtype=float)
        y_max = np.array([s["max"] for s in serie], dtype=float)
        y_max_err = np.array([s["max_err"] for s in serie], dtype=float)
        y_fwhm = np.array([s["fwhm"] for s in serie], dtype=float)
        y_fwhm_err = np.array([s["fwhm_err"] for s in serie], dtype=float)
        suffix = f"debit_{_safe_name(debit)}"
        p1 = None if out_dir is None else out_dir / f"max_vs_tension_{suffix}.png"
        p2 = None if out_dir is None else out_dir / f"fwhm_vs_tension_{suffix}.png"
        _plot_series_with_linear_fit(
            x,
            y_max,
            y_max_err,
            xlabel="Tension",
            ylabel="Intensite maximale (Gray value)",
            titre=f"Intensite maximale en fonction de la tension (debit={debit:g})",
            serie_label=f"Donnees agregees (debit={debit:g})",
            out_png=p1,
        )
        nb_figures += 1
        _plot_series_with_linear_fit(
            x,
            y_fwhm,
            y_fwhm_err,
            xlabel="Tension",
            ylabel="Largeur a mi-hauteur (pixels)",
            titre=f"Largeur a mi-hauteur en fonction de la tension (debit={debit:g})",
            serie_label=f"Donnees agregees (debit={debit:g})",
            out_png=p2,
        )
        nb_figures += 1

    for tension, serie in sorted(by_tension.items(), key=lambda t: t[0]):
        if len({s["debit"] for s in serie}) < 2:
            continue
        x = np.array([s["debit"] for s in serie], dtype=float)
        y_max = np.array([s["max"] for s in serie], dtype=float)
        y_max_err = np.array([s["max_err"] for s in serie], dtype=float)
        y_fwhm = np.array([s["fwhm"] for s in serie], dtype=float)
        y_fwhm_err = np.array([s["fwhm_err"] for s in serie], dtype=float)
        suffix = f"tension_{_safe_name(tension)}"
        p1 = None if out_dir is None else out_dir / f"max_vs_debit_{suffix}.png"
        p2 = None if out_dir is None else out_dir / f"fwhm_vs_debit_{suffix}.png"
        _plot_series_with_linear_fit(
            x,
            y_max,
            y_max_err,
            xlabel="Debit",
            ylabel="Intensite maximale (Gray value)",
            titre=f"Intensite maximale en fonction du debit (tension={tension:g})",
            serie_label=f"Donnees agregees (tension={tension:g})",
            out_png=p1,
        )
        nb_figures += 1
        _plot_series_with_linear_fit(
            x,
            y_fwhm,
            y_fwhm_err,
            xlabel="Debit",
            ylabel="Largeur a mi-hauteur (pixels)",
            titre=f"Largeur a mi-hauteur en fonction du debit (tension={tension:g})",
            serie_label=f"Donnees agregees (tension={tension:g})",
            out_png=p2,
        )
        nb_figures += 1

    if nb_figures == 0:
        print("Aucune courbe agregée tracee: il faut au moins 2 points par serie.")
    else:
        print(f"Nombre de courbes tracees: {nb_figures}")
        # Affiche toutes les figures en meme temps, chacune dans sa propre fenetre.
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Traces et fits pour les courbes Imageur beta.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_single = sub.add_parser("single", help="Tracer + fit double gaussienne d'une courbe CSV.")
    p_single.add_argument("csv_path", type=Path)
    p_single.add_argument("--sigma-x", type=float, default=None)
    p_single.add_argument(
        "--title",
        default="Variation de la tension avec fit en double gaussiennes",
    )
    p_single.add_argument("--save-params", type=Path, default=None)

    p_max = sub.add_parser(
        "maxima",
        help=(
            "Agreger les mesures tension-debit-numero.csv, generer extraction_gaussiennes.csv "
            "et tracer maxima/FWHM vs tension et vs debit avec fit lineaire."
        ),
    )
    p_max.add_argument("--base-dir", type=Path, default=Path("Mesures/Imageur_beta"))
    p_max.add_argument("--out-dir", type=Path, default=Path("Mesures/Imageur_beta/analyse"))
    p_max.add_argument("--no-save-plot", action="store_true")
    p_max.add_argument("--sigma-x", type=float, default=None, help="Incertitude x fixe (taille pixel).")

    args = parser.parse_args()

    if args.cmd == "single":
        x, y, x_name, y_name = load_two_col_csv(args.csv_path)
        x, y = crop_curve(x, y, PIXEL_MIN, PIXEL_MAX)
        fit = fit_double_gaussian(x, y, sigma_x=args.sigma_x)
        if args.save_params is not None:
            args.save_params.parent.mkdir(parents=True, exist_ok=True)
            save_fit_params(args.save_params, fit)
        plot_single_curve(x, y, fit, x_name, y_name, args.title)
        return

    summary_csv = args.out_dir / "extraction_gaussiennes.csv"
    params_dir = args.out_dir / "fit_params"
    summary_csv = extract_and_aggregate_imageur(args.base_dir, summary_csv, params_dir, sigma_x=args.sigma_x)
    out_plot_dir = None if args.no_save_plot else args.out_dir
    plot_aggregated_curves(summary_csv, out_dir=out_plot_dir)
    print(f"Extraction CSV: {summary_csv}")


if __name__ == "__main__":
    main()
