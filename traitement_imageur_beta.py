from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


# Reglages de decoupe en pixels.
# Modifier ici selon le besoin. Mettre None pour ne pas couper.
PIXEL_MIN: float | None = None
PIXEL_MAX: float | None = None

LARGEUR_MH_FACTOR = 2.3548200450309493


def parse_raw_filename(path_csv: Path) -> tuple[float, float, int]:
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\-([0-9]+(?:\.[0-9]+)?)\-([0-9]+)\.csv$", path_csv.name)
    if not m:
        raise ValueError(f"Nom invalide: {path_csv.name} (attendu: tension-debit-numero.csv)")
    return float(m.group(1)), float(m.group(2)), int(m.group(3))


def parse_aggregated_filename(path_csv: Path) -> tuple[float, float]:
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\-([0-9]+(?:\.[0-9]+)?)\.csv$", path_csv.name)
    if not m:
        raise ValueError(f"Nom invalide: {path_csv.name} (attendu: tension-debit.csv)")
    return float(m.group(1)), float(m.group(2))


def _fmt_number(v: float) -> str:
    if float(v).is_integer():
        return str(int(v))
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def pair_filename(tension: float, debit: float) -> str:
    return f"{_fmt_number(tension)}-{_fmt_number(debit)}.csv"


def load_two_col_csv(path_csv: Path) -> tuple[np.ndarray, np.ndarray, str, str]:
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


def load_aggregated_curve(path_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    with path_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames) < 2:
            raise ValueError(f"{path_csv}: header invalide.")
        rows = list(reader)
        if not rows:
            raise ValueError(f"{path_csv}: vide.")
        x_name = reader.fieldnames[0]
        y_name = reader.fieldnames[1]
        yerr_name = reader.fieldnames[2] if len(reader.fieldnames) >= 3 else None

    x_vals: list[float] = []
    y_vals: list[float] = []
    yerr_vals: list[float] = []
    for i, row in enumerate(rows, start=2):
        xs = (row.get(x_name) or "").strip()
        ys = (row.get(y_name) or "").strip()
        if not xs and not ys:
            continue
        if not xs or not ys:
            raise ValueError(f"{path_csv}: ligne incomplete {i}.")
        x_vals.append(float(xs))
        y_vals.append(float(ys))
        if yerr_name is not None:
            yerrs = (row.get(yerr_name) or "").strip()
            yerr_vals.append(float(yerrs) if yerrs else 0.0)
        else:
            yerr_vals.append(0.0)

    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    yerr = np.asarray(yerr_vals, dtype=float)
    idx = np.argsort(x)
    return x[idx], y[idx], yerr[idx], x_name, y_name


def crop_curve(x: np.ndarray, y: np.ndarray, yerr: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    mask = np.ones_like(x, dtype=bool)
    if PIXEL_MIN is not None:
        mask &= x >= float(PIXEL_MIN)
    if PIXEL_MAX is not None:
        mask &= x <= float(PIXEL_MAX)
    xc = x[mask]
    yc = y[mask]
    yerrc = yerr[mask] if yerr is not None else None
    if xc.size < 6:
        raise ValueError("Pas assez de points apres decoupe pixel.")
    return xc, yc, yerrc


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
    g1 = a1 * np.exp(-0.5 * ((x - mu1) / s1) ** 2)
    g2 = a2 * np.exp(-0.5 * ((x - mu2) / s2) ** 2)
    return baseline + g1 + g2


def initial_guess_2g(x: np.ndarray, y: np.ndarray) -> np.ndarray:
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
    if params[2] <= params[5]:
        permutation = np.arange(len(params), dtype=int)
    else:
        permutation = np.array([0, 4, 5, 6, 1, 2, 3], dtype=int)
    params_tries = params[permutation]
    covariance_tries = covariance[np.ix_(permutation, permutation)]
    return params_tries, covariance_tries


def fit_curve(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
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

    if yerr is None:
        sigma = np.full_like(y, fill_value=max(float(np.std(y, ddof=0)), 1e-3), dtype=float)
    else:
        sigma = np.asarray(yerr, dtype=float)
        fallback = max(float(np.median(np.abs(y - np.median(y)))), 1e-6)
        sigma = np.where(sigma <= 0, fallback, sigma)

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
    return params, params_err, covariance, chi2_red, rmse


def estimate_i_max_uncertainty(
    params: np.ndarray,
    covariance: np.ndarray,
    x_min: float,
    x_max: float,
    n_samples: int = 2000,
) -> float:
    covariance = 0.5 * (covariance + covariance.T)
    try:
        eigvals, eigvecs = np.linalg.eigh(covariance)
        eigvals = np.clip(eigvals, 1e-18, None)
        covariance = eigvecs @ np.diag(eigvals) @ eigvecs.T
    except np.linalg.LinAlgError:
        diag = np.clip(np.diag(covariance), 1e-18, None)
        covariance = np.diag(diag)

    rng = np.random.default_rng(0)
    try:
        samples = rng.multivariate_normal(params, covariance, size=n_samples, check_valid="ignore")
    except np.linalg.LinAlgError:
        diag = np.clip(np.diag(covariance), 1e-18, None)
        samples = rng.multivariate_normal(params, np.diag(diag), size=n_samples, check_valid="ignore")

    valides = np.isfinite(samples).all(axis=1) & (samples[:, 3] > 0.0) & (samples[:, 6] > 0.0)
    if np.count_nonzero(valides) < 30:
        return float("nan")

    x_dense = np.linspace(float(x_min), float(x_max), 3000)
    i_max_samples = np.array([np.max(multi_gaussian_2(x_dense, *p)) for p in samples[valides]], dtype=float)
    if i_max_samples.size < 2:
        return float("nan")
    return float(np.std(i_max_samples, ddof=1))


def derive_metrics(
    params: np.ndarray,
    params_err: np.ndarray,
    covariance: np.ndarray,
    x_min: float,
    x_max: float,
) -> dict[str, float]:
    a1, mu1, s1 = params[1:4]
    a2, mu2, s2 = params[4:7]
    ea1, emu1, es1 = params_err[1:4]
    ea2, emu2, es2 = params_err[4:7]

    if a1 >= a2:
        sigma_dom = float(s1)
        sigma_dom_err = float(es1)
    else:
        sigma_dom = float(s2)
        sigma_dom_err = float(es2)

    largeur_mh = LARGEUR_MH_FACTOR * sigma_dom
    largeur_mh_err = LARGEUR_MH_FACTOR * sigma_dom_err

    x_dense = np.linspace(float(x_min), float(x_max), 3000)
    y_dense = multi_gaussian_2(x_dense, *params)
    i_max = float(np.max(y_dense))
    i_max_err = estimate_i_max_uncertainty(params, covariance, x_min, x_max)
    if not np.isfinite(i_max_err) or i_max_err <= 0:
        i_max_err = max(float(params_err[0]), 1e-9)
    x_i_max = float(x_dense[int(np.argmax(y_dense))])

    return {
        "i_max": i_max,
        "i_max_err": i_max_err,
        "x_i_max": x_i_max,
        "sigma_dom": sigma_dom,
        "sigma_dom_err": sigma_dom_err,
        "largeur_mh": largeur_mh,
        "largeur_mh_err": largeur_mh_err,
        "a1": float(a1),
        "a1_err": float(ea1),
        "mu1": float(mu1),
        "mu1_err": float(emu1),
        "sigma1": float(s1),
        "sigma1_err": float(es1),
        "a2": float(a2),
        "a2_err": float(ea2),
        "mu2": float(mu2),
        "mu2_err": float(emu2),
        "sigma2": float(s2),
        "sigma2_err": float(es2),
        "baseline": float(params[0]),
        "baseline_err": float(params_err[0]),
    }


def write_aggregated_curve(
    out_csv: Path,
    x: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    n_points: np.ndarray,
    x_name: str,
    y_name: str,
) -> None:
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([x_name, y_name, f"{y_name}_std", "n_mesures_point"])
        for xi, yi, si, ni in zip(x, y_mean, y_std, n_points):
            w.writerow([f"{xi:.12g}", f"{yi:.12g}", f"{si:.12g}", int(ni)])


def aggregate_group(files: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    values_by_x: dict[float, list[float]] = defaultdict(list)
    x_name_ref = ""
    y_name_ref = ""
    for path in files:
        x_raw, y_raw, x_name, y_name = load_two_col_csv(path)
        x, y, _ = crop_curve(x_raw, y_raw)
        x_name_ref = x_name
        y_name_ref = y_name
        for xi, yi in zip(x, y):
            values_by_x[float(xi)].append(float(yi))

    x_sorted = np.array(sorted(values_by_x.keys()), dtype=float)
    y_mean = np.array([np.mean(values_by_x[x]) for x in x_sorted], dtype=float)
    y_std = np.array(
        [np.std(values_by_x[x], ddof=1) if len(values_by_x[x]) > 1 else 0.0 for x in x_sorted],
        dtype=float,
    )
    n_points = np.array([len(values_by_x[x]) for x in x_sorted], dtype=int)
    return x_sorted, y_mean, y_std, n_points, x_name_ref, y_name_ref


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extrait les parametres des fits gaussiens depuis les courbes agregees "
            "de type tension-debit.csv et produit extractions_gaussiennes.csv."
        )
    )
    parser.add_argument("--in-dir", type=Path, default=Path("Mesures/Imageur_beta"))
    parser.add_argument("--out-csv", type=Path, default=None, help="Chemin de sortie pour extractions_gaussiennes.csv.")
    args = parser.parse_args()

    in_dir = args.in_dir
    in_dir.mkdir(parents=True, exist_ok=True)
    extraction_csv = args.out_csv if args.out_csv is not None else in_dir / "extractions_gaussiennes.csv"

    fichiers_aggreges: list[tuple[float, float, Path]] = []
    for p in in_dir.glob("*.csv"):
        try:
            tension, debit = parse_aggregated_filename(p)
        except ValueError:
            continue
        fichiers_aggreges.append((tension, debit, p))

    if not fichiers_aggreges:
        raise SystemExit(f"Aucun fichier agrege valide de type tension-debit.csv dans {in_dir}.")

    rows_out: list[dict[str, float | int | str]] = []
    for tension, debit, agg_path in sorted(fichiers_aggreges, key=lambda t: (t[1], t[0])):
        x_raw, y_raw, yerr_raw, _, _ = load_aggregated_curve(agg_path)
        x, y, yerr = crop_curve(x_raw, y_raw, yerr_raw)
        params, params_err, covariance, _, _ = fit_curve(x, y, yerr)
        metrics_fit = derive_metrics(params, params_err, covariance, float(np.min(x)), float(np.max(x)))

        rows_out.append(
            {
                "tension": tension,
                "debit": debit,
                "fichier_agrege": agg_path.name,
                "i_max_fit": metrics_fit["i_max"],
                "i_max_fit_err": metrics_fit["i_max_err"],
                "a1": metrics_fit["a1"],
                "a1_err": metrics_fit["a1_err"],
                "mu1": metrics_fit["mu1"],
                "mu1_err": metrics_fit["mu1_err"],
                "sigma1": metrics_fit["sigma1"],
                "sigma1_err": metrics_fit["sigma1_err"],
                "a2": metrics_fit["a2"],
                "a2_err": metrics_fit["a2_err"],
                "mu2": metrics_fit["mu2"],
                "mu2_err": metrics_fit["mu2_err"],
                "sigma2": metrics_fit["sigma2"],
                "sigma2_err": metrics_fit["sigma2_err"],
            }
        )

    fields = [
        "tension",
        "debit",
        "fichier_agrege",
        "i_max_fit",
        "i_max_fit_err",
        "a1",
        "a1_err",
        "mu1",
        "mu1_err",
        "sigma1",
        "sigma1_err",
        "a2",
        "a2_err",
        "mu2",
        "mu2_err",
        "sigma2",
        "sigma2_err",
    ]
    with extraction_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows_out:
            w.writerow(row)

    print(f"Fichier genere: {extraction_csv}")
    print(f"Nombre de courbes agregees traitees: {len(rows_out)}")


if __name__ == "__main__":
    main()
