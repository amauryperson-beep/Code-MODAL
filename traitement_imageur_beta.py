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

FWHM_FACTOR = 2.3548200450309493


def parse_raw_filename(path_csv: Path) -> tuple[float, float, int]:
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\-([0-9]+(?:\.[0-9]+)?)\-([0-9]+)\.csv$", path_csv.name)
    if not m:
        raise ValueError(f"Nom invalide: {path_csv.name} (attendu: tension-debit-numero.csv)")
    return float(m.group(1)), float(m.group(2)), int(m.group(3))


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


def crop_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones_like(x, dtype=bool)
    if PIXEL_MIN is not None:
        mask &= x >= float(PIXEL_MIN)
    if PIXEL_MAX is not None:
        mask &= x <= float(PIXEL_MAX)
    xc = x[mask]
    yc = y[mask]
    if xc.size < 6:
        raise ValueError("Pas assez de points apres decoupe pixel.")
    return xc, yc


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


def sort_by_mu(params: np.ndarray, perr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g1 = (params[1:4], perr[1:4])
    g2 = (params[4:7], perr[4:7])
    pairs = [g1, g2]
    pairs.sort(key=lambda pair: pair[0][1])
    p_sorted = np.concatenate([[params[0]], pairs[0][0], pairs[1][0]])
    e_sorted = np.concatenate([[perr[0]], pairs[0][1], pairs[1][1]])
    return p_sorted, e_sorted


def fit_curve(x: np.ndarray, y: np.ndarray, yerr: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float, float]:
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
    perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
    params, params_err = sort_by_mu(popt, perr)

    y_hat = multi_gaussian_2(x, *params)
    resid = y - y_hat
    chi2 = float(np.sum((resid / sigma) ** 2))
    dof = len(x) - len(params)
    chi2_red = chi2 / dof if dof > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(resid**2)))
    return params, params_err, chi2_red, rmse


def derive_metrics(params: np.ndarray, params_err: np.ndarray, x_min: float, x_max: float) -> dict[str, float]:
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

    fwhm = FWHM_FACTOR * sigma_dom
    fwhm_err = FWHM_FACTOR * sigma_dom_err

    x_dense = np.linspace(float(x_min), float(x_max), 3000)
    y_dense = multi_gaussian_2(x_dense, *params)
    i_max = float(np.max(y_dense))
    x_i_max = float(x_dense[int(np.argmax(y_dense))])

    return {
        "i_max": i_max,
        "x_i_max": x_i_max,
        "sigma_dom": sigma_dom,
        "sigma_dom_err": sigma_dom_err,
        "fwhm": fwhm,
        "fwhm_err": fwhm_err,
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
        x, y = crop_curve(x_raw, y_raw)
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
            "Agrege les mesures imageur beta depuis Mesures/Imageur_beta/Données brutes "
            "et produit les courbes agregees + extractions_gaussiennes.csv."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("Mesures/Imageur_beta/Données brutes"))
    parser.add_argument("--out-dir", type=Path, default=Path("Mesures/Imageur_beta"))
    parser.add_argument("--out-csv", type=Path, default=None, help="Chemin de sortie pour extractions_gaussiennes.csv")
    args = parser.parse_args()

    raw_dir = args.raw_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    extraction_csv = args.out_csv if args.out_csv is not None else out_dir / "extractions_gaussiennes.csv"

    groups: dict[tuple[float, float], list[tuple[int, Path]]] = defaultdict(list)
    for p in raw_dir.glob("*.csv"):
        try:
            tension, debit, numero = parse_raw_filename(p)
        except ValueError:
            continue
        groups[(tension, debit)].append((numero, p))

    if not groups:
        raise SystemExit(f"Aucun fichier brut valide dans {raw_dir}.")

    rows_out: list[dict[str, float | int | str]] = []
    for (tension, debit), items in sorted(groups.items(), key=lambda t: (t[0][1], t[0][0])):
        items.sort(key=lambda t: t[0])
        paths = [p for _, p in items]

        x, y_mean, y_std, n_points, x_name, y_name = aggregate_group(paths)
        agg_path = out_dir / pair_filename(tension, debit)
        write_aggregated_curve(agg_path, x, y_mean, y_std, n_points, x_name, y_name)

        params, params_err, chi2_red, rmse = fit_curve(x, y_mean, y_std)
        metrics_fit = derive_metrics(params, params_err, float(np.min(x)), float(np.max(x)))

        i_max_rep: list[float] = []
        fwhm_rep: list[float] = []
        for p in paths:
            xr, yr, _, _ = load_two_col_csv(p)
            xr, yr = crop_curve(xr, yr)
            try:
                pr, er, _, _ = fit_curve(xr, yr, None)
                mr = derive_metrics(pr, er, float(np.min(xr)), float(np.max(xr)))
                i_max_rep.append(mr["i_max"])
                fwhm_rep.append(mr["fwhm"])
            except Exception:
                continue

        n_mes = len(paths)
        i_max_mean = float(np.mean(i_max_rep)) if i_max_rep else metrics_fit["i_max"]
        fwhm_mean = float(np.mean(fwhm_rep)) if fwhm_rep else metrics_fit["fwhm"]
        i_max_std = float(np.std(i_max_rep, ddof=1)) if len(i_max_rep) > 1 else 0.0
        fwhm_std = float(np.std(fwhm_rep, ddof=1)) if len(fwhm_rep) > 1 else 0.0
        i_max_err_used = i_max_std if i_max_std > 0 else max(rmse, 1e-9)
        fwhm_err_used = fwhm_std if fwhm_std > 0 else max(metrics_fit["fwhm_err"], 1e-9)

        rows_out.append(
            {
                "tension": tension,
                "debit": debit,
                "n_mesures": n_mes,
                "fichier_agrege": agg_path.name,
                "i_max_mean": i_max_mean,
                "i_max_std": i_max_std,
                "i_max_err_utilisee": i_max_err_used,
                "fwhm_mean": fwhm_mean,
                "fwhm_std": fwhm_std,
                "fwhm_err_utilisee": fwhm_err_used,
                "baseline": metrics_fit["baseline"],
                "baseline_err": metrics_fit["baseline_err"],
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
                "sigma_dominante": metrics_fit["sigma_dom"],
                "sigma_dominante_err": metrics_fit["sigma_dom_err"],
                "chi2_reduced": chi2_red,
                "rmse": rmse,
                "pixel_min": "" if PIXEL_MIN is None else PIXEL_MIN,
                "pixel_max": "" if PIXEL_MAX is None else PIXEL_MAX,
            }
        )

    fields = [
        "tension",
        "debit",
        "n_mesures",
        "fichier_agrege",
        "i_max_mean",
        "i_max_std",
        "i_max_err_utilisee",
        "fwhm_mean",
        "fwhm_std",
        "fwhm_err_utilisee",
        "baseline",
        "baseline_err",
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
        "sigma_dominante",
        "sigma_dominante_err",
        "chi2_reduced",
        "rmse",
        "pixel_min",
        "pixel_max",
    ]
    with extraction_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows_out:
            w.writerow(row)

    print(f"Fichier genere: {extraction_csv}")
    print(f"Nombre de couples (tension, debit): {len(rows_out)}")


if __name__ == "__main__":
    main()
