from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def _colonnes_numeriques(rows: list[dict[str, str]], fieldnames: list[str]) -> list[str]:
    cols: list[str] = []
    for name in fieldnames:
        ok = False
        for row in rows:
            val = (row.get(name) or "").strip()
            if not val:
                continue
            try:
                float(val)
            except ValueError:
                continue
            ok = True
            break
        if ok:
            cols.append(name)
    return cols


def charger_courbe(path_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    with path_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path_csv}: fichier vide.")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    if not rows:
        raise ValueError(f"{path_csv}: aucune ligne de donnees.")

    cols_num = _colonnes_numeriques(rows, fieldnames)
    if not cols_num:
        raise ValueError(f"{path_csv}: aucune colonne numerique detectee.")

    y_col = None
    y_candidates = ["coups_par_seconde", "coups_1min", "value", "y", "counts", "intensity"]
    for cand in y_candidates:
        if cand in cols_num:
            y_col = cand
            break
    if y_col is None:
        y_col = cols_num[-1]

    excluded_x = {y_col, "repetition", "frame_index", "uncertain_digits"}
    x_col = None
    x_candidates = ["distance_m", "tension_V", "voltage_V", "debit_gaz", "time_s", "x", "energy_keV"]
    for cand in x_candidates:
        if cand in cols_num and cand not in excluded_x:
            x_col = cand
            break
    if x_col is None:
        for c in cols_num:
            if c not in excluded_x:
                x_col = c
                break
    if x_col is None:
        raise ValueError(f"{path_csv}: impossible de determiner l'axe x.")

    x_vals: list[float] = []
    y_vals: list[float] = []
    for row in rows:
        xs = (row.get(x_col) or "").strip()
        ys = (row.get(y_col) or "").strip()
        if not xs or not ys:
            continue
        try:
            x = float(xs)
            y = float(ys)
        except ValueError:
            continue
        if y_col == "coups_1min":
            y /= 60.0
        x_vals.append(x)
        y_vals.append(y)

    if len(x_vals) < 4:
        raise ValueError(f"{path_csv}: pas assez de points valides pour le fit.")

    grouped: dict[float, list[float]] = {}
    for x, y in zip(x_vals, y_vals):
        grouped.setdefault(float(x), []).append(float(y))

    x_unique = np.array(sorted(grouped.keys()), dtype=float)
    y_mean = np.array([np.mean(grouped[x]) for x in x_unique], dtype=float)
    y_std = np.array(
        [np.std(grouped[x], ddof=1) if len(grouped[x]) > 1 else 0.0 for x in x_unique],
        dtype=float,
    )
    return x_unique, y_mean, y_std, x_col, y_col


def somme_gaussiennes(x: np.ndarray, *params: float) -> np.ndarray:
    n = (len(params) - 1) // 3
    c = params[-1]
    y = np.full_like(x, fill_value=c, dtype=float)
    for i in range(n):
        a, mu, sigma = params[3 * i : 3 * i + 3]
        y += a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return y


def _estimer_nb_gaussiennes(y: np.ndarray, max_gauss: int) -> int:
    ptp = float(np.ptp(y))
    prominence = max(0.05 * ptp, 1e-9)
    peaks, _ = find_peaks(y, prominence=prominence)
    if peaks.size == 0:
        return 1
    return int(min(max_gauss, max(1, peaks.size)))


def ajuster_somme_gaussiennes(
    x: np.ndarray,
    y: np.ndarray,
    n_gauss: int | None,
    max_gauss_auto: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if n_gauss is None:
        n_gauss = _estimer_nb_gaussiennes(y, max_gauss=max_gauss_auto)
    n_gauss = max(1, int(n_gauss))

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x_range = max(1e-9, x_max - x_min)
    y_min = float(np.min(y))
    y_ptp = max(1e-9, float(np.ptp(y)))

    peaks, _ = find_peaks(y, prominence=max(0.05 * y_ptp, 1e-9))
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(y))], dtype=int)
    order = np.argsort(y[peaks])[::-1]
    peaks = peaks[order][:n_gauss]
    peaks = peaks[np.argsort(x[peaks])]

    p0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    sigma0 = max(x_range / (8.0 * n_gauss), x_range / 1000.0)
    for i in range(n_gauss):
        p = peaks[min(i, len(peaks) - 1)]
        amp0 = max(1e-9, float(y[p] - y_min))
        mu0 = float(x[p])
        p0.extend([amp0, mu0, sigma0])
        lower.extend([0.0, x_min, x_range / 10000.0])
        upper.extend([10.0 * y_ptp, x_max, 2.0 * x_range])
    p0.append(y_min)
    lower.append(-2.0 * y_ptp)
    upper.append(2.0 * max(y_min + y_ptp, np.max(y)))

    params_opt, cov = curve_fit(
        somme_gaussiennes,
        x,
        y,
        p0=np.array(p0, dtype=float),
        bounds=(np.array(lower, dtype=float), np.array(upper, dtype=float)),
        maxfev=200000,
    )

    n = (len(params_opt) - 1) // 3
    triplets = []
    for i in range(n):
        triplets.append(tuple(params_opt[3 * i : 3 * i + 3]))
    triplets.sort(key=lambda t: t[1])
    params_sorted: list[float] = []
    for a, mu, sigma in triplets:
        params_sorted.extend([float(a), float(mu), float(sigma)])
    params_sorted.append(float(params_opt[-1]))
    params_opt = np.array(params_sorted, dtype=float)
    return params_opt, cov, n_gauss


def evaluer_fit(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> tuple[float, float]:
    y_pred = somme_gaussiennes(x, *params)
    resid = y - y_pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
    return rmse, r2


def sauvegarder_params(path: Path, params: np.ndarray, cov: np.ndarray) -> None:
    n = (len(params) - 1) // 3
    errs = np.sqrt(np.diag(cov)) if cov.size else np.full_like(params, np.nan)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parametre", "valeur", "incertitude"])
        for i in range(n):
            w.writerow([f"A{i+1}", params[3 * i], errs[3 * i]])
            w.writerow([f"mu{i+1}", params[3 * i + 1], errs[3 * i + 1]])
            w.writerow([f"sigma{i+1}", params[3 * i + 2], errs[3 * i + 2]])
        w.writerow(["offset", params[-1], errs[-1]])


def tracer(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    params: np.ndarray,
    x_label: str,
    y_label: str,
    out_png: Path,
) -> None:
    x_dense = np.linspace(float(np.min(x)), float(np.max(x)), 1200)
    y_fit = somme_gaussiennes(x_dense, *params)

    plt.figure(figsize=(9, 5.2))
    if np.any(yerr > 0):
        plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, label="Donnees")
    else:
        plt.plot(x, y, "o", label="Donnees")
    plt.plot(x_dense, y_fit, "-", linewidth=2.0, label="Fit somme de gaussiennes")

    n = (len(params) - 1) // 3
    offset = params[-1]
    for i in range(n):
        a, mu, sigma = params[3 * i : 3 * i + 3]
        comp = a * np.exp(-0.5 * ((x_dense - mu) / sigma) ** 2) + offset / n
        plt.plot(x_dense, comp, "--", linewidth=1.0, label=f"G{i+1}")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Courbe et fit par somme de gaussiennes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace et fit une courbe CSV avec une somme de gaussiennes.")
    parser.add_argument("csv", type=Path, help="Chemin vers le fichier CSV")
    parser.add_argument("--n-gauss", type=int, default=None, help="Nombre de gaussiennes (auto si absent)")
    parser.add_argument("--max-gauss-auto", type=int, default=4, help="Maximum de gaussiennes en auto")
    parser.add_argument("--out-dir", type=Path, default=Path("mesures/Geiger"), help="Dossier de sortie")
    args = parser.parse_args()

    x, y, yerr, x_col, y_col = charger_courbe(args.csv)
    params, cov, n_used = ajuster_somme_gaussiennes(x, y, args.n_gauss, args.max_gauss_auto)
    rmse, r2 = evaluer_fit(x, y, params)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.csv.stem
    out_png = args.out_dir / f"{stem}_fit_gaussiennes.png"
    out_params = args.out_dir / f"{stem}_fit_gaussiennes_parametres.csv"

    y_label = "Coups par seconde" if y_col in {"coups_par_seconde", "coups_1min"} else y_col
    tracer(x, y, yerr, params, x_col, y_label, out_png)
    sauvegarder_params(out_params, params, cov)

    print(f"Fichier: {args.csv}")
    print(f"Points utilises: {len(x)}")
    print(f"Gaussiennes: {n_used}")
    print(f"RMSE: {rmse:.6g}")
    print(f"R2: {r2:.6g}")
    print(f"Figure: {out_png}")
    print(f"Parametres: {out_params}")
    for i in range(n_used):
        a, mu, sigma = params[3 * i : 3 * i + 3]
        print(f"G{i+1}: A={a:.6g}, mu={mu:.6g}, sigma={sigma:.6g}")
    print(f"offset={params[-1]:.6g}")

    plt.show()


if __name__ == "__main__":
    main()
