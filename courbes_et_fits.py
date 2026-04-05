from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


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


def list_5xx0_40(base_dir: Path) -> list[Path]:
    rgx = re.compile(r"^(5\d{3})-40\.csv$")
    items: list[tuple[int, Path]] = []
    for p in base_dir.glob("*.csv"):
        m = rgx.match(p.name)
        if m:
            items.append((int(m.group(1)), p))
    items.sort(key=lambda t: t[0])
    return [p for _, p in items]


def summarize_5xx0_40(base_dir: Path, out_csv: Path, out_params_dir: Path) -> Path:
    curves = list_5xx0_40(base_dir)
    if not curves:
        raise ValueError(f"No 5XX0-40 curves found in {base_dir}")
    out_params_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for curve in curves:
        x, y, _, _ = load_two_col_csv(curve)
        fit = fit_double_gaussian(x, y)
        save_fit_params(out_params_dir / f"{curve.stem}_fit_gaussiennes_parametres.csv", fit)

        imax_idx = int(np.argmax(y))
        x_max_data = float(x[imax_idx])
        max_data = float(y[imax_idx])

        x_dense = np.linspace(float(np.min(x)), float(np.max(x)), 3000)
        y_dense = multi_gaussian(x_dense, fit.baseline, *fit.params)
        i_fit = int(np.argmax(y_dense))
        x_max_fit = float(x_dense[i_fit])
        max_fit = float(y_dense[i_fit])
        y_err = max(fit.rmse, 1e-9)

        t = int(curve.stem.split("-")[0])
        rows.append(
            {
                "tension": t,
                "fichier": curve.name,
                "x_max_data_px": x_max_data,
                "max_data_gray": max_data,
                "max_data_gray_error": y_err,
                "x_max_fit_px": x_max_fit,
                "max_fit_gray": max_fit,
                "gauss_1_sigma": float(fit.params[2]),
                "gauss_1_sigma_error": float(fit.params_err[2]),
                "gauss_2_sigma": float(fit.params[5]),
                "gauss_2_sigma_error": float(fit.params_err[5]),
                "chi2": fit.chi2,
                "chi2_reduced": fit.chi2_reduced,
                "sigma_x_px": fit.sigma_x,
            }
        )

    rows.sort(key=lambda r: int(r["tension"]))
    fields = [
        "tension",
        "fichier",
        "x_max_data_px",
        "max_data_gray",
        "max_data_gray_error",
        "x_max_fit_px",
        "max_fit_gray",
        "gauss_1_sigma",
        "gauss_1_sigma_error",
        "gauss_2_sigma",
        "gauss_2_sigma_error",
        "chi2",
        "chi2_reduced",
        "sigma_x_px",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return out_csv


def fit_linear(x: np.ndarray, y: np.ndarray, yerr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    def linear(xx: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * xx + b

    sigma = np.asarray(yerr, dtype=float)
    sigma = np.where(sigma <= 0, np.max([np.median(np.abs(y - np.median(y))), 1e-6]), sigma)
    popt, pcov = curve_fit(linear, x, y, sigma=sigma, absolute_sigma=True)
    return popt, np.sqrt(np.maximum(np.diag(pcov), 0.0))


def plot_intensity_max_vs_tension(summary_csv: Path, out_png: Path | None = None) -> None:
    tensions: list[float] = []
    imax: list[float] = []
    ierr: list[float] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            tensions.append(float(row["tension"]))
            imax.append(float(row["max_data_gray"]))
            ierr.append(float(row["max_data_gray_error"]))

    x = np.asarray(tensions, dtype=float)
    y = np.asarray(imax, dtype=float)
    yerr = np.asarray(ierr, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    yerr = yerr[order]

    (a, b), (ea, eb) = fit_linear(x, y, yerr)
    x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 400)
    y_fit = a * x_fit + b
    legend = f"Fit lineaire: I = aU + b\na={a:.4g} +/- {ea:.2g}, b={b:.4g} +/- {eb:.2g}"

    plt.figure(figsize=(8, 5))
    plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=4, label="Maxima (donnees)")
    plt.plot(x_fit, y_fit, "-", linewidth=2, label=legend)
    plt.xlabel("Tension")
    plt.ylabel("Intensite maximale (Gray value)")
    plt.title("Intensite maximale en fonction de la tension")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=9)
    plt.tight_layout()
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=180)
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

    p_max = sub.add_parser("maxima", help="Generer resume 5XX0-40 et tracer Imax=f(tension) + fit lineaire.")
    p_max.add_argument("--base-dir", type=Path, default=Path("mesures/Imageur_beta"))
    p_max.add_argument("--out-dir", type=Path, default=Path("analyses/Imageur_beta"))
    p_max.add_argument("--no-save-plot", action="store_true")

    args = parser.parse_args()

    if args.cmd == "single":
        x, y, x_name, y_name = load_two_col_csv(args.csv_path)
        fit = fit_double_gaussian(x, y, sigma_x=args.sigma_x)
        if args.save_params is not None:
            args.save_params.parent.mkdir(parents=True, exist_ok=True)
            save_fit_params(args.save_params, fit)
        plot_single_curve(x, y, fit, x_name, y_name, args.title)
        return

    summary_csv = args.out_dir / "mesures_5XX0-40_resume.csv"
    params_dir = args.out_dir / "fit_params"
    summary_csv = summarize_5xx0_40(args.base_dir, summary_csv, params_dir)
    out_png = None if args.no_save_plot else args.out_dir / "intensite_max_vs_tension.png"
    plot_intensity_max_vs_tension(summary_csv, out_png=out_png)
    print(f"Resume CSV: {summary_csv}")
    if out_png is not None:
        print(f"Figure: {out_png}")


if __name__ == "__main__":
    main()
