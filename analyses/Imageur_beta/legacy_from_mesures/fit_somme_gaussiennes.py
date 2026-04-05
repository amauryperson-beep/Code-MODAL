from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


@dataclass(frozen=True)
class FitResult:
    n_gauss: int
    baseline: float
    baseline_error: float
    params: np.ndarray
    param_errors: np.ndarray
    rss: float
    aic: float
    bic: float
    sigma_x: float
    chi2: float
    dof: int
    chi2_reduced: float


def load_curve_csv(path_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    with path_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames) < 2:
            raise ValueError(f"{path_csv}: header with at least 2 columns is required.")

        x_col = reader.fieldnames[0]
        y_col = reader.fieldnames[1]

        x_vals: list[float] = []
        y_vals: list[float] = []
        for line_no, row in enumerate(reader, start=2):
            x_str = (row.get(x_col) or "").strip()
            y_str = (row.get(y_col) or "").strip()
            if not x_str and not y_str:
                continue
            if not x_str or not y_str:
                raise ValueError(f"{path_csv}: incomplete row at line {line_no}.")
            x_vals.append(float(x_str))
            y_vals.append(float(y_str))

    if len(x_vals) < 6:
        raise ValueError(f"{path_csv}: not enough data points for gaussian fitting.")

    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    sort_idx = np.argsort(x)
    return x[sort_idx], y[sort_idx]


def multi_gaussian(x: np.ndarray, baseline: float, *params: float) -> np.ndarray:
    y = np.full_like(x, baseline, dtype=float)
    for i in range(0, len(params), 3):
        amp, mu, sigma = params[i : i + 3]
        y += amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return y


def multi_gaussian_derivative(x: np.ndarray, *params: float) -> np.ndarray:
    dy = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 3):
        amp, mu, sigma = params[i : i + 3]
        z = (x - mu) / sigma
        dy += amp * np.exp(-0.5 * z**2) * (-(x - mu) / (sigma**2))
    return dy


def infer_sigma_x_from_sampling(x: np.ndarray) -> float:
    dx = np.diff(np.sort(x))
    dx_pos = dx[dx > 0]
    if dx_pos.size == 0:
        return 1.0
    return float(np.median(dx_pos))


def build_initial_guess(x: np.ndarray, y: np.ndarray, n_gauss: int) -> np.ndarray:
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_span = max(y_max - y_min, 1e-6)

    peaks, _ = find_peaks(y, prominence=0.05 * y_span, distance=max(1, len(x) // (n_gauss + 1)))
    if peaks.size == 0:
        peak_idx = np.array([int(np.argmax(y))], dtype=int)
    else:
        order = np.argsort(y[peaks])[::-1]
        peak_idx = peaks[order]

    # Ensure we always have exactly n_gauss distinct initial centers.
    selected: list[int] = []
    for idx in peak_idx.tolist():
        if idx not in selected:
            selected.append(int(idx))
        if len(selected) == n_gauss:
            break

    if len(selected) < n_gauss:
        evenly_spaced = np.linspace(0, len(x) - 1, max(n_gauss, 2), dtype=int).tolist()
        by_height = np.argsort(y)[::-1].tolist()
        for idx in evenly_spaced + by_height:
            idx_int = int(idx)
            if idx_int not in selected:
                selected.append(idx_int)
            if len(selected) == n_gauss:
                break

    peak_idx = np.asarray(selected[:n_gauss], dtype=int)

    dx = np.diff(x)
    dx_pos = dx[dx > 0]
    min_dx = float(np.min(dx_pos)) if dx_pos.size else 1.0
    span = max(float(np.max(x) - np.min(x)), min_dx)
    sigma0 = max(span / (8.0 * n_gauss), min_dx)

    p0: list[float] = [y_min]
    for idx in peak_idx:
        amp0 = max(float(y[idx] - y_min), 0.1)
        mu0 = float(x[idx])
        p0.extend([amp0, mu0, sigma0])
    return np.asarray(p0, dtype=float)


def fit_with_n_gaussians(x: np.ndarray, y: np.ndarray, n_gauss: int, sigma_x: float) -> FitResult:
    p0 = build_initial_guess(x, y, n_gauss)

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_span = max(y_max - y_min, 1e-6)

    dx = np.diff(x)
    dx_pos = dx[dx > 0]
    min_dx = float(np.min(dx_pos)) if dx_pos.size else 1e-3
    span = max(x_max - x_min, min_dx)

    lower: list[float] = [y_min - 2.0 * y_span]
    upper: list[float] = [y_max + 2.0 * y_span]
    for _ in range(n_gauss):
        lower.extend([0.0, x_min, 0.5 * min_dx])
        upper.extend([10.0 * y_span, x_max, 1.5 * span])

    popt, pcov = curve_fit(
        multi_gaussian,
        x,
        y,
        p0=p0,
        bounds=(np.asarray(lower), np.asarray(upper)),
        maxfev=250000,
    )

    y_hat = multi_gaussian(x, *popt)
    residual = y - y_hat
    rss = float(np.sum(residual**2))

    perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))

    # y uncertainty is declared null; we propagate x uncertainty via local slope:
    # sigma_y,eff = |dy/dx| * sigma_x
    dy_dx = multi_gaussian_derivative(x, *popt[1:])
    sigma_y_eff = np.abs(dy_dx) * float(sigma_x)
    y_scale = max(float(np.max(y) - np.min(y)), 1.0)
    positive_sigma = sigma_y_eff[sigma_y_eff > 0]
    if positive_sigma.size > 0:
        sigma_floor = max(float(np.percentile(positive_sigma, 10)) * 0.1, 1e-6 * y_scale)
    else:
        sigma_floor = 1e-3 * y_scale
    sigma_y_eff = np.maximum(sigma_y_eff, sigma_floor)
    chi2 = float(np.sum((residual / sigma_y_eff) ** 2))

    n_obs = len(x)
    n_params = len(popt)
    safe_rss = max(rss, 1e-12)
    aic = float(2 * n_params + n_obs * np.log(safe_rss / n_obs))
    bic = float(n_params * np.log(n_obs) + n_obs * np.log(safe_rss / n_obs))
    dof = n_obs - n_params
    chi2_reduced = chi2 / dof if dof > 0 else float("nan")

    return FitResult(
        n_gauss=n_gauss,
        baseline=float(popt[0]),
        baseline_error=float(perr[0]),
        params=np.asarray(popt[1:], dtype=float),
        param_errors=np.asarray(perr[1:], dtype=float),
        rss=rss,
        aic=aic,
        bic=bic,
        sigma_x=float(sigma_x),
        chi2=chi2,
        dof=dof,
        chi2_reduced=chi2_reduced,
    )


def select_best_fit(
    x: np.ndarray,
    y: np.ndarray,
    min_gauss: int,
    max_gauss: int,
    criterion: str,
    sigma_x: float,
) -> tuple[FitResult, list[FitResult]]:
    all_results: list[FitResult] = []
    for n_gauss in range(min_gauss, max_gauss + 1):
        try:
            all_results.append(fit_with_n_gaussians(x, y, n_gauss, sigma_x=sigma_x))
        except Exception:
            continue

    if not all_results:
        raise RuntimeError("All gaussian fits failed.")

    if criterion == "aic":
        best = min(all_results, key=lambda r: r.aic)
    elif criterion == "bic":
        best = min(all_results, key=lambda r: r.bic)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    return best, all_results


def build_fit_label(result: FitResult) -> str:
    lines = [
        f"Fit {result.n_gauss} gaussiennes",
        f"b={result.baseline:.3g} +/- {result.baseline_error:.2g}",
    ]
    for i in range(result.n_gauss):
        amp, mu, sigma = result.params[3 * i : 3 * i + 3]
        e_amp, e_mu, e_sigma = result.param_errors[3 * i : 3 * i + 3]
        lines.append(
            f"G{i + 1}: A={amp:.3g} +/- {e_amp:.2g}, "
            f"mu={mu:.3g} +/- {e_mu:.2g}, "
            f"s={sigma:.3g} +/- {e_sigma:.2g}"
        )
    lines.append(f"chi2={result.chi2:.3g}, chi2_red={result.chi2_reduced:.3g}")
    return "\n".join(lines)


def fit_title(n_gauss: int) -> str:
    if n_gauss == 2:
        return "Variation de la tension avec fit en double gaussiennes"
    return f"Variation de la tension avec fit en {n_gauss} gaussiennes"


def plot_raw_curve(
    x: np.ndarray,
    y: np.ndarray,
    sigma_x: float,
    path_png: Path | None = None,
) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.errorbar(x, y, xerr=sigma_x, fmt="o-", linewidth=1.7, markersize=4.5, capsize=3, label="Donnees")
    plt.xlabel("Distance (pixels)")
    plt.ylabel("Gray value")
    plt.title("Variation de la tension")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if path_png is not None:
        plt.savefig(path_png, dpi=180)


def plot_fit(
    x: np.ndarray,
    y: np.ndarray,
    result: FitResult,
    sigma_x: float,
    path_png: Path | None = None,
) -> None:
    x_dense = np.linspace(float(np.min(x)), float(np.max(x)), 1200)
    all_params = np.concatenate([[result.baseline], result.params])
    y_fit_dense = multi_gaussian(x_dense, *all_params)

    plt.figure(figsize=(9, 5))
    plt.errorbar(x, y, xerr=sigma_x, fmt="o", markersize=4.5, capsize=3, label="Donnees")
    plt.plot(x_dense, y_fit_dense, "-", linewidth=2.0, label=build_fit_label(result))

    plt.xlabel("Distance (pixels)")
    plt.ylabel("Gray value")
    plt.title(fit_title(result.n_gauss))
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()
    if path_png is not None:
        plt.savefig(path_png, dpi=180)


def save_parameters(path_csv: Path, best: FitResult, all_results: list[FitResult]) -> None:
    with path_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "n_gauss", "param", "value"])
        writer.writerow(["best_model", best.n_gauss, "baseline", f"{best.baseline:.12g}"])
        writer.writerow(["best_model", best.n_gauss, "baseline_error", f"{best.baseline_error:.12g}"])
        writer.writerow(["best_model", best.n_gauss, "rss", f"{best.rss:.12g}"])
        writer.writerow(["best_model", best.n_gauss, "aic", f"{best.aic:.12g}"])
        writer.writerow(["best_model", best.n_gauss, "bic", f"{best.bic:.12g}"])
        writer.writerow(["best_model", best.n_gauss, "sigma_x", f"{best.sigma_x:.12g}"])
        writer.writerow(["best_model", best.n_gauss, "chi2", f"{best.chi2:.12g}"])
        writer.writerow(["best_model", best.n_gauss, "dof", f"{best.dof:.12g}"])
        writer.writerow(["best_model", best.n_gauss, "chi2_reduced", f"{best.chi2_reduced:.12g}"])
        for i in range(best.n_gauss):
            amp, mu, sigma = best.params[3 * i : 3 * i + 3]
            e_amp, e_mu, e_sigma = best.param_errors[3 * i : 3 * i + 3]
            writer.writerow(["best_model", best.n_gauss, f"gauss_{i + 1}_amplitude", f"{amp:.12g}"])
            writer.writerow(["best_model", best.n_gauss, f"gauss_{i + 1}_amplitude_error", f"{e_amp:.12g}"])
            writer.writerow(["best_model", best.n_gauss, f"gauss_{i + 1}_mu", f"{mu:.12g}"])
            writer.writerow(["best_model", best.n_gauss, f"gauss_{i + 1}_mu_error", f"{e_mu:.12g}"])
            writer.writerow(["best_model", best.n_gauss, f"gauss_{i + 1}_sigma", f"{sigma:.12g}"])
            writer.writerow(["best_model", best.n_gauss, f"gauss_{i + 1}_sigma_error", f"{e_sigma:.12g}"])

        writer.writerow([])
        writer.writerow(["section", "n_gauss", "param", "value"])
        for r in all_results:
            writer.writerow(["candidate_model", r.n_gauss, "baseline", f"{r.baseline:.12g}"])
            writer.writerow(["candidate_model", r.n_gauss, "baseline_error", f"{r.baseline_error:.12g}"])
            writer.writerow(["candidate_model", r.n_gauss, "rss", f"{r.rss:.12g}"])
            writer.writerow(["candidate_model", r.n_gauss, "aic", f"{r.aic:.12g}"])
            writer.writerow(["candidate_model", r.n_gauss, "bic", f"{r.bic:.12g}"])
            writer.writerow(["candidate_model", r.n_gauss, "sigma_x", f"{r.sigma_x:.12g}"])
            writer.writerow(["candidate_model", r.n_gauss, "chi2", f"{r.chi2:.12g}"])
            writer.writerow(["candidate_model", r.n_gauss, "dof", f"{r.dof:.12g}"])
            writer.writerow(["candidate_model", r.n_gauss, "chi2_reduced", f"{r.chi2_reduced:.12g}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot and fit a curve with a sum of gaussians.")
    parser.add_argument("csv_path", type=Path, help="Input CSV with two columns (x, y).")
    parser.add_argument("--min-gauss", type=int, default=1, help="Minimum number of gaussians to test.")
    parser.add_argument("--max-gauss", type=int, default=4, help="Maximum number of gaussians to test.")
    parser.add_argument(
        "--sigma-x",
        type=float,
        default=None,
        help="Uncertainty on x (pixel size). Default: inferred from sampling step.",
    )
    parser.add_argument(
        "--criterion",
        choices=["aic", "bic"],
        default="bic",
        help="Model selection criterion (default: bic).",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save PNG plots next to the CSV file.",
    )
    args = parser.parse_args()

    if args.min_gauss < 1 or args.max_gauss < args.min_gauss:
        raise ValueError("Invalid gaussian range.")

    x, y = load_curve_csv(args.csv_path)
    sigma_x = float(args.sigma_x) if args.sigma_x is not None else infer_sigma_x_from_sampling(x)
    best, all_results = select_best_fit(x, y, args.min_gauss, args.max_gauss, args.criterion, sigma_x)

    stem = args.csv_path.stem
    out_raw = args.csv_path.with_name(f"{stem}_courbe.png") if args.save_plots else None
    out_fit = args.csv_path.with_name(f"{stem}_fit_gaussiennes.png") if args.save_plots else None
    out_params = args.csv_path.with_name(f"{stem}_fit_gaussiennes_parametres.csv")

    plot_raw_curve(x, y, sigma_x, out_raw)
    plot_fit(x, y, best, sigma_x, out_fit)
    save_parameters(out_params, best, all_results)

    if args.save_plots:
        print("Saved files:")
        print(f"- {out_raw}")
        print(f"- {out_fit}")
        print(f"- {out_params}")
    else:
        print("Saved files:")
        print(f"- {out_params}")
    print("")
    print(f"Best model ({args.criterion.upper()}): {best.n_gauss} gaussian(s)")
    print(f"sigma_x = {best.sigma_x:.6g}")
    print(f"baseline = {best.baseline:.6g} +/- {best.baseline_error:.2g}")
    for i in range(best.n_gauss):
        amp, mu, sigma = best.params[3 * i : 3 * i + 3]
        e_amp, e_mu, e_sigma = best.param_errors[3 * i : 3 * i + 3]
        print(
            f"G{i + 1}: A={amp:.6g} +/- {e_amp:.2g}, "
            f"mu={mu:.6g} +/- {e_mu:.2g}, "
            f"sigma={sigma:.6g} +/- {e_sigma:.2g}"
        )
    print(f"chi2 = {best.chi2:.6g}")
    print(f"dof = {best.dof}")
    print(f"chi2_reduced = {best.chi2_reduced:.6g}")

    plt.show()


if __name__ == "__main__":
    main()
