from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
FIT_SCRIPT = BASE_DIR / "fit_somme_gaussiennes.py"
PATTERN = re.compile(r"^(5\d{3})-40\.csv$")


def list_curves() -> list[Path]:
    files: list[tuple[int, Path]] = []
    for path in BASE_DIR.glob("*.csv"):
        m = PATTERN.match(path.name)
        if not m:
            continue
        files.append((int(m.group(1)), path))
    files.sort(key=lambda t: t[0])
    return [p for _, p in files]


def load_xy(path_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    x_vals: list[float] = []
    y_vals: list[float] = []
    with path_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames) < 2:
            raise ValueError(f"{path_csv.name}: header invalide.")
        x_col = reader.fieldnames[0]
        y_col = reader.fieldnames[1]
        for row in reader:
            x_str = (row.get(x_col) or "").strip()
            y_str = (row.get(y_col) or "").strip()
            if not x_str and not y_str:
                continue
            x_vals.append(float(x_str))
            y_vals.append(float(y_str))
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    idx = np.argsort(x)
    return x[idx], y[idx]


def run_two_gaussian_fit(path_csv: Path) -> Path:
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    subprocess.run(
        [sys.executable, str(FIT_SCRIPT), str(path_csv), "--min-gauss", "2", "--max-gauss", "2"],
        check=True,
        env=env,
    )
    return path_csv.with_name(f"{path_csv.stem}_fit_gaussiennes_parametres.csv")


def read_fit_params(path_params_csv: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path_params_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("section") or "").strip() != "best_model":
                continue
            param = (row.get("param") or "").strip()
            value = (row.get("value") or "").strip()
            if not param or not value:
                continue
            values[param] = float(value)
    return values


def model_y(x: np.ndarray, baseline: float, a1: float, mu1: float, s1: float, a2: float, mu2: float, s2: float) -> np.ndarray:
    g1 = a1 * np.exp(-0.5 * ((x - mu1) / s1) ** 2)
    g2 = a2 * np.exp(-0.5 * ((x - mu2) / s2) ** 2)
    return baseline + g1 + g2


def build_summary_row(path_csv: Path, fit_values: dict[str, float]) -> dict[str, float | str]:
    x, y = load_xy(path_csv)
    max_idx = int(np.argmax(y))
    x_max_data = float(x[max_idx])
    y_max_data = float(y[max_idx])

    baseline = fit_values["baseline"]
    a1 = fit_values["gauss_1_amplitude"]
    mu1 = fit_values["gauss_1_mu"]
    s1 = fit_values["gauss_1_sigma"]
    a2 = fit_values["gauss_2_amplitude"]
    mu2 = fit_values["gauss_2_mu"]
    s2 = fit_values["gauss_2_sigma"]

    x_dense = np.linspace(float(np.min(x)), float(np.max(x)), 3000)
    y_dense = model_y(x_dense, baseline, a1, mu1, s1, a2, mu2, s2)
    fit_max_idx = int(np.argmax(y_dense))
    x_max_fit = float(x_dense[fit_max_idx])
    y_max_fit = float(y_dense[fit_max_idx])

    tension = int(path_csv.stem.split("-")[0])
    return {
        "tension": tension,
        "fichier": path_csv.name,
        "x_max_data_px": x_max_data,
        "max_data_gray": y_max_data,
        "x_max_fit_px": x_max_fit,
        "max_fit_gray": y_max_fit,
        "gauss_1_sigma": fit_values.get("gauss_1_sigma", float("nan")),
        "gauss_1_sigma_error": fit_values.get("gauss_1_sigma_error", float("nan")),
        "gauss_2_sigma": fit_values.get("gauss_2_sigma", float("nan")),
        "gauss_2_sigma_error": fit_values.get("gauss_2_sigma_error", float("nan")),
        "gauss_1_mu": fit_values.get("gauss_1_mu", float("nan")),
        "gauss_1_mu_error": fit_values.get("gauss_1_mu_error", float("nan")),
        "gauss_2_mu": fit_values.get("gauss_2_mu", float("nan")),
        "gauss_2_mu_error": fit_values.get("gauss_2_mu_error", float("nan")),
        "chi2": fit_values.get("chi2", float("nan")),
        "chi2_reduced": fit_values.get("chi2_reduced", float("nan")),
        "sigma_x_px": fit_values.get("sigma_x", float("nan")),
    }


def main() -> None:
    curves = list_curves()
    if not curves:
        raise SystemExit("Aucune courbe 5XX0-40.csv trouvee.")

    rows: list[dict[str, float | str]] = []
    for curve in curves:
        params_path = run_two_gaussian_fit(curve)
        fit_values = read_fit_params(params_path)
        rows.append(build_summary_row(curve, fit_values))

    out_csv = BASE_DIR / "mesures_5XX0-40_resume.csv"
    fieldnames = [
        "tension",
        "fichier",
        "x_max_data_px",
        "max_data_gray",
        "x_max_fit_px",
        "max_fit_gray",
        "gauss_1_sigma",
        "gauss_1_sigma_error",
        "gauss_2_sigma",
        "gauss_2_sigma_error",
        "gauss_1_mu",
        "gauss_1_mu_error",
        "gauss_2_mu",
        "gauss_2_mu_error",
        "chi2",
        "chi2_reduced",
        "sigma_x_px",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"CSV genere: {out_csv}")
    print(f"Nombre de courbes: {len(rows)}")


if __name__ == "__main__":
    main()
