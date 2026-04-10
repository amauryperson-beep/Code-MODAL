from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


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
        masque = np.ones_like(x, dtype=bool)
        if log_x:
            masque &= x > 0
        if log_y:
            x = x[masque]
            y = y[masque]
            yerr = yerr[masque]
            masque = y > 0
        x = x[masque]
        y = y[masque]
        yerr = yerr[masque]
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
            masque = y > 0
            x = x[masque]
            y = y[masque]
            yerr = yerr[masque]
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


def _load_extractions(path_csv: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            i_max_key = "i_max_fit" if "i_max_fit" in row else "i_max_mean"
            i_max_err_key = "i_max_fit_err" if "i_max_fit_err" in row else "i_max_err_utilisee"
            largeur_key = "largeur_mh_fit" if "largeur_mh_fit" in row else ("largeur_mh_mean" if "largeur_mh_mean" in row else None)
            largeur_err_key = (
                "largeur_mh_fit_err"
                if "largeur_mh_fit_err" in row
                else ("largeur_mh_err_utilisee" if "largeur_mh_err_utilisee" in row else None)
            )
            largeur_mh_mean = float(row[largeur_key]) if largeur_key is not None else float("nan")
            largeur_mh_err = float(row[largeur_err_key]) if largeur_err_key is not None else float("nan")
            rows.append(
                {
                    "tension": float(row["tension"]),
                    "debit": float(row["debit"]),
                    "i_max_mean": float(row[i_max_key]),
                    "i_max_err": float(row[i_max_err_key]),
                    "largeur_mh_mean": largeur_mh_mean,
                    "largeur_mh_err": largeur_mh_err,
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


def plot_largeur_mh_vs_tension(extractions_csv: Path, debit: float) -> None:
    rows = [r for r in _load_extractions(extractions_csv) if np.isclose(r["debit"], debit, rtol=0.0, atol=1e-9)]
    if len(rows) < 2:
        raise ValueError(f"Pas assez de points pour debit={debit:g} (minimum 2 tensions).")
    x = np.array([r["tension"] for r in rows], dtype=float)
    y = np.array([r["largeur_mh_mean"] for r in rows], dtype=float)
    yerr = np.array([r["largeur_mh_err"] for r in rows], dtype=float)
    if np.isnan(y).all():
        raise ValueError("Les colonnes largeur_mh_* ne sont pas presentes dans le CSV d'extraction.")
    _plot_imageur(
        x=x,
        y=y,
        yerr=yerr,
        x_label="Tension",
        y_label="largeur_mh (pixels)",
        titre=f"largeur_mh en fonction de la tension (debit={debit:g})",
        label_donnees=f"largeur_mh moyenne (debit={debit:g})",
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


def plot_largeur_mh_vs_debit(extractions_csv: Path, tension: float) -> None:
    rows = [r for r in _load_extractions(extractions_csv) if np.isclose(r["tension"], tension, rtol=0.0, atol=1e-9)]
    if len(rows) < 2:
        raise ValueError(f"Pas assez de points pour tension={tension:g} (minimum 2 debits).")
    x = np.array([r["debit"] for r in rows], dtype=float)
    y = np.array([r["largeur_mh_mean"] for r in rows], dtype=float)
    yerr = np.array([r["largeur_mh_err"] for r in rows], dtype=float)
    if np.isnan(y).all():
        raise ValueError("Les colonnes largeur_mh_* ne sont pas presentes dans le CSV d'extraction.")
    _plot_imageur(
        x=x,
        y=y,
        yerr=yerr,
        x_label="Debit",
        y_label="largeur_mh (pixels)",
        titre=f"largeur_mh en fonction du debit (tension={tension:g})",
        label_donnees=f"largeur_mh moyenne (tension={tension:g})",
    )


def plot_sigma_metrics_vs_tension(extractions_csv: Path, debit: float, a: float) -> None:
    if not (0.0 <= a <= 1.0):
        raise ValueError(f"Le parametre a doit etre entre 0 et 1 (recu: {a:g}).")

    rows = [r for r in _load_extractions(extractions_csv) if np.isclose(r["debit"], debit, rtol=0.0, atol=1e-9)]
    if len(rows) < 2:
        raise ValueError(f"Pas assez de points pour debit={debit:g} (minimum 2 tensions).")

    x = np.array([r["tension"] for r in rows], dtype=float)
    sigma1 = np.array([r["sigma1"] for r in rows], dtype=float)
    sigma1_err = np.array([r["sigma1_err"] for r in rows], dtype=float)
    sigma2 = np.array([r["sigma2"] for r in rows], dtype=float)
    sigma2_err = np.array([r["sigma2_err"] for r in rows], dtype=float)

    if np.isnan(sigma1).all() or np.isnan(sigma2).all():
        raise ValueError("Les colonnes sigma1/sigma2 ne sont pas presentes dans le CSV d'extraction.")

    metrique_1 = sigma1
    metrique_1_err = sigma1_err
    metrique_2 = sigma2
    metrique_2_err = sigma2_err
    metrique_3 = (a * sigma1) + ((1.0 - a) * sigma2)
    metrique_3_err = np.sqrt(((a * sigma1_err) ** 2) + (((1.0 - a) * sigma2_err) ** 2))

    # Propagation d'erreur (hypothese d'independance):
    # m4 = sigma1^a * sigma2^(1-a)
    # (dm4/m4)^2 = (a*ds1/s1)^2 + ((1-a)*ds2/s2)^2
    metrique_4 = np.full_like(sigma1, np.nan, dtype=float)
    metrique_4_err = np.full_like(sigma1, np.nan, dtype=float)
    masque_pos = (sigma1 > 0.0) & (sigma2 > 0.0)
    if np.any(masque_pos):
        s1p = sigma1[masque_pos]
        s2p = sigma2[masque_pos]
        es1p = sigma1_err[masque_pos]
        es2p = sigma2_err[masque_pos]
        m4 = (s1p**a) * (s2p ** (1.0 - a))
        rel = np.sqrt(((a * es1p / s1p) ** 2) + ((((1.0 - a) * es2p / s2p) ** 2)))
        metrique_4[masque_pos] = m4
        metrique_4_err[masque_pos] = m4 * rel

    ordre = np.argsort(x)
    x = x[ordre]
    metrique_1 = metrique_1[ordre]
    metrique_1_err = metrique_1_err[ordre]
    metrique_2 = metrique_2[ordre]
    metrique_2_err = metrique_2_err[ordre]
    metrique_3 = metrique_3[ordre]
    metrique_3_err = metrique_3_err[ordre]
    metrique_4 = metrique_4[ordre]
    metrique_4_err = metrique_4_err[ordre]

    plt.figure(figsize=(9, 5.5))
    plt.errorbar(x, metrique_1, yerr=metrique_1_err, fmt="o-", capsize=4, label="metrique_1 = sigma_1")
    plt.errorbar(x, metrique_2, yerr=metrique_2_err, fmt="s-", capsize=4, label="metrique_2 = sigma_2")
    plt.errorbar(
        x,
        metrique_3,
        yerr=metrique_3_err,
        fmt="^-",
        capsize=4,
        label=f"metrique_3 = a*sigma_1 + (1-a)*sigma_2 (a={a:g})",
    )
    plt.errorbar(
        x,
        metrique_4,
        yerr=metrique_4_err,
        fmt="d-",
        capsize=4,
        label=f"metrique_4 = sigma_1^a * sigma_2^(1-a) (a={a:g})",
    )
    plt.xlabel("Tension")
    plt.ylabel("Valeur de metrique (pixels)")
    plt.title(f"Metriques sigma en fonction de la tension (debit={debit:g})")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Affichage des courbes (Geiger/Imageur beta).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("imageur-i-max-vs-tension")
    p1.add_argument("--debit", type=float, required=True)
    p1.add_argument("--csv", type=Path, default=Path("Mesures/Imageur_beta/extractions_gaussiennes.csv"))

    p2 = sub.add_parser("imageur-largeur-mh-vs-tension")
    p2.add_argument("--debit", type=float, required=True)
    p2.add_argument("--csv", type=Path, default=Path("Mesures/Imageur_beta/extractions_gaussiennes.csv"))

    p3 = sub.add_parser("imageur-i-max-vs-debit")
    p3.add_argument("--tension", type=float, required=True)
    p3.add_argument("--csv", type=Path, default=Path("Mesures/Imageur_beta/extractions_gaussiennes.csv"))

    p4 = sub.add_parser("imageur-largeur-mh-vs-debit")
    p4.add_argument("--tension", type=float, required=True)
    p4.add_argument("--csv", type=Path, default=Path("Mesures/Imageur_beta/extractions_gaussiennes.csv"))

    p5 = sub.add_parser("imageur-metriques-vs-tension")
    p5.add_argument("--debit", type=float, required=True)
    p5.add_argument("--a", type=float, default=0.5)
    p5.add_argument("--csv", type=Path, default=Path("Mesures/Imageur_beta/extractions_gaussiennes.csv"))

    args = parser.parse_args()
    try:
        if args.cmd == "imageur-i-max-vs-tension":
            plot_i_max_vs_tension(args.csv, args.debit)
        elif args.cmd == "imageur-largeur-mh-vs-tension":
            plot_largeur_mh_vs_tension(args.csv, args.debit)
        elif args.cmd == "imageur-i-max-vs-debit":
            plot_i_max_vs_debit(args.csv, args.tension)
        elif args.cmd == "imageur-largeur-mh-vs-debit":
            plot_largeur_mh_vs_debit(args.csv, args.tension)
        elif args.cmd == "imageur-metriques-vs-tension":
            plot_sigma_metrics_vs_tension(args.csv, args.debit, args.a)
    except ValueError as exc:
        raise SystemExit(f"Erreur: {exc}") from exc


if __name__ == "__main__":
    main()
