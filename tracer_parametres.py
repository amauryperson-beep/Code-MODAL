#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def charger_extractions(chemin_csv: Path) -> dict[str, np.ndarray]:
    donnees: dict[str, list[float]] = {}
    with chemin_csv.open("r", encoding="utf-8", newline="") as f:
        lecteur = csv.DictReader(f)
        for ligne in lecteur:
            for cle, valeur in ligne.items():
                if cle not in donnees:
                    donnees[cle] = []
                try:
                    donnees[cle].append(float(valeur) if valeur else np.nan)
                except (ValueError, TypeError):
                    donnees[cle].append(np.nan)
    return {cle: np.array(vals, dtype=float) for cle, vals in donnees.items()}


def lister_parametres(donnees: dict[str, np.ndarray]) -> list[str]:
    params = sorted(cle[:-5] for cle in donnees if cle.endswith("_mean"))
    return params


def _resoudre_parametre(param: str, donnees: dict[str, np.ndarray]) -> str:
    base = param[:-5] if param.endswith("_mean") else param
    key = f"{base}_mean"
    if key not in donnees:
        choix = ", ".join(lister_parametres(donnees))
        raise ValueError(f"Paramètre inconnu: {param}. Choix: {choix}")
    return base


def _parse_filtre(valeur: str | None, nom: str) -> float | None:
    if valeur is None or valeur.lower() == "all":
        return None
    try:
        return float(valeur)
    except ValueError as exc:
        raise ValueError(f"{nom} doit être un nombre ou 'all' (reçu: {valeur}).") from exc


def _fit_lineaire_pondere(x: np.ndarray, y: np.ndarray, yerr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    def lineaire(xx: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * xx + b

    sigma = np.asarray(yerr, dtype=float)
    fallback = max(float(np.nanmedian(np.abs(y - np.nanmedian(y)))), 1e-6)
    sigma = np.where((~np.isfinite(sigma)) | (sigma <= 0), fallback, sigma)
    popt, pcov = curve_fit(lineaire, x, y, sigma=sigma, absolute_sigma=True)
    perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
    resid = y - lineaire(x, *popt)
    chi2 = float(np.sum((resid / sigma) ** 2))
    dof = len(x) - 2
    chi2_red = chi2 / dof if dof > 0 else float("nan")
    return popt, perr, chi2_red


def _tracer_serie_et_fit(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    label_serie: str,
) -> None:
    ordre = np.argsort(x)
    x = x[ordre]
    y = y[ordre]
    yerr = yerr[ordre]
    masque = np.isfinite(x) & np.isfinite(y)
    x = x[masque]
    y = y[masque]
    yerr = yerr[masque]
    if len(x) == 0:
        return

    plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=4, label=label_serie)

    if len(x) >= 2:
        try:
            (a, b), _, chi2_red = _fit_lineaire_pondere(x, y, yerr)
            x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 300)
            y_fit = (a * x_fit) + b
            plt.plot(x_fit, y_fit, "--", linewidth=1.6, label=f"fit {label_serie} (chi2r={chi2_red:.2g})")
        except Exception:
            pass


def tracer_parametre(
    donnees: dict[str, np.ndarray],
    param_base: str,
    vs: str,
    debit: float | None,
    tension: float | None,
    out: Path | None,
) -> None:
    y_key = f"{param_base}_mean"
    yerr_key = f"{param_base}_std"
    y_all = donnees[y_key]
    yerr_all = donnees[yerr_key] if yerr_key in donnees else np.zeros_like(y_all)

    plt.figure(figsize=(9.5, 5.8))

    if vs == "tension":
        if debit is None:
            debits = sorted(np.unique(donnees["debit"]))
            for d in debits:
                mask = np.isclose(donnees["debit"], d, rtol=0.0, atol=1e-9)
                _tracer_serie_et_fit(
                    x=donnees["tension"][mask],
                    y=y_all[mask],
                    yerr=yerr_all[mask],
                    label_serie=f"debit={d:g}",
                )
            titre = f"{param_base} vs tension (tous les débits)"
        else:
            mask = np.isclose(donnees["debit"], debit, rtol=0.0, atol=1e-9)
            if not np.any(mask):
                raise ValueError(f"Aucune donnée pour débit={debit:g}.")
            _tracer_serie_et_fit(
                x=donnees["tension"][mask],
                y=y_all[mask],
                yerr=yerr_all[mask],
                label_serie=f"debit={debit:g}",
            )
            titre = f"{param_base} vs tension (debit={debit:g})"
        plt.xlabel("Tension (V)")
    else:
        if tension is None:
            tensions = sorted(np.unique(donnees["tension"]))
            for t in tensions:
                mask = np.isclose(donnees["tension"], t, rtol=0.0, atol=1e-9)
                _tracer_serie_et_fit(
                    x=donnees["debit"][mask],
                    y=y_all[mask],
                    yerr=yerr_all[mask],
                    label_serie=f"tension={t:g} V",
                )
            titre = f"{param_base} vs débit (toutes les tensions)"
        else:
            mask = np.isclose(donnees["tension"], tension, rtol=0.0, atol=1e-9)
            if not np.any(mask):
                raise ValueError(f"Aucune donnée pour tension={tension:g}.")
            _tracer_serie_et_fit(
                x=donnees["debit"][mask],
                y=y_all[mask],
                yerr=yerr_all[mask],
                label_serie=f"tension={tension:g} V",
            )
            titre = f"{param_base} vs débit (tension={tension:g} V)"
        plt.xlabel("Débit")

    plt.ylabel(param_base)
    plt.title(titre)
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Graphe sauvegardé: {out}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace un paramètre extrait en fonction de la tension ou du débit.")
    parser.add_argument("--in-csv", type=Path, default=Path("Mesures/Imageur_beta/extractions_gaussiennes.csv"))
    parser.add_argument("--list", action="store_true", help="Liste les paramètres disponibles.")
    parser.add_argument("--param", type=str, default=None, help="Paramètre à tracer (ex: sigma_eff ou sigma_eff_mean).")
    parser.add_argument("--vs", type=str, default="tension", choices=["tension", "debit"])
    parser.add_argument("--debit", type=str, default="all", help="Débit cible ou 'all'.")
    parser.add_argument("--tension", type=str, default="all", help="Tension cible ou 'all'.")
    parser.add_argument("--out", type=Path, default=None, help="Fichier PNG de sortie (sinon affichage à l'écran).")
    args = parser.parse_args()

    if not args.in_csv.exists():
        raise SystemExit(f"Erreur: fichier introuvable: {args.in_csv}")

    donnees = charger_extractions(args.in_csv)

    if args.list:
        print("Paramètres disponibles:")
        for p in lister_parametres(donnees):
            print(f" - {p}")
        return

    if args.param is None:
        raise SystemExit("Erreur: préciser --param (ou --list).")

    try:
        param_base = _resoudre_parametre(args.param, donnees)
        debit = _parse_filtre(args.debit, "debit")
        tension = _parse_filtre(args.tension, "tension")
        tracer_parametre(donnees, param_base, args.vs, debit, tension, args.out)
    except ValueError as exc:
        raise SystemExit(f"Erreur: {exc}") from exc


if __name__ == "__main__":
    main()
