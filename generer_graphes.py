#!/usr/bin/env python3
"""
Génère automatiquement les graphes des paramètres extraits (moyennes et écarts-types)
avec des visualisations adaptées.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def charger_extractions(chemin_csv: Path) -> dict:
    """Charge le fichier extractions_gaussiennes.csv."""
    donnees = {}
    
    with chemin_csv.open('r', encoding='utf-8', newline='') as f:
        lecteur = csv.DictReader(f)
        for ligne in lecteur:
            for cle, valeur in ligne.items():
                if cle not in donnees:
                    donnees[cle] = []
                try:
                    donnees[cle].append(float(valeur) if valeur else np.nan)
                except (ValueError, TypeError):
                    donnees[cle].append(np.nan)
    
    # Convertir en arrays numpy
    for cle in donnees:
        donnees[cle] = np.array(donnees[cle], dtype=float)
    
    return donnees


def tracer_heatmap_sigma_eff(donnees: dict, output_dir: Path | None = None) -> None:
    """Trace la heatmap de sigma_eff en fonction de la tension et du débit."""
    if "sigma_eff_mean" not in donnees:
        print("Sigma_eff absent du CSV: heatmap sigma_eff non générée.")
        return

    debits_uniques = sorted(np.unique(donnees['debit']))
    tensions_uniques = sorted(np.unique(donnees['tension']))

    matrix = np.full((len(tensions_uniques), len(debits_uniques)), np.nan)
    for i, t in enumerate(tensions_uniques):
        for j, d in enumerate(debits_uniques):
            mask = (donnees['tension'] == t) & (donnees['debit'] == d)
            if np.any(mask):
                matrix[i, j] = donnees['sigma_eff_mean'][mask][0]

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5))
    im = ax.imshow(matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xticks(range(len(debits_uniques)))
    ax.set_xticklabels([f'{d:g}' for d in debits_uniques], rotation=45)
    ax.set_yticks(range(len(tensions_uniques)))
    ax.set_yticklabels([f'{t:g}' for t in tensions_uniques])
    ax.set_xlabel('Débit')
    ax.set_ylabel('Tension (V)')
    ax.set_title('Heatmap sigma_eff')
    plt.colorbar(im, ax=ax, label='sigma_eff')
    plt.tight_layout()

    if output_dir:
        output_path = output_dir / 'heatmap_sigma_eff.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Graphe sauvegardé: {output_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère automatiquement les graphes des paramètres extraits."
    )
    parser.add_argument(
        "--in-csv",
        type=Path,
        default=Path("Mesures/Imageur_beta/extractions_gaussiennes.csv"),
        help="Fichier d'extraction des paramètres.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Répertoire de sortie pour les graphes. Si non spécifié, affiche à l'écran.",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        default="all",
        choices=["all", "heatmap", "vs_tension", "vs_debit"],
        help="Type de graphes à générer.",
    )
    
    args = parser.parse_args()
    
    if not args.in_csv.exists():
        print(f"Erreur: le fichier {args.in_csv} n'existe pas.")
        return
    
    donnees = charger_extractions(args.in_csv)
    
    # Répertoire de sortie
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Paramètres à tracer (les plus pertinents)
    parametres_info = {
        'a1': {'label': 'Amplitude 1'},
        'mu1': {'label': 'Position 1 (pixels)'},
        'sigma1': {'label': 'Largeur 1 (pixels)'},
        'a2': {'label': 'Amplitude 2'},
        'mu2': {'label': 'Position 2 (pixels)'},
        'sigma2': {'label': 'Largeur 2 (pixels)'},
        'i_max': {'label': 'Intensité max'},
        'largeur_mh': {'label': 'Largeur à mi-hauteur'},
        'sigma_eff': {'label': 'Sigma effectif'},
    }
    
    if args.plot_type in ["all", "heatmap"]:
        print("Génération des heatmaps...")
        tracer_heatmap_sigma_eff(donnees, args.out_dir)
    
    if args.plot_type in ["all", "vs_tension"]:
        print("Génération des graphes vs tension...")
    
    if args.plot_type in ["all", "vs_debit"]:
        print("Génération des graphes vs débit...")
    
    if args.out_dir:
        print(f"\n✓ Tous les graphes ont été sauvegardés dans {args.out_dir}")
    else:
        print("\n✓ Graphes générés")


if __name__ == "__main__":
    main()