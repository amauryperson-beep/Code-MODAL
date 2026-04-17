#!/usr/bin/env python3
"""
Pipeline complet pour l'analyse des paramètres gaussiens.
Exécute automatiquement les trois étapes :
1. Extraction des paramètres des fichiers bruts
2. Génération des graphes
3. Optionnel: affichage d'un paramètre spécifique
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Exécute une commande et retourne True si succès."""
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ {description} - Succès\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} - Erreur (code {e.returncode})\n")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline complet pour l'analyse des paramètres gaussiens.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemple d'utilisation:
  python3 pipeline_complet.py                    # Exécute tout
  python3 pipeline_complet.py --graphs-only      # Génère les graphes uniquement
  python3 pipeline_complet.py --explore a1_mean --debit 40  # Exploration spécifique
        """,
    )
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=Path("Mesures/Imageur_beta"),
        help="Répertoire des données (défaut: Mesures/Imageur_beta)",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=None,
        help="Répertoire de sortie des graphes. Si None, affiche à l'écran.",
    )
    parser.add_argument(
        "--extraction-only",
        action="store_true",
        help="Exécute uniquement l'extraction des paramètres.",
    )
    parser.add_argument(
        "--graphs-only",
        action="store_true",
        help="Génère uniquement les graphes (suppose que l'extraction est faite).",
    )
    parser.add_argument(
        "--explore",
        type=str,
        default=None,
        help="Explore un paramètre spécifique après tout. Ex: 'a1_mean'",
    )
    parser.add_argument(
        "--debit",
        type=str,
        default=None,
        help="Débit à utiliser avec --explore (nombre ou 'all').",
    )
    parser.add_argument(
        "--tension",
        type=str,
        default=None,
        help="Tension à utiliser avec --explore (nombre ou 'all').",
    )
    parser.add_argument(
        "--vs",
        type=str,
        default="tension",
        choices=["tension", "debit"],
        help="Variable d'abscisse pour --explore.",
    )
    
    args = parser.parse_args()
    
    # Déterminer le répertoire des graphes
    if args.graphs_dir is None:
        graphs_dir_str = "(Affichage à l'écran)"
        graphs_dir = None
    else:
        graphs_dir = args.graphs_dir
        graphs_dir_str = str(graphs_dir)
    
    # Déterminer le fichier CSV d'extraction
    csv_path = args.in_dir / "extractions_gaussiennes.csv"
    
    print("\n" + "="*60)
    print("PIPELINE D'ANALYSE DES PARAMÈTRES GAUSSIENS")
    print("="*60)
    print(f"\nConfiguration :")
    print(f"  Répertoire d'entrée: {args.in_dir}")
    print(f"  CSV d'extraction: {csv_path}")
    print(f"  Répertoire des graphes: {graphs_dir_str}")
    
    all_success = True
    
    # Étape 1: Extraction
    if not args.graphs_only:
        cmd = [
            sys.executable,
            "traitement_imageur_beta.py",
            "--in-dir", str(args.in_dir),
            "--out-csv", str(csv_path),
        ]
        if not run_command(cmd, "Étape 1: Extraction des paramètres"):
            all_success = False
            sys.exit(1)
    else:
        if not csv_path.exists():
            print(f"\n✗ Erreur: fichier d'extraction non trouvé: {csv_path}")
            sys.exit(1)
        print(f"\n✓ Utilisation du fichier d'extraction existant: {csv_path}\n")
    
    # Étape 2: Génération des graphes
    if not args.extraction_only:
        cmd = [
            sys.executable,
            "generer_graphes.py",
            "--in-csv", str(csv_path),
        ]
        if graphs_dir:
            cmd.extend(["--out-dir", str(graphs_dir)])
        
        if not run_command(cmd, "Étape 2: Génération des graphes"):
            all_success = False
    
    # Étape 3: Exploration optionnelle
    if args.explore and not args.extraction_only:
        cmd = [
            sys.executable,
            "tracer_parametres.py",
            "--in-csv", str(csv_path),
            "--param", args.explore,
            "--vs", args.vs,
        ]
        if args.debit is not None:
            cmd.extend(["--debit", str(args.debit)])
        if args.tension is not None:
            cmd.extend(["--tension", str(args.tension)])
        
        run_command(cmd, f"Étape 3: Exploration du paramètre '{args.explore}'")
    
    # Résumé
    print("\n" + "="*60)
    if all_success:
        print("✓ PIPELINE COMPLET - SUCCÈS")
    else:
        print("✗ PIPELINE COMPLET - ERREURS")
    print("="*60)
    
    if graphs_dir:
        print(f"\nLes graphes sont disponibles dans: {graphs_dir}")
    
    print(f"Le fichier d'extraction est disponible dans: {csv_path}")
    print("\nProchaines étapes:")
    print(f"  - Examiner les graphes pour identifier les tendances")
    print(f"  - Utiliser 'tracer_parametres.py' pour explorer des paramètres spécifiques")
    print(f"  - Vérifier le CSV pour les valeurs brutes")


if __name__ == "__main__":
    main()
