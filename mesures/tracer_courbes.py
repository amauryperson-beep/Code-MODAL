from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
MESURES_DIR = BASE_DIR / "mesures"
SORTIES_DIR = BASE_DIR / "sorties"


def lire_mesures_fichier(path_csv: Path, nom_colonne_x: str) -> list[tuple[float, float]]:
    mesures: list[tuple[float, float]] = []

    with path_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path_csv.name}: fichier vide ou en-tete manquant.")

        champs = set(reader.fieldnames)
        if nom_colonne_x not in champs or "coups_1min" not in champs:
            raise ValueError(
                f"{path_csv.name}: colonnes attendues '{nom_colonne_x}' et 'coups_1min'."
            )

        for i, row in enumerate(reader, start=2):
            x_str = (row.get(nom_colonne_x) or "").strip()
            coups_1min_str = (row.get("coups_1min") or "").strip()

            if not x_str and not coups_1min_str:
                continue
            if not x_str or not coups_1min_str:
                raise ValueError(
                    f"{path_csv.name}: ligne {i} incomplete. "
                    f"Renseigner '{nom_colonne_x}' et 'coups_1min'."
                )

            x = float(x_str)
            coups_1min = float(coups_1min_str)
            mesures.append((x, coups_1min / 60.0))

    if not mesures:
        raise ValueError(f"{path_csv.name}: aucune mesure trouvee.")

    return mesures


def lire_mesures_dossier(path_dossier: Path, nom_colonne_x: str) -> tuple[list[float], list[float], list[float]]:
    fichiers_csv = sorted(path_dossier.glob("*.csv"))
    if not fichiers_csv:
        raise ValueError(f"{path_dossier}: aucun fichier CSV trouve.")

    mesures_par_x: dict[float, list[float]] = {}

    for path_csv in fichiers_csv:
        mesures = lire_mesures_fichier(path_csv, nom_colonne_x)
        for x, cps in mesures:
            mesures_par_x.setdefault(x, []).append(cps)

    x_vals = sorted(mesures_par_x.keys())
    cps_moy = [mean(mesures_par_x[x]) for x in x_vals]
    cps_std = [pstdev(mesures_par_x[x]) if len(mesures_par_x[x]) > 1 else 0.0 for x in x_vals]
    return x_vals, cps_moy, cps_std


def tracer_une_courbe(
    x: list[float],
    y: list[float],
    yerr: list[float],
    x_label: str,
    titre: str,
    sortie_png: Path,
) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(x, y, yerr=yerr, fmt="o-", linewidth=1.8, markersize=5, capsize=4)
    plt.xlabel(x_label)
    plt.ylabel("Coups par seconde")
    plt.title(titre)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(sortie_png, dpi=180)


def tracer_figure_combinee(
    x_voltage: list[float],
    cps_voltage: list[float],
    cps_voltage_err: list[float],
    x_debit: list[float],
    cps_debit: list[float],
    cps_debit_err: list[float],
    sortie_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].errorbar(
        x_voltage,
        cps_voltage,
        yerr=cps_voltage_err,
        fmt="o-",
        linewidth=1.8,
        markersize=5,
        capsize=4,
    )
    axes[0].set_xlabel("Voltage (V)")
    axes[0].set_ylabel("Coups par seconde")
    axes[0].set_title("Coups/s en fonction du voltage")
    axes[0].grid(True, alpha=0.35)

    axes[1].errorbar(
        x_debit,
        cps_debit,
        yerr=cps_debit_err,
        fmt="o-",
        linewidth=1.8,
        markersize=5,
        capsize=4,
    )
    axes[1].set_xlabel("Debit de gaz")
    axes[1].set_ylabel("Coups par seconde")
    axes[1].set_title("Coups/s en fonction du debit de gaz")
    axes[1].grid(True, alpha=0.35)

    fig.tight_layout()
    fig.savefig(sortie_png, dpi=180)


def main() -> None:
    path_voltage = MESURES_DIR / "voltage"
    path_debit = MESURES_DIR / "debit_gaz"
    SORTIES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        x_voltage, cps_voltage, cps_voltage_err = lire_mesures_dossier(path_voltage, "voltage_V")
        x_debit, cps_debit, cps_debit_err = lire_mesures_dossier(path_debit, "debit_gaz")
    except ValueError as exc:
        print(f"Erreur de donnees: {exc}")
        print("Remplissez les CSV dans le dossier mesures/ puis relancez: python3 tracer_courbes.py")
        return

    tracer_une_courbe(
        x_voltage,
        cps_voltage,
        cps_voltage_err,
        "Voltage (V)",
        "Coups/s en fonction du voltage",
        SORTIES_DIR / "courbe_voltage.png",
    )
    tracer_une_courbe(
        x_debit,
        cps_debit,
        cps_debit_err,
        "Debit de gaz",
        "Coups/s en fonction du debit de gaz",
        SORTIES_DIR / "courbe_debit_gaz.png",
    )
    tracer_figure_combinee(
        x_voltage,
        cps_voltage,
        cps_voltage_err,
        x_debit,
        cps_debit,
        cps_debit_err,
        SORTIES_DIR / "courbes_mesures.png",
    )

    print("Courbes generees :")
    print(f"- {(SORTIES_DIR / 'courbe_voltage.png').as_posix()}")
    print(f"- {(SORTIES_DIR / 'courbe_debit_gaz.png').as_posix()}")
    print(f"- {(SORTIES_DIR / 'courbes_mesures.png').as_posix()}")
    plt.show()


if __name__ == "__main__":
    main()
