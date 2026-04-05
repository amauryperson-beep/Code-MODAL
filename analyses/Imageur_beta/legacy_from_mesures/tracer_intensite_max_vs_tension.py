from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "mesures_5XX0-40_resume.csv"
OUTPUT_PNG = BASE_DIR / "intensite_max_vs_tension.png"


def load_points(path_csv: Path) -> tuple[list[float], list[float]]:
    tensions: list[float] = []
    intensites: list[float] = []
    with path_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tensions.append(float(row["tension"]))
            intensites.append(float(row["max_data_gray"]))
    pairs = sorted(zip(tensions, intensites), key=lambda t: t[0])
    x, y = zip(*pairs)
    return list(x), list(y)


def main() -> None:
    x, y = load_points(INPUT_CSV)
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(x, y, "o-", linewidth=1.8, markersize=5, label="Intensite maximale (donnees)")
    plt.xlabel("Tension")
    plt.ylabel("Intensite maximale (Gray value)")
    plt.title("Intensite maximale en fonction de la tension")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=180)
    print(f"Courbe enregistree: {OUTPUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
