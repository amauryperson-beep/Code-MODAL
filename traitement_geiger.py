from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


RAW_FILES = {
    "geiger_1_grille": "geiger_1_grille.csv",
    "geiger_2_grille": "geiger_2_grille.csv",
    "geiger_1_plomb": "geiger_1_plomb.csv",
    "geiger_2_plomb": "geiger_2_plomb.csv",
}

OUT_FILES = {
    "geiger_1_distance": "geiger_1_distance.csv",
    "geiger_1_tension": "geiger_1_tension.csv",
    "geiger_2_distance": "geiger_2_distance.csv",
    "geiger_2_tension": "geiger_2_tension.csv",
    "geiger_1_plomb": "geiger_1_plomb.csv",
    "geiger_2_plomb": "geiger_2_plomb.csv",
}


def _read_csv(path_csv: Path) -> list[dict[str, str]]:
    with path_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], col: str, path_csv: Path) -> float:
    value = (row.get(col) or "").strip()
    if not value:
        raise ValueError(f"{path_csv.name}: colonne '{col}' vide.")
    return float(value)


def _int_opt(row: dict[str, str], col: str) -> int | None:
    if col not in row:
        return None
    value = (row.get(col) or "").strip()
    if not value:
        return None
    return int(value)


def _normaliser_repetitions(
    grouped: dict[tuple[float, ...], list[tuple[int | None, int, float]]]
) -> dict[tuple[float, ...], list[tuple[int, float]]]:
    out: dict[tuple[float, ...], list[tuple[int, float]]] = {}
    for key, items in grouped.items():
        reps = [rep for rep, _, _ in items if rep is not None]
        all_have_rep = len(reps) == len(items)
        if all_have_rep and len(set(reps)) == len(reps):
            ordered = sorted(items, key=lambda t: int(t[0]))  # type: ignore[arg-type]
            out[key] = [(int(rep), cps) for rep, _, cps in ordered if rep is not None]
        else:
            ordered = sorted(items, key=lambda t: t[1])
            out[key] = [(i + 1, cps) for i, (_, _, cps) in enumerate(ordered)]
    return out


def _write_rows(path_csv: Path, fieldnames: list[str], rows: list[dict[str, float | int]]) -> None:
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    with path_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _traiter_grille(raw_csv: Path, out_distance_csv: Path, out_tension_csv: Path) -> None:
    lignes = _read_csv(raw_csv)
    grouped: dict[tuple[float, float], list[tuple[int | None, int, float]]] = defaultdict(list)
    for idx, ligne in enumerate(lignes):
        d = _float(ligne, "distance_m", raw_csv)
        u = _float(ligne, "tension_V", raw_csv)
        cps = _float(ligne, "coups_par_seconde", raw_csv)
        rep = _int_opt(ligne, "repetition")
        grouped[(d, u)].append((rep, idx, cps))

    norm = _normaliser_repetitions(grouped)
    rows: list[dict[str, float | int]] = []
    for (d, u), reps in sorted(norm.items(), key=lambda t: (t[0][0], t[0][1])):
        for rep, cps in reps:
            rows.append(
                {
                    "distance_m": d,
                    "tension_V": u,
                    "repetition": rep,
                    "coups_par_seconde": cps,
                }
            )

    fields = ["distance_m", "tension_V", "repetition", "coups_par_seconde"]
    _write_rows(out_distance_csv, fields, rows)
    _write_rows(out_tension_csv, fields, rows)


def _traiter_serie(raw_csv: Path, out_csv: Path) -> None:
    lignes = _read_csv(raw_csv)
    grouped: dict[tuple[float], list[tuple[int | None, int, float]]] = defaultdict(list)
    for idx, ligne in enumerate(lignes):
        d = _float(ligne, "distance_m", raw_csv)
        cps = _float(ligne, "coups_par_seconde", raw_csv)
        rep = _int_opt(ligne, "repetition")
        grouped[(d,)].append((rep, idx, cps))

    norm = _normaliser_repetitions(grouped)
    rows: list[dict[str, float | int]] = []
    for (d,), reps in sorted(norm.items(), key=lambda t: t[0][0]):
        for rep, cps in reps:
            rows.append(
                {
                    "distance_m": d,
                    "repetition": rep,
                    "coups_par_seconde": cps,
                }
            )
    _write_rows(out_csv, ["distance_m", "repetition", "coups_par_seconde"], rows)


def _traiter_multi_serie(raw_csv: Path, out_csv: Path) -> None:
    lignes = _read_csv(raw_csv)
    grouped: dict[tuple[float, float], list[tuple[int | None, int, float]]] = defaultdict(list)
    for idx, ligne in enumerate(lignes):
        p = _float(ligne, "nb_plaques", raw_csv)
        d = _float(ligne, "distance_m", raw_csv)
        cps = _float(ligne, "coups_par_seconde", raw_csv)
        rep = _int_opt(ligne, "repetition")
        grouped[(p, d)].append((rep, idx, cps))

    norm = _normaliser_repetitions(grouped)
    rows: list[dict[str, float | int]] = []
    for (p, d), reps in sorted(norm.items(), key=lambda t: (t[0][0], t[0][1])):
        for rep, cps in reps:
            rows.append(
                {
                    "nb_plaques": int(p) if float(p).is_integer() else p,
                    "distance_m": d,
                    "repetition": rep,
                    "coups_par_seconde": cps,
                }
            )
    _write_rows(out_csv, ["nb_plaques", "distance_m", "repetition", "coups_par_seconde"], rows)


def _bootstrap_raw_if_missing(raw_dir: Path, out_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    mapping = [
        ("geiger_1_grille.csv", "geiger_1_distance.csv"),
        ("geiger_2_grille.csv", "geiger_2_distance.csv"),
        ("geiger_1_plomb.csv", "geiger_1_plomb.csv"),
        ("geiger_2_plomb.csv", "geiger_2_plomb.csv"),
    ]
    for raw_name, out_name in mapping:
        raw_path = raw_dir / raw_name
        out_path = out_dir / out_name
        if raw_path.exists():
            continue
        if out_path.exists():
            shutil.copy2(out_path, raw_path)
            print(f"Bootstrap raw: {raw_path} (copie de {out_path.name})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline Geiger: lit les CSV de Mesures/Geiger/Données brutes et regenere "
            "les CSV par experience dans Mesures/Geiger."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("Mesures/Geiger/Données brutes"))
    parser.add_argument("--out-dir", type=Path, default=Path("Mesures/Geiger"))
    args = parser.parse_args()

    raw_dir = args.raw_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _bootstrap_raw_if_missing(raw_dir, out_dir)

    raw_geiger_1 = raw_dir / RAW_FILES["geiger_1_grille"]
    raw_geiger_2 = raw_dir / RAW_FILES["geiger_2_grille"]
    raw_plomb_1 = raw_dir / RAW_FILES["geiger_1_plomb"]
    raw_plomb_2 = raw_dir / RAW_FILES["geiger_2_plomb"]

    missing = [p for p in [raw_geiger_1, raw_geiger_2, raw_plomb_1, raw_plomb_2] if not p.exists()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise SystemExit(f"Fichiers bruts manquants: {names}")

    _traiter_grille(raw_geiger_1, out_dir / OUT_FILES["geiger_1_distance"], out_dir / OUT_FILES["geiger_1_tension"])
    _traiter_grille(raw_geiger_2, out_dir / OUT_FILES["geiger_2_distance"], out_dir / OUT_FILES["geiger_2_tension"])
    _traiter_serie(raw_plomb_1, out_dir / OUT_FILES["geiger_1_plomb"])
    _traiter_multi_serie(raw_plomb_2, out_dir / OUT_FILES["geiger_2_plomb"])

    print("Pipeline Geiger termine.")
    for key in [
        "geiger_1_distance",
        "geiger_1_tension",
        "geiger_2_distance",
        "geiger_2_tension",
        "geiger_1_plomb",
        "geiger_2_plomb",
    ]:
        print(f"- {out_dir / OUT_FILES[key]}")


if __name__ == "__main__":
    main()
