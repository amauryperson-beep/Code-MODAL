import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from scipy.optimize import curve_fit
from courbes_et_fits import (
    plot_courbe_ajustee,
    plot_grille_vs_distance,
    plot_grille_vs_tension,
    plot_multi_serie_vs_distance,
    plot_serie_vs_distance,
)

# Exemple imageur beta:
# python3 traitement_imageur_beta.py
# python3 courbes_et_fits.py imageur-i-max-vs-tension --debit 40


@dataclass(frozen=True)
class JeuDonneesGrille:
    """Mesures indexees par distance et tension."""

    nom: str
    distances: np.ndarray
    tensions: np.ndarray
    mesures_brutes: np.ndarray  # shape: (n_distances, n_tensions, n_repetitions)

    def __post_init__(self):
        object.__setattr__(self, "distances", np.asarray(self.distances, dtype=float))
        object.__setattr__(self, "tensions", np.asarray(self.tensions, dtype=float))
        object.__setattr__(self, "mesures_brutes", np.asarray(self.mesures_brutes, dtype=float))

        forme_attendue = (len(self.distances), len(self.tensions))
        if self.mesures_brutes.ndim != 3 or self.mesures_brutes.shape[:2] != forme_attendue:
            raise ValueError(
                f"{self.nom}: mesures_brutes doit avoir la forme "
                f"(n_distances, n_tensions, n_repetitions). "
                f"Attendu {forme_attendue}, obtenu {self.mesures_brutes.shape}."
            )

    @property
    def moyenne(self) -> np.ndarray:
        return np.mean(self.mesures_brutes, axis=2)

    @property
    def ecart_type(self) -> np.ndarray:
        return np.std(self.mesures_brutes, axis=2)


@dataclass(frozen=True)
class JeuDonneesSerie:
    """Mesures indexees uniquement par distance."""

    nom: str
    distances: np.ndarray
    mesures_brutes: np.ndarray  # shape: (n_distances, n_repetitions)

    def __post_init__(self):
        object.__setattr__(self, "distances", np.asarray(self.distances, dtype=float))
        object.__setattr__(self, "mesures_brutes", np.asarray(self.mesures_brutes, dtype=float))

        if self.mesures_brutes.ndim != 2 or self.mesures_brutes.shape[0] != len(self.distances):
            raise ValueError(
                f"{self.nom}: mesures_brutes doit avoir la forme (n_distances, n_repetitions). "
                f"Attendu {len(self.distances)}, obtenu {self.mesures_brutes.shape}."
            )

    @property
    def moyenne(self) -> np.ndarray:
        return np.mean(self.mesures_brutes, axis=1)

    @property
    def ecart_type(self) -> np.ndarray:
        return np.std(self.mesures_brutes, axis=1)


@dataclass(frozen=True)
class JeuDonneesMultiSerie:
    """Mesures indexees par categorie (ex: nombre de plaques) et distance."""

    nom: str
    categories: np.ndarray
    distances: np.ndarray
    mesures_brutes: np.ndarray  # shape: (n_categories, n_distances, n_repetitions)

    def __post_init__(self):
        object.__setattr__(self, "categories", np.asarray(self.categories, dtype=float))
        object.__setattr__(self, "distances", np.asarray(self.distances, dtype=float))
        object.__setattr__(self, "mesures_brutes", np.asarray(self.mesures_brutes, dtype=float))

        forme_attendue = (len(self.categories), len(self.distances))
        if self.mesures_brutes.ndim != 3 or self.mesures_brutes.shape[:2] != forme_attendue:
            raise ValueError(
                f"{self.nom}: mesures_brutes doit avoir la forme "
                f"(n_categories, n_distances, n_repetitions). "
                f"Attendu {forme_attendue}, obtenu {self.mesures_brutes.shape}."
            )

    @property
    def moyenne(self) -> np.ndarray:
        return np.mean(self.mesures_brutes, axis=2)

    @property
    def ecart_type(self) -> np.ndarray:
        return np.std(self.mesures_brutes, axis=2)


@dataclass(frozen=True)
class ResultatFit:
    nom_modele: str
    noms_parametres: tuple[str, ...]
    parametres: np.ndarray
    erreurs: np.ndarray
    chi2: float
    ddl: int

    @property
    def chi2_reduit(self) -> float:
        return self.chi2 / self.ddl if self.ddl > 0 else np.nan


class Traceur:
    @staticmethod
    def grille_vs_distance(
        jeu_donnees: JeuDonneesGrille,
        titre: str,
        log_x: bool = False,
        log_y: bool = False,
    ) -> None:
        plot_grille_vs_distance(
            distances=jeu_donnees.distances,
            tensions=jeu_donnees.tensions,
            moyenne=jeu_donnees.moyenne,
            ecart_type=jeu_donnees.ecart_type,
            titre=titre,
            log_x=log_x,
            log_y=log_y,
        )

    @staticmethod
    def grille_vs_tension(jeu_donnees: JeuDonneesGrille, titre: str, log_y: bool = False) -> None:
        plot_grille_vs_tension(
            distances=jeu_donnees.distances,
            tensions=jeu_donnees.tensions,
            moyenne=jeu_donnees.moyenne,
            ecart_type=jeu_donnees.ecart_type,
            titre=titre,
            log_y=log_y,
        )

    @staticmethod
    def serie_vs_distance(jeu_donnees: JeuDonneesSerie, titre: str) -> None:
        plot_serie_vs_distance(
            distances=jeu_donnees.distances,
            moyenne=jeu_donnees.moyenne,
            ecart_type=jeu_donnees.ecart_type,
            titre=titre,
            label=jeu_donnees.nom,
        )

    @staticmethod
    def multi_serie_vs_distance(jeu_donnees: JeuDonneesMultiSerie, titre: str) -> None:
        plot_multi_serie_vs_distance(
            distances=jeu_donnees.distances,
            categories=jeu_donnees.categories,
            moyenne=jeu_donnees.moyenne,
            ecart_type=jeu_donnees.ecart_type,
            titre=titre,
        )

    @staticmethod
    def courbe_ajustee(
        x: np.ndarray,
        y: np.ndarray,
        y_erreur: np.ndarray,
        x_fit: np.ndarray,
        y_fit: np.ndarray,
        titre: str,
        etiquette: str
        ) -> None:
        plot_courbe_ajustee(
            x=x,
            y=y,
            y_erreur=y_erreur,
            x_fit=x_fit,
            y_fit=y_fit,
            titre=titre,
            etiquette=etiquette,
        )


class ModelesFit:
    @staticmethod
    def inverse_carre(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a / (x + b) ** 2

    @staticmethod
    def inverse_carre_plus_exp(x: np.ndarray, a: float, b: float, c: float, lam: float) -> np.ndarray:
        return (a / (x + b) ** 2) + c * np.exp(-lam * x)

    @staticmethod
    def inverse_carre_avec_ecran(x: np.ndarray, a: float, b: float, mu: float, epaisseur: float, nb_plaques: int) -> np.ndarray:
        return (a / (x + b) ** 2) * np.exp(-mu * epaisseur * nb_plaques)

    @staticmethod
    def exponentielle_plus_constante(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * np.exp(-b * x) + c


class Analyseur:
    def __init__(self) -> None:
        (
            self.geiger_ancien,
            self.geiger_nouveau,
            self.attenuation_plomb,
            self.attenuation_cs,
        ) = charger_jeux_donnees()

    @staticmethod
    def _sigma_securise(y_erreur: np.ndarray) -> np.ndarray:
        y_erreur = np.asarray(y_erreur, dtype=float)
        return np.where(y_erreur <= 0, 1e-9, y_erreur)

    @staticmethod
    def _executer_actions_nommees(actions: dict[str, callable], selection: list[str] | None, type_action: str) -> None:
        if selection is None:
            cles = list(actions.keys())
        else:
            jetons = [element.strip() for element in selection if element.strip()]
            if "all" in jetons or "tout" in jetons:
                cles = list(actions.keys())
            else:
                cles = [jeton for jeton in jetons if jeton not in {"none", "aucun"}]
                if not cles:
                    return

        invalides = [cle for cle in cles if cle not in actions]
        if invalides:
            choix_valides = ", ".join(actions.keys())
            raise ValueError(f"{type_action}s inconnus: {invalides}. Choix valides: {choix_valides}")

        for cle in cles:
            actions[cle]()

    def tracer_geiger_1_distance(self) -> None:
        Traceur.grille_vs_distance(
            self.geiger_ancien,
            "Geiger 1 - Distance (series en tension)",
        )

    def tracer_geiger_1_tension(self) -> None:
        Traceur.grille_vs_tension(
            self.geiger_ancien,
            "Geiger 1 - Tension (series en distance)",
        )

    def tracer_geiger_2_distance(self) -> None:
        Traceur.grille_vs_distance(
            self.geiger_nouveau,
            "Geiger 2 - Distance (series en tension)",
        )

    def tracer_geiger_2_tension(self) -> None:
        Traceur.grille_vs_tension(
            self.geiger_nouveau,
            "Geiger 2 - Tension (series en distance)",
        )

    def tracer_attenuation_plomb(self) -> None:
        Traceur.serie_vs_distance(
            self.attenuation_plomb,
            "Attenuation plomb - Distance",
        )

    def tracer_geiger_2_distance_avec_plomb(self) -> None:
        Traceur.multi_serie_vs_distance(
            self.attenuation_cs,
            "Geiger 2 - Distance avec plaques de plomb",
        )

    def actions_graphes(self) -> dict[str, callable]:
        return {
            "geiger_1_distance": self.tracer_geiger_1_distance,
            "geiger_1_tension": self.tracer_geiger_1_tension,
            "geiger_2_distance": self.tracer_geiger_2_distance,
            "geiger_2_tension": self.tracer_geiger_2_tension,
            "attenuation_plomb": self.tracer_attenuation_plomb,
            "geiger_2_distance_avec_plomb": self.tracer_geiger_2_distance_avec_plomb,
        }

    def executer_graphes(self, selection: list[str] | None = None) -> None:
        self._executer_actions_nommees(self.actions_graphes(), selection, "graphe")

    def fit_generique(self, nom_modele: str, modele, x: np.ndarray, y: np.ndarray, y_erreur: np.ndarray, p0: list[float], noms_parametres: tuple[str, ...], titre: str,) -> ResultatFit:
        sigma = self._sigma_securise(y_erreur)
        parametres, covariance = curve_fit(modele, x, y, sigma=sigma, absolute_sigma=True, p0=p0)
        erreurs = np.sqrt(np.diag(covariance))

        y_points_fit = modele(x, *parametres)
        chi2 = float(np.sum(((y - y_points_fit) / sigma) ** 2))
        ddl = int(len(y) - len(parametres))

        resultat = ResultatFit(
            nom_modele=nom_modele,
            noms_parametres=noms_parametres,
            parametres=parametres,
            erreurs=erreurs,
            chi2=chi2,
            ddl=ddl,
        )

        self._afficher_fit(resultat)

        x_fit = np.linspace(np.min(x), np.max(x), 300)
        y_fit = modele(x_fit, *parametres)
        etiquette_parametres = ", ".join(
            f"{nom}={valeur:.3f}" for nom, valeur in zip(noms_parametres, parametres)
        )
        Traceur.courbe_ajustee(
            x,
            y,
            sigma,
            x_fit,
            y_fit,
            titre,
            f"Fit: {etiquette_parametres}, chi2/ddl={resultat.chi2_reduit:.3f}",
        )

        return resultat

    @staticmethod
    def _afficher_fit(resultat: ResultatFit) -> None:
        print(f"\nModèle: {resultat.nom_modele}")
        for nom, valeur, erreur in zip(resultat.noms_parametres, resultat.parametres, resultat.erreurs):
            print(f"{nom} = {valeur:.4f} ± {erreur:.4f}")
        print(f"chi2 = {resultat.chi2:.4f}")
        print(f"ddl = {resultat.ddl}")
        print(f"chi2/ddl = {resultat.chi2_reduit:.4f}")

    def fit_geiger_1_distance_modele_1(self) -> ResultatFit:
        return self.fit_generique(
            nom_modele="1/(x+b)^2 (geiger ancien, index tension 2)",
            modele=ModelesFit.inverse_carre,
            x=self.geiger_ancien.distances,
            y=self.geiger_ancien.moyenne[:, 2],
            y_erreur=self.geiger_ancien.ecart_type[:, 2],
            p0=[0.1, 0.02],
            noms_parametres=("a", "b"),
            titre="Geiger 1 - Distance (a/(x+b)^2)",
        )

    def fit_geiger_2_distance_modele_1(self) -> ResultatFit:
        return self.fit_generique(
            nom_modele="1/(x+b)^2 (geiger nouveau, index tension 2)",
            modele=ModelesFit.inverse_carre,
            x=self.geiger_nouveau.distances,
            y=self.geiger_nouveau.moyenne[:, 2],
            y_erreur=self.geiger_nouveau.ecart_type[:, 2],
            p0=[0.3, 0.03],
            noms_parametres=("a", "b"),
            titre="Geiger 2 - Distance (a/(x+b)^2)",
        )

    def fit_geiger_2_distance_modele_2(self) -> ResultatFit:
        return self.fit_generique(
            nom_modele="1/(x+b)^2 + c*exp(-lambda*x) (geiger nouveau, index tension 2)",
            modele=ModelesFit.inverse_carre_plus_exp,
            x=self.geiger_nouveau.distances,
            y=self.geiger_nouveau.moyenne[:, 2],
            y_erreur=self.geiger_nouveau.ecart_type[:, 2],
            p0=[0.3, 0.03, -13.0, 50.0],
            noms_parametres=("a", "b", "c", "lambda"),
            titre="Geiger 2 - Distance (a/(x+b)^2 + c*exp(-lambda*x))",
        )

    def fit_attenuation_plomb_modele_1(
        self,
        nb_plaques: int = 1,
        epaisseur: float = 3e-3,
    ) -> ResultatFit:
        return self.fit_generique(
            nom_modele="1/(x+b)^2 * exp(-mu*e*n) (attenuation)",
            modele=lambda x, a, b, mu: ModelesFit.inverse_carre_avec_ecran(x, a, b, mu, epaisseur, nb_plaques),
            x=self.attenuation_plomb.distances,
            y=self.attenuation_plomb.moyenne,
            y_erreur=self.attenuation_plomb.ecart_type,
            p0=[0.3, 0.03, 1.0],
            noms_parametres=("a", "b", "mu"),
            titre="Attenuation plomb - Distance ((a/(x+b)^2)*exp(-mu*e*n))",
        )

    def fit_attenuation_plomb_modele_2(self) -> ResultatFit:
        y = self.attenuation_plomb.moyenne
        c0 = float(np.min(y))
        a0 = float(np.max(y) - c0)
        b0 = 15.0
        return self.fit_generique(
            nom_modele="a*exp(-b*x)+c (attenuation plomb)",
            modele=ModelesFit.exponentielle_plus_constante,
            x=self.attenuation_plomb.distances,
            y=y,
            y_erreur=self.attenuation_plomb.ecart_type,
            p0=[a0, b0, c0],
            noms_parametres=("a", "b", "c"),
            titre="Attenuation plomb - Distance (a*exp(-b*d)+c)",
        )

    def actions_fits(self) -> dict[str, callable]:
        return {
            "geiger_1_distance_modele_1": self.fit_geiger_1_distance_modele_1,
            "geiger_2_distance_modele_1": self.fit_geiger_2_distance_modele_1,
            "geiger_2_distance_modele_2": self.fit_geiger_2_distance_modele_2,
            "attenuation_plomb_modele_1": self.fit_attenuation_plomb_modele_1,
            "attenuation_plomb_modele_2": self.fit_attenuation_plomb_modele_2,
        }

    def executer_fits(self, selection: list[str] | None = None) -> None:
        self._executer_actions_nommees(self.actions_fits(), selection, "fit")


def _decouper_arguments_csv(valeurs: list[str] | None) -> list[str]:
    if not valeurs:
        return []
    jetons: list[str] = []
    for valeur in valeurs:
        for jeton in valeur.split(","):
            propre = jeton.strip()
            if propre:
                jetons.append(propre)
    return jetons


def _construire_parseur_arguments() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(
        description="Analyse Geiger: affichage des graphes et ajustements.",
    )
    parseur.add_argument(
        "--list",
        action="store_true",
        help="Liste les graphes/fits disponibles puis quitte.",
    )
    parseur.add_argument(
        "--plots",
        "--graphes",
        dest="graphes",
        nargs="+",
        default=["none"],
        help="Graphes a afficher (espace ou virgule). Utiliser all/tout ou none/aucun.",
    )
    parseur.add_argument(
        "--graphe",
        action="append",
        default=[],
        help="Raccourci pour afficher un graphe precis (option repetable).",
    )
    parseur.add_argument(
        "--fits",
        nargs="+",
        default=["none"],
        help="Fits a lancer (espace ou virgule). Utiliser all/tout ou none/aucun.",
    )
    parseur.add_argument(
        "--fit",
        action="append",
        default=[],
        help="Raccourci pour lancer un fit precis (option repetable).",
    )
    return parseur


def _afficher_options_disponibles(analyseur: Analyseur) -> None:
    print("Graphes disponibles:")
    for nom in analyseur.actions_graphes().keys():
        print(f" - {nom}")
    print("\nFits disponibles:")
    for nom in analyseur.actions_fits().keys():
        print(f" - {nom}")
    print("\nRaccourcis: all/tout, none/aucun")
    print("\nExemples:")
    print(" - python3 main.py --graphe geiger_1_distance")
    print(" - python3 main.py --fit geiger_2_distance_modele_1")


def _charger_grille_depuis_csv(path_csv: Path, nom: str) -> JeuDonneesGrille:
    with path_csv.open(newline="", encoding="utf-8") as fichier:
        lignes = list(csv.DictReader(fichier))

    distances = np.array(sorted({float(ligne["distance_m"]) for ligne in lignes}), dtype=float)
    tensions = np.array(sorted({float(ligne["tension_V"]) for ligne in lignes}), dtype=float)
    repetitions = np.array(sorted({int(ligne["repetition"]) for ligne in lignes}), dtype=int)

    mesures = np.full((len(distances), len(tensions), len(repetitions)), np.nan, dtype=float)
    index_distance = {valeur: i for i, valeur in enumerate(distances)}
    index_tension = {valeur: i for i, valeur in enumerate(tensions)}
    index_repetition = {valeur: i for i, valeur in enumerate(repetitions)}

    for ligne in lignes:
        d = index_distance[float(ligne["distance_m"])]
        t = index_tension[float(ligne["tension_V"])]
        r = index_repetition[int(ligne["repetition"])]
        if not np.isnan(mesures[d, t, r]):
            raise ValueError(f"{path_csv.name}: doublon detecte pour distance/tension/repetition.")
        mesures[d, t, r] = float(ligne["coups_par_seconde"])

    if np.isnan(mesures).any():
        raise ValueError(f"{path_csv.name}: valeurs manquantes dans la grille de mesures.")

    return JeuDonneesGrille(
        nom=nom,
        distances=distances,
        tensions=tensions,
        mesures_brutes=mesures,
    )


def _charger_serie_depuis_csv(path_csv: Path, nom: str) -> JeuDonneesSerie:
    with path_csv.open(newline="", encoding="utf-8") as fichier:
        lignes = list(csv.DictReader(fichier))

    distances = np.array(sorted({float(ligne["distance_m"]) for ligne in lignes}), dtype=float)
    repetitions = np.array(sorted({int(ligne["repetition"]) for ligne in lignes}), dtype=int)

    mesures = np.full((len(distances), len(repetitions)), np.nan, dtype=float)
    index_distance = {valeur: i for i, valeur in enumerate(distances)}
    index_repetition = {valeur: i for i, valeur in enumerate(repetitions)}

    for ligne in lignes:
        d = index_distance[float(ligne["distance_m"])]
        r = index_repetition[int(ligne["repetition"])]
        if not np.isnan(mesures[d, r]):
            raise ValueError(f"{path_csv.name}: doublon detecte pour distance/repetition.")
        mesures[d, r] = float(ligne["coups_par_seconde"])

    if np.isnan(mesures).any():
        raise ValueError(f"{path_csv.name}: valeurs manquantes dans la serie de mesures.")

    return JeuDonneesSerie(
        nom=nom,
        distances=distances,
        mesures_brutes=mesures,
    )


def _charger_multi_serie_depuis_csv(path_csv: Path, nom: str) -> JeuDonneesMultiSerie:
    with path_csv.open(newline="", encoding="utf-8") as fichier:
        lignes = list(csv.DictReader(fichier))

    categories = np.array(sorted({float(ligne["nb_plaques"]) for ligne in lignes}), dtype=float)
    distances = np.array(sorted({float(ligne["distance_m"]) for ligne in lignes}), dtype=float)
    repetitions = np.array(sorted({int(ligne["repetition"]) for ligne in lignes}), dtype=int)

    mesures = np.full((len(categories), len(distances), len(repetitions)), np.nan, dtype=float)
    index_categorie = {valeur: i for i, valeur in enumerate(categories)}
    index_distance = {valeur: i for i, valeur in enumerate(distances)}
    index_repetition = {valeur: i for i, valeur in enumerate(repetitions)}

    for ligne in lignes:
        c = index_categorie[float(ligne["nb_plaques"])]
        d = index_distance[float(ligne["distance_m"])]
        r = index_repetition[int(ligne["repetition"])]
        if not np.isnan(mesures[c, d, r]):
            raise ValueError(f"{path_csv.name}: doublon detecte pour categorie/distance/repetition.")
        mesures[c, d, r] = float(ligne["coups_par_seconde"])

    if np.isnan(mesures).any():
        raise ValueError(f"{path_csv.name}: valeurs manquantes dans la multi-serie de mesures.")

    return JeuDonneesMultiSerie(
        nom=nom,
        categories=categories,
        distances=distances,
        mesures_brutes=mesures,
    )


def charger_jeux_donnees() -> tuple[JeuDonneesGrille, JeuDonneesGrille, JeuDonneesSerie, JeuDonneesMultiSerie]:
    base_dir = Path(__file__).resolve().parent / "Mesures" / "Geiger"
    jeu_ancien = _charger_grille_depuis_csv(base_dir / "geiger_1_distance.csv", "Geiger 1")
    jeu_nouveau = _charger_grille_depuis_csv(base_dir / "geiger_2_distance.csv", "Geiger 2")
    jeu_attenuation = _charger_serie_depuis_csv(base_dir / "geiger_1_plomb.csv", "Attenuation plomb")
    jeu_attenuation_cs = _charger_multi_serie_depuis_csv(base_dir / "geiger_2_plomb.csv", "Geiger 2 plomb")
    return jeu_ancien, jeu_nouveau, jeu_attenuation, jeu_attenuation_cs


def main() -> None:
    parseur = _construire_parseur_arguments()
    arguments = parseur.parse_args()

    plt.close("all")
    analyseur = Analyseur()
    if arguments.list:
        _afficher_options_disponibles(analyseur)
        return

    graphes_selectionnes = _decouper_arguments_csv(arguments.graphes) + arguments.graphe
    fits_selectionnes = _decouper_arguments_csv(arguments.fits) + arguments.fit

    analyseur.executer_graphes(graphes_selectionnes)
    analyseur.executer_fits(fits_selectionnes)


def afficher_graphe(nom_graphe: str) -> None:
    """Affiche un seul graphe par son identifiant."""
    plt.close("all")
    analyseur = Analyseur()
    analyseur.executer_graphes([nom_graphe])


def lancer_fit(nom_fit: str) -> None:
    """Lance un seul fit par son identifiant."""
    plt.close("all")
    analyseur = Analyseur()
    analyseur.executer_fits([nom_fit])


if __name__ == "__main__":
    main()
