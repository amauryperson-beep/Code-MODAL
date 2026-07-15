
"""
physics.py — Modèles physiques de la simulation bêta
=====================================================
Constantes et fonctions physiques utilisées par le moteur de simulation.

Modèles implémentés :
  - Multiplication de Townsend (1er coefficient) stochastique
  - Diffusion transverse et longitudinale (Einstein : D = μ·kT/e)
  - Vitesse de dérive linéaire simplifiée
  - Émission radioactive (loi de Poisson dans le temps)
"""

import numpy as np

# ─── Constantes physiques ────────────────────────────────────────────────────
E_CHARGE   = 1.602e-19   # C
K_BOLTZMANN= 1.381e-23   # J/K
T_GAS      = 293.0       # K  (température ambiante)
M_ELECTRON = 9.109e-31   # kg

# ─── Paramètres du gaz (Ar/CO2 typique) ─────────────────────────────────────
MOBILITY   = 0.1         # m²/(V·s)  — mobilité réduite (valeur illustrative)
DT_BASE    = 1e-9        # s  — pas de temps de base

# ─── Fonctions physiques vectorisées ─────────────────────────────────────────

def drift_velocity(E_field: float) -> float:
    """
    Vitesse de dérive des électrons dans le gaz.
    Modèle linéaire simplifié : v_d = μ · E
    Retourne la vitesse en m/s (vers le bas, donc signée négativement en Y).
    """
    return MOBILITY * abs(E_field)


def diffusion_coefficients(E_field: float) -> tuple[float, float]:
    """
    Coefficients de diffusion transverse (DT) et longitudinale (DL).
    Relation d'Einstein simplifiée : D = μ · kT/e
    Un facteur empirique ETA sépare DT et DL (anisotropie faible).
    """
    D_base = MOBILITY * K_BOLTZMANN * T_GAS / E_CHARGE
    ETA    = 0.6   # rapport DL/DT
    DT     = D_base
    DL     = D_base * ETA
    return DT, DL


def townsend_gain_vectorized(n_electrons: int, alpha: float, dx: float,
                              rng: np.random.Generator) -> np.ndarray:
    """
    Multiplication stochastique de Townsend vectorisée.
    Pour chaque électron entrant, le nombre de secondaires produits suit
    une loi de Poisson de paramètre λ = alpha · dx.
    Retourne un tableau d'entiers (nombre de secondaires par électron).
    Référence : W. Blum, W. Riegler, L. Rolandi, "Particle Detection with Drift Chambers"
    """
    lam = alpha * abs(dx)
    lam = np.clip(lam, 0.0, 20.0)   # sécurité numérique
    return rng.poisson(lam, size=n_electrons)


def random_emission_times(n_total: int, rate: float,
                           t_max: float, rng: np.random.Generator) -> np.ndarray:
    """
    Génère des temps d'émission radioactive selon un processus de Poisson.
    rate  : taux d'émission en électrons/seconde
    t_max : durée totale de la simulation
    Retourne un tableau trié de temps d'émission.
    """
    n_expected = int(rate * t_max)
    if n_expected <= 0:
        return np.array([], dtype=float)
    times = rng.uniform(0.0, t_max, size=min(n_total, n_expected))
    return np.sort(times)


def apply_diffusion_vectorized(positions: np.ndarray, velocities: np.ndarray,
                                dt: float, E_field: float,
                                rng: np.random.Generator) -> np.ndarray:
    """
    Ajoute la diffusion gaussienne à un lot d'électrons.
    positions  : (N, 2) tableau [x, y]
    velocities : (N, 2) tableau [vx, vy]  (modifié in-place)
    Retourne les positions mises à jour.
    """
    DT, DL = diffusion_coefficients(E_field)
    n = positions.shape[0]
    if n == 0:
        return positions

    # Sigma de diffusion sur ce pas de temps
    sigma_T = np.sqrt(2.0 * DT * dt)
    sigma_L = np.sqrt(2.0 * DL * dt)

    positions[:, 0] += rng.normal(0.0, sigma_T, n)   # diffusion transverse (X)
    positions[:, 1] += rng.normal(0.0, sigma_L, n)   # diffusion longitudinale (Y)
    return positions