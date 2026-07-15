"""
regions.py — Zones physiques du détecteur
==========================================
Chaque région définit :
  - ses limites Y (y_top, y_bottom)
  - son action sur les particules (vectorisée)

Architecture :
  Source          → émission radioactive
  AvalancheRegion → accélération + multiplication de Townsend
  DriftRegion     → dérive pure + diffusion
"""

import numpy as np
from physics import (drift_velocity, diffusion_coefficients,
                     townsend_gain_vectorized, apply_diffusion_vectorized,
                     E_CHARGE, M_ELECTRON)
from particles import ParticleArray, IX, IY, IVX, IVY, IALIVE, IAGE, IWEIGHT


# ─────────────────────────────────────────────────────────────────────────────
#  Source radioactive
# ─────────────────────────────────────────────────────────────────────────────

class Source:
    """
    Source ponctuelle émettant des électrons bêta.
    L'émission est un processus de Poisson : les temps d'émission
    sont tirés au hasard, puis les électrons sont injectés quand
    t_sim dépasse leur temps d'émission.
    """

    def __init__(self, x_pos: float, y_pos: float,
                 rate: float = 500.0,
                 source_id: int = 0,
                 v_spread: float = 2e4):
        """
        x_pos, y_pos : position dans le détecteur (m)
        rate         : taux d'émission (électrons/s)
        v_spread     : dispersion de vitesse initiale (m/s)
        """
        self.x_pos     = x_pos
        self.y_pos     = y_pos
        self.rate      = rate
        self.source_id = source_id
        self.v_spread  = v_spread

        self._pending_times: np.ndarray = np.array([])
        self._ptr = 0   # index du prochain électron à émettre

    def prebuild_schedule(self, t_max: float, rng: np.random.Generator) -> None:
        """
        Pré-calcule tous les temps d'émission pour [0, t_max].
        Appelé une seule fois au reset de la simulation.
        """
        n_expected = max(1, int(self.rate * t_max * 3))   # marge x3
        inter_times = rng.exponential(1.0 / self.rate, size=n_expected)
        self._pending_times = np.cumsum(inter_times)
        # Tronquer à t_max
        self._pending_times = self._pending_times[self._pending_times <= t_max]
        self._ptr = 0

    def emit(self, t_now: float, particles: ParticleArray,
             rng: np.random.Generator) -> int:
        """
        Émet les électrons dont le temps d'émission est ≤ t_now.
        Retourne le nombre d'électrons émis.
        """
        if self._ptr >= len(self._pending_times):
            return 0

        # Trouver les indices des électrons à émettre
        mask = self._pending_times[self._ptr:] <= t_now
        n    = int(mask.sum())
        if n == 0:
            return 0

        # Direction aléatoire (hémisphère vers le bas, axe Y)
        angles = rng.uniform(-np.pi / 3, np.pi / 3, n)   # ±60° autour de Y
        v0     = rng.normal(self.v_spread, self.v_spread * 0.2, n).astype(np.float32)
        vx_em  = (v0 * np.sin(angles)).astype(np.float32)
        vy_em  = (-np.abs(v0 * np.cos(angles))).astype(np.float32)  # vers le bas

        xs = np.full(n, self.x_pos, dtype=np.float32)
        ys = np.full(n, self.y_pos, dtype=np.float32)

        particles.add(xs, ys, vx_em, vy_em, source_id=self.source_id)
        self._ptr += n
        return n

    def reset(self) -> None:
        self._ptr = 0


# ─────────────────────────────────────────────────────────────────────────────
#  Zone d'avalanche
# ─────────────────────────────────────────────────────────────────────────────

class AvalancheRegion:
    """
    Zone de multiplication de Townsend.
    Accélère les électrons vers le bas et crée des secondaires.

    alpha (1er coefficient de Townsend) est modélisé par :
        alpha = A * p * exp(-B * p / E)
    avec p = pression en bar (normalisée), A et B = constantes du gaz.
    Pour simplicité, on utilise un alpha effectif configurable.
    """

    def __init__(self, y_top: float, y_bottom: float,
                 E_field: float = 5e4,      # V/m
                 alpha_eff: float = 3e3,    # m⁻¹  (coefficient de Townsend effectif)
                 label: str = "Avalanche"):
        self.y_top    = y_top
        self.y_bottom = y_bottom
        self.E_field  = E_field       # positif → force vers le bas (−Y)
        self.alpha    = alpha_eff
        self.label    = label
        self.thickness = abs(y_bottom - y_top)

    def contains(self, y: np.ndarray) -> np.ndarray:
        """Masque booléen : électrons dans cette zone."""
        return (y <= self.y_top) & (y >= self.y_bottom)

    def step(self, particles: ParticleArray, dt: float,
             rng: np.random.Generator) -> None:
        """
        Pas de temps pour les électrons dans cette zone :
          1. Accélération sous E_field
          2. Déplacement
          3. Multiplication stochastique de Townsend
          4. Diffusion
        """
        mask = self.contains(particles.y) & (particles.alive > 0.5)
        if not mask.any():
            return

        idx = np.where(mask)[0]
        n   = len(idx)

        # 1. Accélération : F = eE, a = eE/m
        ax = np.float32(E_CHARGE * self.E_field / M_ELECTRON)  # m/s²
        particles._buf[idx, IVY] -= ax * dt     # vers le bas = −Y

        # 2. Déplacement
        particles._buf[idx, IX] += particles._buf[idx, IVX] * dt
        particles._buf[idx, IY] += particles._buf[idx, IVY] * dt

        # 3. Diffusion
        pos = particles._buf[idx][:, [IX, IY]].copy()
        pos = apply_diffusion_vectorized(pos, None, dt, self.E_field, rng)
        particles._buf[idx, IX] = pos[:, 0]
        particles._buf[idx, IY] = pos[:, 1]

        # 4. Multiplication de Townsend
        dy       = abs(particles._buf[idx, IVY] * dt)  # distance parcourue
        sec_counts = townsend_gain_vectorized(n, self.alpha, dy.mean(), rng)
        sec_counts = np.clip(sec_counts, 0, 10)   # limite l'explosion

        # Ajouter les secondaires
        parent_x = particles._buf[idx, IX]
        parent_y = particles._buf[idx, IY]
        vy_base  = float(particles._buf[idx, IVY].mean())
        particles.add_secondary(parent_x, parent_y, vy_base, sec_counts, rng)

        # 5. Marquer comme morts les électrons qui ont quitté la zone par le bas
        out_mask = particles._buf[idx, IY] < self.y_bottom
        particles._buf[idx[out_mask], IALIVE] = 0.0

    @property
    def gain_estimate(self) -> float:
        """Gain de Townsend théorique : G = exp(alpha * d)"""
        return np.exp(self.alpha * self.thickness)


# ─────────────────────────────────────────────────────────────────────────────
#  Zone de dérive
# ─────────────────────────────────────────────────────────────────────────────

class DriftRegion:
    """
    Zone de dérive : champ uniforme, pas de multiplication.
    Les électrons dérivent à vitesse constante et diffusent.
    C'est ici que l'élargissement spatial du nuage est le plus visible.
    """

    def __init__(self, y_top: float, y_bottom: float,
                 E_field: float = 1e3,    # V/m
                 label: str = "Dérive"):
        self.y_top    = y_top
        self.y_bottom = y_bottom
        self.E_field  = E_field
        self.label    = label
        self._v_drift = drift_velocity(E_field)  # m/s

    def contains(self, y: np.ndarray) -> np.ndarray:
        return (y <= self.y_top) & (y >= self.y_bottom)

    def step(self, particles: ParticleArray, dt: float,
             rng: np.random.Generator) -> None:
        """
        Dérive uniforme + diffusion.
        Pas d'accélération : vitesse de dérive imposée (régime de saturation).
        """
        mask = self.contains(particles.y) & (particles.alive > 0.5)
        if not mask.any():
            return

        idx = np.where(mask)[0]

        # Imposer la vitesse de dérive
        particles._buf[idx, IVY] = np.float32(-self._v_drift)

        # Déplacement
        particles._buf[idx, IX] += particles._buf[idx, IVX] * dt
        particles._buf[idx, IY] += particles._buf[idx, IVY] * dt

        # Diffusion (principale source d'élargissement)
        pos = particles._buf[idx][:, [IX, IY]].copy()
        pos = apply_diffusion_vectorized(pos, None, dt, self.E_field, rng)
        particles._buf[idx, IX] = pos[:, 0]
        particles._buf[idx, IY] = pos[:, 1]

        # Hors zone → mort
        out_mask = particles._buf[idx, IY] < self.y_bottom
        particles._buf[idx[out_mask], IALIVE] = 0.0

    def update_field(self, E_field: float) -> None:
        self.E_field  = E_field
        self._v_drift = drift_velocity(E_field)