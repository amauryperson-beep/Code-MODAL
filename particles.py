"""
particles.py — Représentation vectorisée des électrons
=======================================================
DESIGN : aucun objet Python par électron.
Tous les électrons sont stockés dans des tableaux NumPy plats.

Colonnes du tableau d'état `state` (N, STATE_DIM) :
  0 : x      — position transverse (m)
  1 : y      — profondeur (m)
  2 : vx     — vitesse transverse (m/s)
  3 : vy     — vitesse longitudinale (m/s)
  4 : alive  — 1.0 = actif, 0.0 = absorbé/hors zone
  5 : age    — temps depuis l'émission (s)
  6 : weight — poids statistique (1 = primaire, N = secondaire agrégé)
  7 : source — indice de source (0 ou 1)
"""

import numpy as np

STATE_DIM = 8

# Indices nommés pour lisibilité
IX, IY, IVX, IVY, IALIVE, IAGE, IWEIGHT, ISOURCE = range(STATE_DIM)


class ParticleArray:
    """
    Conteneur vectorisé de tous les électrons de la simulation.
    Les opérations sont entièrement vectorisées via NumPy.
    La capacité est allouée une fois ; `count` track le nombre d'actifs.
    """

    def __init__(self, max_particles: int = 50_000):
        self.max_particles = max_particles
        self._buf = np.zeros((max_particles, STATE_DIM), dtype=np.float32)
        self.count = 0   # nombre de particules actuellement actives

    # ── Propriétés d'accès rapide ───────────────────────────────────────────

    @property
    def state(self) -> np.ndarray:
        """Vue sur les particules actives uniquement."""
        return self._buf[:self.count]

    @property
    def x(self)     -> np.ndarray: return self._buf[:self.count, IX]
    @property
    def y(self)     -> np.ndarray: return self._buf[:self.count, IY]
    @property
    def vx(self)    -> np.ndarray: return self._buf[:self.count, IVX]
    @property
    def vy(self)    -> np.ndarray: return self._buf[:self.count, IVY]
    @property
    def alive(self) -> np.ndarray: return self._buf[:self.count, IALIVE]
    @property
    def age(self)   -> np.ndarray: return self._buf[:self.count, IAGE]
    @property
    def weight(self)-> np.ndarray: return self._buf[:self.count, IWEIGHT]
    @property
    def source(self)-> np.ndarray: return self._buf[:self.count, ISOURCE]

    # ── Ajout de particules ─────────────────────────────────────────────────

    def add(self, x: np.ndarray, y: np.ndarray,
            vx: np.ndarray, vy: np.ndarray,
            source_id: int = 0,
            weight: float = 1.0) -> None:
        """Ajoute un lot de nouvelles particules au buffer."""
        n = len(x)
        if n == 0:
            return
        end = min(self.count + n, self.max_particles)
        n   = end - self.count
        if n <= 0:
            return

        sl = slice(self.count, end)
        self._buf[sl, IX]      = x[:n]
        self._buf[sl, IY]      = y[:n]
        self._buf[sl, IVX]     = vx[:n]
        self._buf[sl, IVY]     = vy[:n]
        self._buf[sl, IALIVE]  = 1.0
        self._buf[sl, IAGE]    = 0.0
        self._buf[sl, IWEIGHT] = weight
        self._buf[sl, ISOURCE] = float(source_id)
        self.count = end

    def add_secondary(self, parent_x: np.ndarray, parent_y: np.ndarray,
                      vy_base: float,
                      counts: np.ndarray,
                      rng: np.random.Generator) -> None:
        """
        Ajoute des électrons secondaires d'avalanche.
        `counts[i]` = nombre de secondaires pour le i-ème parent.
        Les secondaires héritent de la position du parent + petite diffusion.
        """
        total = int(counts.sum())
        if total == 0:
            return

        # Répliquer les positions parentes
        rep_x = np.repeat(parent_x, counts)
        rep_y = np.repeat(parent_y, counts)

        # Petite dispersion initiale (σ ~ 10 µm)
        rep_x += rng.normal(0.0, 1e-5, total)
        rep_y += rng.normal(0.0, 1e-5, total)

        vx_sec = rng.normal(0.0, 1e3, total).astype(np.float32)
        vy_sec = np.full(total, vy_base, dtype=np.float32)

        self.add(rep_x.astype(np.float32), rep_y.astype(np.float32),
                 vx_sec, vy_sec, weight=1.0)

    # ── Compactage (suppression des particules mortes) ──────────────────────

    def compact(self) -> int:
        """
        Supprime les particules mortes (alive == 0) du buffer.
        Retourne le nombre de particules supprimées.
        Appelé périodiquement pour garder le buffer compact.
        """
        if self.count == 0:
            return 0
        alive_mask = self._buf[:self.count, IALIVE] > 0.5
        n_alive    = alive_mask.sum()
        n_removed  = self.count - n_alive
        if n_removed > 0:
            self._buf[:n_alive] = self._buf[:self.count][alive_mask]
            self.count = n_alive
        return n_removed

    def reset(self) -> None:
        """Remet tous les compteurs à zéro."""
        self.count = 0
        # Pas besoin de zeroing — le count protège l'accès

    def __len__(self) -> int:
        return self.count