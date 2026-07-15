"""
engine.py — Moteur de simulation vectorisé
============================================
SimulationEngine orchestre :
  - l'émission des sources
  - le pas de temps vectorisé sur toutes les zones
  - la collecte par le capteur
  - la gestion du temps simulé

Optimisations clés :
  - Un seul tableau NumPy pour tous les électrons
  - Compact() appelé périodiquement (pas à chaque pas)
  - dt adaptatif basé sur la vitesse max
"""

import numpy as np
from particles import ParticleArray, IY, IALIVE
from regions  import Source, AvalancheRegion, DriftRegion
from detector import Detector1D

# ─── Géométrie du détecteur (en mètres) ──────────────────────────────────────
#
#   y = 0       ── sources ──────────────────────────────────
#   y = -1 mm   ── top avalanche 1 ──────────────────────────
#   y = -2 mm   ── bottom avalanche 1 ────────────────────────
#   y = -7 mm   ── bottom zone dérive ───────────────────────
#   y = -8 mm   ── top avalanche 2 ──────────────────────────
#   y = -9 mm   ── bottom avalanche 2 = capteur ─────────────

Y_SOURCE   =  0.0
Y_AV1_TOP  = -1e-3
Y_AV1_BOT  = -2e-3
Y_DRIFT_BOT= -7e-3
Y_AV2_TOP  = -8e-3
Y_AV2_BOT  = -9e-3
Y_SENSOR   = -9.0e-3

COMPACT_EVERY = 20    # steps entre deux compactages
MAX_PARTICLES = 80_000


class SimulationEngine:
    """
    Orchestre la simulation complète.
    """

    def __init__(self,
                 # Sources
                 source_separation: float = 2e-3,
                 source_rate: float       = 300.0,
                 # Champs (V/m)
                 E_av1: float  = 5e4,
                 E_drift: float= 1e3,
                 E_av2: float  = 8e4,
                 # Townsend
                 alpha1: float = 2e3,
                 alpha2: float = 4e3,
                 # Simulation
                 t_max: float  = 5e-5,
                 dt_base: float= 5e-10,
                 seed: int     = 42):

        self.t_max    = t_max
        self.dt       = dt_base
        self.t_now    = 0.0
        self._step_n  = 0
        self.running  = False
        self.finished = False

        self.rng = np.random.default_rng(seed)

        # Particules
        self.particles = ParticleArray(MAX_PARTICLES)

        # Sources
        dx = source_separation / 2.0
        self.sources = [
            Source(x_pos=-dx, y_pos=Y_SOURCE, rate=source_rate, source_id=0),
            Source(x_pos=+dx, y_pos=Y_SOURCE, rate=source_rate, source_id=1),
        ]

        # Zones physiques
        self.avalanche1 = AvalancheRegion(Y_AV1_TOP, Y_AV1_BOT,
                                          E_field=E_av1, alpha_eff=alpha1,
                                          label="Avalanche 1")
        self.drift      = DriftRegion(Y_AV1_BOT, Y_DRIFT_BOT,
                                      E_field=E_drift, label="Dérive")
        self.avalanche2 = AvalancheRegion(Y_AV2_TOP, Y_AV2_BOT,
                                          E_field=E_av2, alpha_eff=alpha2,
                                          label="Avalanche 2")

        # Capteur
        self.detector = Detector1D(y_sensor=Y_SENSOR,
                                   x_min=-6e-3, x_max=6e-3, n_bins=200)

        # Pré-calculer les plannings d'émission
        self._prebuild_schedules()

        # Statistiques
        self.n_emitted  = 0
        self.n_detected = 0
        self.gain_history = []

    # ── Initialisation ───────────────────────────────────────────────────────

    def _prebuild_schedules(self) -> None:
        for src in self.sources:
            src.prebuild_schedule(self.t_max, self.rng)

    def reset(self) -> None:
        self.t_now    = 0.0
        self._step_n  = 0
        self.running  = False
        self.finished = False
        self.particles.reset()
        self.detector.reset()
        self.n_emitted  = 0
        self.n_detected = 0
        self.gain_history = []
        self.rng = np.random.default_rng(42)
        self._prebuild_schedules()

    # ── Pas de temps ────────────────────────────────────────────────────────

    def step(self, n_steps: int = 5) -> None:
        """
        Avance la simulation de `n_steps` pas de temps.
        Retourne rapidement si finished ou paused.
        """
        if self.finished:
            return

        for _ in range(n_steps):
            if self.t_now >= self.t_max:
                self.finished = True
                self.running  = False
                break

            # Émission des sources
            for src in self.sources:
                n = src.emit(self.t_now, self.particles, self.rng)
                self.n_emitted += n

            # Tuer les électrons qui ont dépassé le capteur ou qui sont hors X
            if self.particles.count > 0:
                buf = self.particles._buf[:self.particles.count]
                # Hors X
                out_x = (buf[:, 0] < -8e-3) | (buf[:, 0] > 8e-3)
                # Déjà sous le capteur
                out_y = buf[:, 1] < Y_SENSOR - 1e-4
                buf[out_x | out_y, IALIVE.real if hasattr(IALIVE, 'real') else IALIVE] = 0.0

            # Zones physiques (ordre : avalanche1 → dérive → avalanche2)
            self.avalanche1.step(self.particles, self.dt, self.rng)
            self.drift.step(self.particles, self.dt, self.rng)
            self.avalanche2.step(self.particles, self.dt, self.rng)

            # Transfert des électrons qui tombent entre av1 et la dérive
            self._handoff_av1_to_drift()

            # Collecte
            n_det = self.detector.collect(self.particles, self.t_now)
            self.n_detected += n_det

            # Âge
            if self.particles.count > 0:
                self.particles._buf[:self.particles.count, 5] += self.dt

            # Compactage périodique
            self._step_n += 1
            if self._step_n % COMPACT_EVERY == 0:
                self.particles.compact()

            self.t_now += self.dt

    def _handoff_av1_to_drift(self) -> None:
        """
        Les électrons qui sortent de l'avalanche 1 par le bas
        entrent en zone de dérive : réinitialiser vy à la vitesse de dérive.
        """
        if self.particles.count == 0:
            return
        buf  = self.particles._buf[:self.particles.count]
        mask = (buf[:, 1] <= self.drift.y_top) & \
               (buf[:, 1] >= self.drift.y_bottom) & \
               (buf[:, 4] > 0.5)
        if mask.any():
            buf[mask, 3] = -self.drift._v_drift  # IVY

    # ── Paramètres dynamiques ───────────────────────────────────────────────

    def set_E_av1(self, E: float)   -> None: self.avalanche1.E_field = E
    def set_E_drift(self, E: float) -> None: self.drift.update_field(E)
    def set_E_av2(self, E: float)   -> None: self.avalanche2.E_field = E
    def set_alpha1(self, a: float)  -> None: self.avalanche1.alpha   = a
    def set_alpha2(self, a: float)  -> None: self.avalanche2.alpha   = a
    def set_rate(self, r: float)    -> None:
        for src in self.sources:
            src.rate = r

    def set_separation(self, d: float) -> None:
        self.sources[0].x_pos = -d / 2.0
        self.sources[1].x_pos = +d / 2.0

    # ── Statistiques ────────────────────────────────────────────────────────

    @property
    def progress(self) -> float:
        """Avancement de 0 à 1."""
        return min(1.0, self.t_now / self.t_max)

    @property
    def gain_total(self) -> float:
        if self.n_emitted == 0:
            return 0.0
        return self.n_detected / max(1, self.n_emitted)