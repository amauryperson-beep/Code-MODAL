"""
detector.py — Capteur final 1D
================================
Le capteur est une ligne horizontale à y = y_sensor.
Il accumule les impacts en construisant un histogramme.
"""

import numpy as np
from particles import ParticleArray, IX, IY, IALIVE, ISOURCE


class Detector1D:
    """
    Capteur linéaire 1D.
    Collecte les électrons qui atteignent y <= y_sensor.
    """

    def __init__(self, y_sensor: float,
                 x_min: float = -5e-3,
                 x_max: float =  5e-3,
                 n_bins: int  = 200):
        self.y_sensor = y_sensor
        self.x_min    = x_min
        self.x_max    = x_max
        self.n_bins   = n_bins

        self.bin_edges   = np.linspace(x_min, x_max, n_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

        # Histogramme cumulatif
        self.histogram      = np.zeros(n_bins, dtype=np.float64)
        self.hist_src0      = np.zeros(n_bins, dtype=np.float64)
        self.hist_src1      = np.zeros(n_bins, dtype=np.float64)
        self.n_detected     = 0
        self.detect_times   = []          # pour distribution temporelle
        self.t_current      = 0.0

    def collect(self, particles: ParticleArray, t_now: float) -> int:
        """
        Collecte les électrons ayant atteint le capteur.
        Retourne le nombre d'électrons détectés ce pas de temps.
        """
        mask = (particles.y <= self.y_sensor) & (particles.alive > 0.5)
        if not mask.any():
            return 0

        idx = np.where(mask)[0]
        xs  = particles._buf[idx, IX]

        # Histogramme global
        counts, _ = np.histogram(xs, bins=self.bin_edges)
        self.histogram += counts

        # Par source
        src = particles._buf[idx, ISOURCE]
        xs0 = xs[src < 0.5];  xs1 = xs[src >= 0.5]
        if len(xs0):
            c0, _ = np.histogram(xs0, bins=self.bin_edges)
            self.hist_src0 += c0
        if len(xs1):
            c1, _ = np.histogram(xs1, bins=self.bin_edges)
            self.hist_src1 += c1

        n = len(idx)
        self.n_detected += n
        self.detect_times.extend([t_now] * n)
        self.t_current = t_now

        # Tuer les électrons détectés
        particles._buf[idx, IALIVE] = 0.0
        return n

    def reset(self) -> None:
        self.histogram[:] = 0
        self.hist_src0[:] = 0
        self.hist_src1[:] = 0
        self.n_detected   = 0
        self.detect_times = []

    def export_csv(self, filename: str) -> None:
        """Exporte l'histogramme en CSV."""
        data = np.column_stack([
            self.bin_centers * 1e3,   # en mm
            self.histogram,
            self.hist_src0,
            self.hist_src1
        ])
        np.savetxt(filename, data,
                   header='x_mm, counts_total, counts_src0, counts_src1',
                   delimiter=',', comments='')
        print(f"[Detector] Exporté → {filename}")