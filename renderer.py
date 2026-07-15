"""
renderer.py — Rendu Matplotlib optimisé par blitting
=====================================================
STRATÉGIE ANTI-GOULOT :
  - Fond statique dessiné une seule fois (draw_static_background)
  - Seuls les artistes dynamiques sont mis à jour (set_data / set_xdata)
  - FuncAnimation avec blit=True → seuls les artistes retournés sont retraçés
  - Pas de cla() / clf() pendant l'animation
  - Scatter remplacé par Line2D avec marker (beaucoup plus rapide)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from engine import (SimulationEngine, Y_SOURCE, Y_AV1_TOP, Y_AV1_BOT,
                    Y_DRIFT_BOT, Y_AV2_TOP, Y_AV2_BOT, Y_SENSOR)


# ─── Palette ──────────────────────────────────────────────────────────────────
CLR_BG      = '#0d1117'
CLR_AV      = '#1a2744'
CLR_DRIFT   = '#0f1f0f'
CLR_SENSOR  = '#ff6b35'
CLR_SRC0    = '#00d4ff'
CLR_SRC1    = '#ff4fcf'
CLR_ELEC    = '#a8e6cf'
CLR_HIST    = '#4fc3f7'
CLR_HIST0   = CLR_SRC0
CLR_HIST1   = CLR_SRC1
CLR_TEXT    = '#e0e0e0'
CLR_GRID    = '#2a2a3a'

MM = 1e-3  # conversion m → mm pour l'affichage


class VisualizationManager:
    """
    Gère la figure Matplotlib avec blitting optimisé.
    """

    def __init__(self, engine: SimulationEngine):
        self.engine = engine
        self.fig    = None
        self.anim   = None
        self._static_drawn = False
        self._steps_per_frame = 8   # pas de simulation par frame d'animation

    # ── Construction de la figure ─────────────────────────────────────────

    def build(self) -> None:
        matplotlib.rcParams.update({
            'font.family'     : 'monospace',
            'axes.facecolor'  : CLR_BG,
            'figure.facecolor': CLR_BG,
            'text.color'      : CLR_TEXT,
            'axes.labelcolor' : CLR_TEXT,
            'xtick.color'     : CLR_TEXT,
            'ytick.color'     : CLR_TEXT,
            'axes.edgecolor'  : CLR_GRID,
            'grid.color'      : CLR_GRID,
        })

        self.fig = plt.figure(figsize=(16, 9), facecolor=CLR_BG)
        gs  = GridSpec(3, 3, figure=self.fig,
                       left=0.06, right=0.97,
                       top=0.93, bottom=0.08,
                       hspace=0.45, wspace=0.35)

        # Panneau principal : vue 2D du détecteur
        self.ax_det = self.fig.add_subplot(gs[:, 0])

        # Histogramme capteur
        self.ax_hist = self.fig.add_subplot(gs[0, 1:])

        # Distribution temporelle
        self.ax_time = self.fig.add_subplot(gs[1, 1])

        # Statistiques texte
        self.ax_stat = self.fig.add_subplot(gs[1, 2])
        self.ax_stat.axis('off')

        # Histogramme par source
        self.ax_src = self.fig.add_subplot(gs[2, 1:])

        self.fig.suptitle('β-Imager — Simulation 2D par avalanche électronique',
                          color='#a0c4ff', fontsize=13, fontweight='bold')

        self._setup_axes()
        self._draw_static_background()
        self._create_dynamic_artists()

    def _setup_axes(self) -> None:
        """Configure les axes (labels, limites, grilles)."""
        eng = self.engine

        # Vue 2D
        self.ax_det.set_xlim(-6, 6)    # mm
        self.ax_det.set_ylim(
            (Y_SENSOR - 0.5e-3) / MM,
            (Y_SOURCE  + 0.5e-3) / MM
        )
        self.ax_det.set_xlabel('X (mm)')
        self.ax_det.set_ylabel('Y (mm)')
        self.ax_det.set_title('Vue 2D du détecteur', color=CLR_TEXT, fontsize=10)
        self.ax_det.grid(True, alpha=0.2)

        # Histogramme global
        self.ax_hist.set_xlim(-6, 6)
        self.ax_hist.set_xlabel('X (mm)')
        self.ax_hist.set_ylabel('Coups')
        self.ax_hist.set_title('Signal capteur (histogramme)', color=CLR_TEXT, fontsize=10)
        self.ax_hist.grid(True, alpha=0.2)

        # Distribution temporelle
        self.ax_time.set_xlabel('Temps (µs)')
        self.ax_time.set_ylabel('Coups / bin')
        self.ax_time.set_title('Distribution temporelle', color=CLR_TEXT, fontsize=10)
        self.ax_time.grid(True, alpha=0.2)

        # Par source
        self.ax_src.set_xlim(-6, 6)
        self.ax_src.set_xlabel('X (mm)')
        self.ax_src.set_ylabel('Coups')
        self.ax_src.set_title('Signal par source', color=CLR_TEXT, fontsize=10)
        self.ax_src.grid(True, alpha=0.2)

    def _draw_static_background(self) -> None:
        """
        Dessine le fond statique : zones du détecteur, labels.
        Appelé une seule fois — sauvegardé par blit.
        """
        ax = self.ax_det

        def band(y_top, y_bot, color, alpha, label):
            ax.axhspan(y_bot / MM, y_top / MM,
                       color=color, alpha=alpha, zorder=1)
            yc = (y_top + y_bot) / 2.0 / MM
            ax.text(5.5, yc, label,
                    color=CLR_TEXT, fontsize=7, ha='right', va='center',
                    zorder=2, style='italic')

        band(Y_SOURCE,   Y_AV1_TOP,  '#111122', 0.8, 'Espace source')
        band(Y_AV1_TOP,  Y_AV1_BOT,  CLR_AV,   0.7, 'Avalanche 1')
        band(Y_AV1_BOT,  Y_DRIFT_BOT,CLR_DRIFT, 0.7, 'Dérive')
        band(Y_AV2_TOP,  Y_AV2_BOT,  CLR_AV,   0.7, 'Avalanche 2')

        # Ligne capteur
        ax.axhline(Y_SENSOR / MM, color=CLR_SENSOR, lw=2, zorder=3,
                   label='Capteur')

        # Icônes sources
        for src in self.engine.sources:
            clr = CLR_SRC0 if src.source_id == 0 else CLR_SRC1
            ax.plot(src.x_pos / MM, src.y_pos / MM,
                    's', color=clr, ms=10, zorder=5,
                    markeredgecolor='white', markeredgewidth=0.5)
            ax.text(src.x_pos / MM, src.y_pos / MM + 0.3,
                    f'S{src.source_id}', color=clr,
                    fontsize=7, ha='center', zorder=5)

        self._static_drawn = True

    def _create_dynamic_artists(self) -> None:
        """
        Crée les artistes Matplotlib qui seront mis à jour à chaque frame.
        Line2D avec marker est ~5× plus rapide que scatter().
        """
        # ── Vue 2D : nuage d'électrons (deux couleurs selon source) ──
        self.ln_e0, = self.ax_det.plot(
            [], [], 'o', ms=1.5, color=CLR_SRC0, alpha=0.6,
            zorder=4, label='Source 0', markeredgewidth=0)
        self.ln_e1, = self.ax_det.plot(
            [], [], 'o', ms=1.5, color=CLR_SRC1, alpha=0.6,
            zorder=4, label='Source 1', markeredgewidth=0)

        # Barre de progression temporelle
        self.ln_prog, = self.ax_det.plot(
            [], [], '-', lw=1, color='#ffff00', alpha=0.4, zorder=3)

        # ── Histogramme global ──
        det     = self.engine.detector
        bx      = det.bin_centers / MM
        zeros   = np.zeros(len(bx))
        self.bar_hist = self.ax_hist.bar(
            bx, zeros,
            width=(bx[1] - bx[0]) * 0.9,
            color=CLR_HIST, alpha=0.8, zorder=2)

        # ── Histogramme par source ──
        self.bar_s0 = self.ax_src.bar(
            bx, zeros,
            width=(bx[1] - bx[0]) * 0.9,
            color=CLR_SRC0, alpha=0.6, label='Source 0', zorder=2)
        self.bar_s1 = self.ax_src.bar(
            bx, zeros,
            width=(bx[1] - bx[0]) * 0.9,
            color=CLR_SRC1, alpha=0.6, label='Source 1', zorder=2)
        self.ax_src.legend(fontsize=8)

        # ── Distribution temporelle ──
        self.ln_time, = self.ax_time.plot([], [], '-', lw=1.5,
                                           color='#ffd700', zorder=2)

        # ── Texte statistiques ──
        self.txt_stat = self.ax_stat.text(
            0.05, 0.95, '', transform=self.ax_stat.transAxes,
            color=CLR_TEXT, fontsize=9, va='top', fontfamily='monospace')

        # ── Listes des artistes blit ──
        self._blit_artists = [
            self.ln_e0, self.ln_e1, self.ln_prog,
            self.ln_time, self.txt_stat,
        ]

    # ── Mise à jour frame ─────────────────────────────────────────────────

    def update(self, frame: int):
        """Appelé par FuncAnimation à chaque frame."""
        eng = self.engine

        # Avancer la simulation
        if eng.running and not eng.finished:
            eng.step(self._steps_per_frame)

        # ── Nuage d'électrons ──
        p = eng.particles
        if p.count > 0:
            buf    = p._buf[:p.count]
            alive  = buf[:, 4] > 0.5
            if alive.any():
                ab = buf[alive]
                mask0 = ab[:, 7] < 0.5
                mask1 = ab[:, 7] >= 0.5

                # Sous-échantillonnage si trop de particules (>5000)
                def subsample(arr, n_max=4000):
                    if len(arr) > n_max:
                        idx = np.random.choice(len(arr), n_max, replace=False)
                        return arr[idx]
                    return arr

                ab0 = subsample(ab[mask0])
                ab1 = subsample(ab[mask1])

                self.ln_e0.set_data(ab0[:, 0] / MM, ab0[:, 1] / MM)
                self.ln_e1.set_data(ab1[:, 0] / MM, ab1[:, 1] / MM)
            else:
                self.ln_e0.set_data([], [])
                self.ln_e1.set_data([], [])
        else:
            self.ln_e0.set_data([], [])
            self.ln_e1.set_data([], [])

        # ── Histogramme global (mise à jour des hauteurs, pas recréation) ──
        det = eng.detector
        h   = det.histogram
        if h.max() > 0:
            for bar, height in zip(self.bar_hist, h):
                bar.set_height(height)
            self.ax_hist.set_ylim(0, h.max() * 1.15)

        # ── Histogramme par source ──
        h0, h1 = det.hist_src0, det.hist_src1
        mx = max(h0.max(), h1.max(), 1)
        for bar, hv in zip(self.bar_s0, h0):
            bar.set_height(hv)
        for bar, hv in zip(self.bar_s1, h1):
            bar.set_height(hv)
        self.ax_src.set_ylim(0, mx * 1.15)

        # ── Distribution temporelle ──
        if len(det.detect_times) > 10:
            t_arr = np.array(det.detect_times) * 1e6   # → µs
            counts, edges = np.histogram(t_arr, bins=min(50, len(t_arr)//5 + 1))
            centers = 0.5 * (edges[:-1] + edges[1:])
            self.ln_time.set_data(centers, counts)
            self.ax_time.set_xlim(0, eng.t_max * 1e6)
            self.ax_time.set_ylim(0, counts.max() * 1.15)

        # ── Statistiques ──
        gain = eng.gain_total
        txt = (
            f"  Temps  : {eng.t_now*1e6:6.2f} µs\n"
            f"  Émis   : {eng.n_emitted:>6d}\n"
            f"  Détect : {eng.n_detected:>6d}\n"
            f"  Actifs : {eng.particles.count:>6d}\n"
            f"  Gain   : {gain:>8.2f}\n"
            f"  Av1 E  : {eng.avalanche1.E_field/1e3:>6.1f} kV/m\n"
            f"  Drift E: {eng.drift.E_field/1e3:>6.1f} kV/m\n"
            f"  Av2 E  : {eng.avalanche2.E_field/1e3:>6.1f} kV/m\n"
            f"  {'[FIN]' if eng.finished else '[RUN]' if eng.running else '[PAUSE]'}"
        )
        self.txt_stat.set_text(txt)

        return self._blit_artists

    # ── Lancement de l'animation ─────────────────────────────────────────

    def start_animation(self, interval_ms: int = 40) -> None:
        """
        Lance FuncAnimation avec blit=True pour les performances.
        interval_ms : délai entre frames (40ms = 25 fps)
        """
        self.anim = FuncAnimation(
            self.fig,
            self.update,
            interval=interval_ms,
            blit=True,
            cache_frame_data=False,
        )