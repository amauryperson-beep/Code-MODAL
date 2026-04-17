"""
main.py – Visualisation Pygame de l'avalanche de Townsend
Commandes :
  ESPACE      Lecture / Pause
  ← →         Reculer / Avancer d'un pas
  R           Relancer une nouvelle simulation
  +  -        Tension ±100 V  (relance)
  [  ]        Temps de recombinaison ±10 pas
  H           Afficher / fermer l'histogramme
  E           Exporter données (CSV + PNG)
  Q / ESC     Quitter
"""

import sys, math, time, random, os, csv
import pygame
import pygame.gfxdraw
from physics import AvalancheSimulation

# ── Palette ──────────────────────────────────────────────────────────────────
BG           = (8,   10,  20)
ANODE_COL    = (255,  80,  80)
CATHODE_COL  = (80,  160, 255)
ATOM_NEUT    = (60,   80, 100)
ATOM_ION     = (255, 160,  30)
ATOM_EXC     = (180, 100, 255)
ATOM_RECOMB  = (60,  210, 130)   # recombination flash: green
ELECTRON_C   = (100, 220, 255)
ION_C        = (255, 130,  30)
TRAIL_E      = (40,  120, 180)
TRAIL_I      = (150,  70,  10)
TEXT_COL     = (200, 210, 230)
PANEL_COL    = (15,   20,  35)
SLIDER_BG    = (35,   45,  65)
SLIDER_FG    = (80,  160, 255)
IONISE_FLASH = (255, 220,  60)
HIST_BAR     = (80,  160, 255)
HIST_BG      = (12,   16,  28)
HIST_GRID    = (30,   40,  60)
EXPORT_OK    = (60,  210, 130)

# ── Dimensions ───────────────────────────────────────────────────────────────
WIN_W, WIN_H = int(960*1.1), int(700*1.1)
SIM_W, SIM_H = int(660*1.1), int(520*1.1)
SIM_X, SIM_Y = int(1.1*20),  int(1.1*80)
PANEL_X      = SIM_X + SIM_W + 20
PANEL_W      = WIN_W - PANEL_X - 10
SLIDER_H     = 28
SLIDER_Y     = WIN_H - 50
SLIDER_X     = SIM_X
SLIDER_W     = SIM_W

# Recombination-time slider (bottom-right panel area)
RECOMB_SLIDER_X = PANEL_X + 8
RECOMB_SLIDER_W = PANEL_W - 16

N_STEPS = 400
N_ATOMS = 160*9
FPS     = 50

# Histogram window
HIST_W, HIST_H   = int(700*1.1), int(420*1.1)
HIST_BINS        = 28
HIST_MARGIN      = 60    # px around the chart area

# ── Helpers ──────────────────────────────────────────────────────────────────

def draw_rrect(surf, color, rect, r=6, alpha=255):
    if alpha < 255:
        s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
        pygame.draw.rect(s, (*color, alpha), (0, 0, rect[2], rect[3]), border_radius=r)
        surf.blit(s, (rect[0], rect[1]))
    else:
        pygame.draw.rect(surf, color, rect, border_radius=r)


def draw_glow(surf, color, cx, cy, r):
    for i in range(3, 0, -1):
        alpha = 60 - i * 15
        d = r * i * 2
        s = pygame.Surface((d * 2, d * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, alpha), (d, d), d)
        surf.blit(s, (cx - d, cy - d))
    pygame.draw.circle(surf, color, (int(cx), int(cy)), r)


# ── Histogram export (pure pygame, no matplotlib) ────────────────────────────

def render_histogram(detections_x: list, sim_width: int,
                     voltage: int, recomb_steps: int,
                     n_bins: int = HIST_BINS) -> pygame.Surface:
    """
    Render a histogram of cathode-hit x-positions onto a Surface.
    Returns the Surface (can be blitted or saved).
    """
    surf = pygame.Surface((HIST_W, HIST_H))
    surf.fill(HIST_BG)

    font_s  = pygame.font.SysFont("monospace", 11)
    font_m  = pygame.font.SysFont("monospace", 13, bold=True)
    font_t  = pygame.font.SysFont("monospace", 10)

    chart_x = HIST_MARGIN
    chart_y = HIST_MARGIN
    chart_w = HIST_W - 2 * HIST_MARGIN
    chart_h = HIST_H - 2 * HIST_MARGIN - 30   # room for title at bottom

    # Grid
    pygame.draw.rect(surf, HIST_GRID,
                     (chart_x, chart_y, chart_w, chart_h), 1)
    n_grid = 5
    for gi in range(1, n_grid):
        gy = chart_y + int(gi / n_grid * chart_h)
        pygame.draw.line(surf, HIST_GRID,
                         (chart_x, gy), (chart_x + chart_w, gy))

    if not detections_x:
        msg = font_m.render("Aucune détection enregistrée", True, (120, 130, 160))
        surf.blit(msg, (HIST_W // 2 - msg.get_width() // 2,
                        HIST_H // 2))
        return surf

    # Bin counts
    min_x, max_x = 0, sim_width
    bins = [0] * n_bins
    for x in detections_x:
        b = int((x - min_x) / (max_x - min_x) * n_bins)
        b = max(0, min(n_bins - 1, b))
        bins[b] += 1

    max_count = max(bins) if bins else 1
    bar_w = chart_w / n_bins

    # Bars
    for i, count in enumerate(bins):
        bx = chart_x + int(i * bar_w)
        bh = int(count / max_count * chart_h)
        by = chart_y + chart_h - bh
        bw = max(1, int(bar_w) - 2)

        # Gradient effect: brighter at top
        for row in range(bh):
            frac   = 1 - row / max(bh, 1)
            bright = tuple(int(c * (0.5 + 0.5 * frac)) for c in HIST_BAR)
            pygame.draw.line(surf, bright,
                             (bx, by + row), (bx + bw, by + row))

        # Count label on tall bars
        if count > 0 and bh > 20:
            lbl = font_t.render(str(count), True, (180, 200, 230))
            surf.blit(lbl, (bx + bw // 2 - lbl.get_width() // 2, by - 14))

    # Y-axis labels
    for gi in range(n_grid + 1):
        val = int(max_count * (1 - gi / n_grid))
        gy  = chart_y + int(gi / n_grid * chart_h)
        lbl = font_t.render(str(val), True, (100, 110, 140))
        surf.blit(lbl, (chart_x - lbl.get_width() - 4,
                        gy - lbl.get_height() // 2))

    # X-axis labels (position in px mapped to %)
    n_xlabels = 6
    for xi in range(n_xlabels + 1):
        px_val = int(xi / n_xlabels * sim_width)
        pct    = int(xi / n_xlabels * 100)
        gx     = chart_x + int(xi / n_xlabels * chart_w)
        lbl    = font_t.render(f"{pct}%", True, (100, 110, 140))
        surf.blit(lbl, (gx - lbl.get_width() // 2,
                        chart_y + chart_h + 4))
        pygame.draw.line(surf, HIST_GRID,
                         (gx, chart_y), (gx, chart_y + chart_h))

    # Axis titles
    title = font_m.render(
        f"Distribution des impacts à la cathode  —  V={voltage}V  "
        f"recomb.={recomb_steps if recomb_steps > 0 else 'OFF'} pas",
        True, TEXT_COL)
    surf.blit(title, (HIST_W // 2 - title.get_width() // 2, 10))

    xlabel = font_s.render("Position X sur la cathode (% de la largeur)", True, (120, 130, 160))
    surf.blit(xlabel, (HIST_W // 2 - xlabel.get_width() // 2, HIST_H - 18))

    ylabel_surf = font_s.render("Nombre d'impacts", True, (120, 130, 160))
    ylabel_rot  = pygame.transform.rotate(ylabel_surf, 90)
    surf.blit(ylabel_rot, (6, HIST_H // 2 - ylabel_rot.get_height() // 2))

    # Total
    total_lbl = font_s.render(f"Total : {len(detections_x)} détections  |  {n_bins} bins",
                               True, (140, 150, 170))
    surf.blit(total_lbl, (chart_x, chart_y + chart_h + 18))

    return surf


# ── Main app ─────────────────────────────────────────────────────────────────

class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Simulation : Avalanche de Townsend")
        self.clock  = pygame.time.Clock()

        self.font_s = pygame.font.SysFont("monospace", 12)
        self.font_m = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_t = pygame.font.SysFont("monospace", 11)

        self.voltage          = 500
        self.n_atoms          = N_ATOMS
        self.recomb_steps     = 80     # 0 = disabled
        self.playing          = False
        self.step_idx         = 0
        self.dragging_time    = False
        self.dragging_recomb  = False
        self.flash_list       = []

        # Histogram state
        self.show_hist        = False
        self.hist_surface     = None
        self.export_msg       = ""
        self.export_msg_timer = 0

        self._run_simulation()

    # ── Simulation ────────────────────────────────────────────────────────────

    def _run_simulation(self):
        self.sim = AvalancheSimulation(
            width=SIM_W, height=SIM_H,
            n_atoms=self.n_atoms,
            voltage=self.voltage,
            recombination_steps=self.recomb_steps,
            seed=int(time.time()) % 10000
        )
        self.snapshots = self.sim.run_full(N_STEPS)
        self.total     = len(self.snapshots)
        self.step_idx  = 0
        self.flash_list = []
        self.hist_surface = None

    def _get_detections_up_to_step(self):
        """Return list of X positions of cathode hits up to current step."""
        snap = self.snapshots[self.step_idx]
        return [e['x'] for e in snap['events'] if e['type'] == 'detected']

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self):
        detections = self._get_detections_up_to_step()
        timestamp  = time.strftime("%Y%m%d_%H%M%S")
        base       = f"avalanche_export_{timestamp}"

        # CSV: raw x positions
        csv_path = base + ".csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["impact_x_px",
                             "impact_x_pct",
                             "voltage_V",
                             "recomb_steps"])
            for x in detections:
                writer.writerow([round(x, 2),
                                 round(x / SIM_W * 100, 2),
                                 self.voltage,
                                 self.recomb_steps])

        # PNG: render histogram and save via pygame
        png_path = base + ".png"
        hist_surf = render_histogram(
            detections, SIM_W, self.voltage, self.recomb_steps)
        pygame.image.save(hist_surf, png_path)

        self.export_msg       = f"Exporté : {csv_path}  +  {png_path}"
        self.export_msg_timer = 180   # frames

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def to_screen(self, x, y):
        return int(SIM_X + x), int(SIM_Y + y)

    # ── Drawing: main scene ───────────────────────────────────────────────────

    def draw_scene(self, snap):
        pygame.draw.rect(self.screen, PANEL_COL,
                         (SIM_X, SIM_Y, SIM_W, SIM_H), border_radius=6)

        # Electrodes
        pygame.draw.line(self.screen, ANODE_COL,
                         (SIM_X, SIM_Y), (SIM_X + SIM_W, SIM_Y), 3)
        pygame.draw.line(self.screen, CATHODE_COL,
                         (SIM_X, SIM_Y + SIM_H), (SIM_X + SIM_W, SIM_Y + SIM_H), 3)

        # Field lines (faint)
        for xi in range(0, SIM_W, 30):
            s = pygame.Surface((1, SIM_H), pygame.SRCALPHA)
            s.fill((80, 120, 200, 18))
            self.screen.blit(s, (SIM_X + xi, SIM_Y))

        # Atoms (5-tuple now includes recombine_flash)
        for entry in snap['atoms']:
            ax, ay, ionised, excited, recomb_flash = entry
            sx, sy = self.to_screen(ax, ay)

            if recomb_flash > 0:
                # Recombination glow (green, fading)
                alpha = int(recomb_flash * 180)
                r_size = int(6 + recomb_flash * 10)
                gs = pygame.Surface((r_size * 4, r_size * 4), pygame.SRCALPHA)
                pygame.draw.circle(gs, (*ATOM_RECOMB, alpha),
                                   (r_size * 2, r_size * 2), r_size * 2)
                self.screen.blit(gs, (sx - r_size * 2, sy - r_size * 2))
                pygame.draw.circle(self.screen, ATOM_RECOMB, (sx, sy), 4)

            elif ionised:
                draw_glow(self.screen, ATOM_ION, sx, sy, 6)
            elif excited:
                draw_glow(self.screen, ATOM_EXC, sx, sy, 5)
            else:
                pygame.draw.circle(self.screen, ATOM_NEUT, (sx, sy), 4)
                pygame.draw.circle(self.screen, (90, 110, 140), (sx, sy), 4, 1)

        # Ionisation flashes
        new_flashes = []
        for (fx, fy, timer) in self.flash_list:
            if timer > 0:
                sx, sy = self.to_screen(fx, fy)
                r = int(8 + (1 - timer) * 20)
                alpha = int(timer * 200)
                s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*IONISE_FLASH, alpha), (r, r), r)
                self.screen.blit(s, (sx - r, sy - r))
                new_flashes.append((fx, fy, timer - 0.07))
        self.flash_list = new_flashes

        for ev in snap['events']:
            if ev['type'] == 'ionisation':
                if not any(abs(fx - ev['x']) < 5 and abs(fy - ev['y']) < 5
                           for fx, fy, _ in self.flash_list):
                    self.flash_list.append((ev['x'], ev['y'], 1.0))

        # Ions
        for px, py, trail in snap['ions']:
            sx, sy = self.to_screen(px, py)
            for k in range(1, len(trail)):
                a_pct = k / len(trail)
                color = tuple(int(c * a_pct * 0.5) for c in TRAIL_I)
                pygame.draw.line(self.screen, color,
                                 self.to_screen(*trail[k - 1]),
                                 self.to_screen(*trail[k]), 1)
            draw_glow(self.screen, ION_C, sx, sy, 3)

        # Electrons
        for px, py, trail in snap['electrons']:
            sx, sy = self.to_screen(px, py)
            for k in range(1, len(trail)):
                a_pct = k / len(trail)
                color = tuple(int(c * a_pct * 0.7) for c in TRAIL_E)
                pygame.draw.line(self.screen, color,
                                 self.to_screen(*trail[k - 1]),
                                 self.to_screen(*trail[k]), 1)
            draw_glow(self.screen, ELECTRON_C, sx, sy, 4)

        # Electrode labels
        lbl_a = self.font_s.render("+ Anode", True, ANODE_COL)
        lbl_c = self.font_s.render("- Cathode (détecteur)", True, CATHODE_COL)
        self.screen.blit(lbl_a, (SIM_X + 4, SIM_Y + 4))
        self.screen.blit(lbl_c, (SIM_X + 4, SIM_Y + SIM_H - 16))

    # ── Drawing: right panel ──────────────────────────────────────────────────

    def draw_panel(self, snap):
        x, y = PANEL_X, SIM_Y
        w    = PANEL_W
        draw_rrect(self.screen, PANEL_COL, (x, y, w, SIM_H), r=6)

        def row(label, value, col=TEXT_COL):
            nonlocal y
            y += 22
            lbl = self.font_t.render(label, True, (120, 130, 160))
            val = self.font_m.render(str(value), True, col)
            self.screen.blit(lbl, (x + 8, y))
            self.screen.blit(val, (x + 8, y + 13))

        y = SIM_Y + 6
        self.screen.blit(self.font_l.render("Paramètres", True, TEXT_COL),
                         (x + 8, y))

        y += 30
        row("Tension", f"{self.voltage} V", ANODE_COL)
        row("Gaz", self.sim.gas)
        row("Pot. ionisation",
            f"{15.7} eV" if self.sim.gas == 'Ar' else "21.6 eV")
        row("Champ E", f"{self.sim.E_field:.3f} eV/px")
        row("Atomes", self.n_atoms)

        # Recombination slider
        y += 14
        recomb_label = ("OFF" if self.recomb_steps == 0
                        else f"{self.recomb_steps} pas")
        self.screen.blit(
            self.font_t.render("Recombinaison", True, (120, 130, 160)),
            (x + 8, y))
        self.screen.blit(
            self.font_m.render(recomb_label, True, ATOM_RECOMB),
            (x + 8, y + 13))

        y += 28
        rs_y = y
        rs_w = w - 16
        self._recomb_slider_rect = (x + 8, rs_y, rs_w, 14)
        draw_rrect(self.screen, SLIDER_BG,
                   self._recomb_slider_rect, r=4)
        max_r = 200
        frac  = self.recomb_steps / max_r
        fill_w = int(frac * rs_w)
        if fill_w > 4:
            draw_rrect(self.screen, ATOM_RECOMB,
                       (x + 8, rs_y, fill_w, 14), r=4)
        pygame.draw.circle(self.screen, (200, 240, 220),
                           (x + 8 + fill_w, rs_y + 7), 8)
        self.screen.blit(
            self.font_t.render("[  ]  ou glisser", True, (80, 100, 120)),
            (x + 8, rs_y + 16))

        y += 38
        self.screen.blit(self.font_l.render("État actuel", True, TEXT_COL),
                         (x + 8, y))

        row("Étape", f"{self.step_idx + 1} / {self.total}")
        row("Électrons libres", snap['n_electrons'], ELECTRON_C)
        row("Atomes ionisés", snap['n_ionised'], ATOM_ION)
        row("Ions Ar⁺", len(snap['ions']), ION_C)
        row("Ionisations totales",
            sum(1 for e in snap['events'] if e['type'] == 'ionisation'),
            IONISE_FLASH)
        row("Recombinées",
            sum(1 for e in snap['events'] if e['type'] == 'recombined'),
            ATOM_RECOMB)
        row("Détections cathode",
            sum(1 for e in snap['events'] if e['type'] == 'detected'),
            CATHODE_COL)

        # Mini electron history graph
        y += 26
        graph_h = 72
        graph_w = w - 16
        graph_rect = (x + 8, y, graph_w, graph_h)
        pygame.draw.rect(self.screen, (20, 28, 48), graph_rect, border_radius=4)
        pygame.draw.rect(self.screen, SLIDER_BG, graph_rect, 1, border_radius=4)
        self.screen.blit(
            self.font_t.render("Électrons libres", True, (100, 110, 140)),
            (x + 8, y - 13))
        history = [s['n_electrons'] for s in self.snapshots[:self.step_idx + 1]]
        max_e = max(history) if history else 1
        if len(history) > 1 and max_e > 0:
            pts = []
            for i, v in enumerate(history):
                gx = x + 8 + int(i / (len(history) - 1) * (graph_w - 2))
                gy = y + graph_h - 4 - int(v / max_e * (graph_h - 8))
                pts.append((gx, gy))
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, ELECTRON_C, False, pts, 2)

        y += graph_h + 16
        self.screen.blit(self.font_l.render("Contrôles", True, TEXT_COL),
                         (x + 8, y))
        controls = [
            ("ESPACE", "Lecture / Pause"),
            ("← →",    "Pas par pas"),
            ("R",      "Nouvelle simulation"),
            ("+  -",   "Tension ±100 V"),
            ("[  ]",   "Recomb. ±10 pas"),
            ("H",      "Histogramme"),
            ("E",      "Exporter CSV+PNG"),
            ("ESC",    "Quitter"),
        ]
        for key, desc in controls:
            y += 17
            self.screen.blit(self.font_s.render(key,  True, SLIDER_FG),
                             (x + 8,      y))
            self.screen.blit(self.font_t.render(desc, True, (140, 150, 170)),
                             (x + 8 + 42, y))

    # ── Drawing: header ───────────────────────────────────────────────────────

    def draw_header(self):
        title = self.font_l.render(
            "  Avalanche de Townsend — Compteur Geiger", True, TEXT_COL)
        self.screen.blit(title, (10, 14))

        state = "▶ LECTURE" if self.playing else "⏸ PAUSE"
        col   = (100, 220, 100) if self.playing else (200, 150, 50)
        self.screen.blit(self.font_m.render(state, True, col),
                         (WIN_W - 130, 14))

        # Export feedback
        if self.export_msg_timer > 0:
            alpha = min(255, self.export_msg_timer * 4)
            msg   = self.font_s.render(self.export_msg, True, EXPORT_OK)
            self.screen.blit(msg, (10, WIN_H - 30))
            self.export_msg_timer -= 1

    # ── Drawing: time scrubber ────────────────────────────────────────────────

    def draw_scrubber(self):
        draw_rrect(self.screen, SLIDER_BG,
                   (SLIDER_X, SLIDER_Y, SLIDER_W, SLIDER_H), r=6)
        fill_w = int(self.step_idx / max(1, self.total - 1) * SLIDER_W)
        if fill_w > 4:
            draw_rrect(self.screen, SLIDER_FG,
                       (SLIDER_X, SLIDER_Y, fill_w, SLIDER_H), r=6)
        thumb_x = SLIDER_X + fill_w
        pygame.draw.circle(self.screen, (220, 230, 255),
                           (thumb_x, SLIDER_Y + SLIDER_H // 2), 10)
        pct = self.step_idx / max(1, self.total - 1) * 100
        self.screen.blit(
            self.font_s.render(
                f"  Temps : étape {self.step_idx + 1}/{self.total}  ({pct:.0f}%)",
                True, TEXT_COL),
            (SLIDER_X, SLIDER_Y - 18))

    # ── Drawing: histogram overlay ────────────────────────────────────────────

    def draw_histogram_overlay(self):
        detections = self._get_detections_up_to_step()
        # Re-render only if needed (lazy)
        if self.hist_surface is None:
            self.hist_surface = render_histogram(
                detections, SIM_W, self.voltage, self.recomb_steps)

        # Centre on screen
        hx = (WIN_W - HIST_W) // 2
        hy = (WIN_H - HIST_H) // 2

        # Dim background
        dim = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 170))
        self.screen.blit(dim, (0, 0))

        self.screen.blit(self.hist_surface, (hx, hy))

        # Close hint
        close = self.font_m.render("H  ou  ESC  pour fermer", True, (140, 150, 170))
        self.screen.blit(close,
                         (hx + HIST_W // 2 - close.get_width() // 2,
                          hy + HIST_H - 22))

    # ── Slider interaction helpers ────────────────────────────────────────────

    def in_time_scrubber(self, mx, my):
        return (SLIDER_X <= mx <= SLIDER_X + SLIDER_W and
                SLIDER_Y - 10 <= my <= SLIDER_Y + SLIDER_H + 10)

    def in_recomb_slider(self, mx, my):
        if not hasattr(self, '_recomb_slider_rect'):
            return False
        rx, ry, rw, rh = self._recomb_slider_rect
        return rx <= mx <= rx + rw and ry - 8 <= my <= ry + rh + 8

    def update_time_from_mouse(self, mx):
        frac = (mx - SLIDER_X) / SLIDER_W
        self.step_idx = int(max(0, min(1, frac)) * (self.total - 1))

    def update_recomb_from_mouse(self, mx):
        rx, ry, rw, rh = self._recomb_slider_rect
        frac = (mx - rx) / rw
        self.recomb_steps = int(max(0, min(1, frac)) * 200)
        self.hist_surface = None   # invalidate histogram

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        running = True
        while running:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.show_hist:
                            self.show_hist = False
                        else:
                            running = False
                    elif event.key == pygame.K_q and not self.show_hist:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                    elif event.key == pygame.K_RIGHT:
                        self.step_idx = min(self.total - 1, self.step_idx + 1)
                        self.hist_surface = None
                    elif event.key == pygame.K_LEFT:
                        self.step_idx = max(0, self.step_idx - 1)
                        self.hist_surface = None
                    elif event.key == pygame.K_r:
                        self._run_simulation()
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS,
                                       pygame.K_KP_PLUS):
                        self.voltage = min(5000, self.voltage + 100)
                        #self._run_simulation()
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.voltage = max(50, self.voltage - 50)
                        #self._run_simulation()
                    elif event.key == pygame.K_RIGHTBRACKET:
                        self.recomb_steps = min(200, self.recomb_steps + 10)
                        self.hist_surface  = None
                    elif event.key == pygame.K_LEFTBRACKET:
                        self.recomb_steps = max(0, self.recomb_steps - 10)
                        self.hist_surface  = None
                    elif event.key == pygame.K_h:
                        self.show_hist = not self.show_hist
                        self.hist_surface = None
                    elif event.key == pygame.K_e:
                        self._export()

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    if self.in_time_scrubber(mx, my):
                        self.dragging_time = True
                        self.playing = False
                        self.update_time_from_mouse(mx)
                        self.hist_surface = None
                    elif self.in_recomb_slider(mx, my):
                        self.dragging_recomb = True
                        self.update_recomb_from_mouse(mx)

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.dragging_time   = False
                    self.dragging_recomb = False

                elif event.type == pygame.MOUSEMOTION:
                    mx, my = event.pos
                    if self.dragging_time:
                        self.update_time_from_mouse(mx)
                        self.hist_surface = None
                    elif self.dragging_recomb:
                        self.update_recomb_from_mouse(mx)

            # Auto-advance
            if self.playing:
                self.step_idx += 1
                if self.step_idx >= self.total:
                    self.step_idx = self.total - 1
                    self.playing  = False
                self.hist_surface = None

            # ── Render ────────────────────────────────────────────────────
            self.screen.fill(BG)
            snap = self.snapshots[min(self.step_idx, self.total - 1)]
            self.draw_header()
            self.draw_scene(snap)
            self.draw_panel(snap)
            self.draw_scrubber()

            if self.show_hist:
                self.draw_histogram_overlay()

            pygame.display.flip()

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    App().run()
