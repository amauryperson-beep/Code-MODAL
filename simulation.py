"""Simulation interactive pygame d'avalanche electronique 2D.

- Panneau de droite avec curseurs interactifs (souris)
- Modification des parametres avec fleches gauche/droite sur le curseur selectionne
- Histogramme des impacts sur la plaque y=0 + fit gaussien
"""

from __future__ import annotations

import math

import numpy as np
import pygame

E_CHARGE = 1.602_176_634e-19
M_ELECTRON = 9.109_383_7015e-31
EV_TO_J = E_CHARGE
PI = math.pi

WINDOW_W = 1450
WINDOW_H = 900
SIM_MARGIN = 20
PANEL_W = 460
FPS = 60
MAX_IMPACTS_STORED = 120_000

BITMAP_5X7 = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "11110", "10001", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "11110", "10000", "10000", "10000", "11111"],
    "F": ["11111", "10000", "11110", "10000", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00001", "00001", "00001", "00001", "10001", "10001", "01110"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "01010", "00100", "00100", "00100", "01010", "10001"],
    "Y": ["10001", "01010", "00100", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00010", "00100", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10011", "10101", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "00110", "00110"],
    ":": ["00000", "00110", "00110", "00000", "00110", "00110", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    "/": ["00001", "00010", "00100", "00100", "01000", "10000", "00000"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    "^": ["00100", "01010", "10001", "00000", "00000", "00000", "00000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "=": ["00000", "11111", "00000", "11111", "00000", "00000", "00000"],
}


def load_fonts() -> tuple[object | None, object | None]:
    try:
        pygame.font.init()
        return pygame.font.SysFont("Consolas", 18), pygame.font.SysFont("Consolas", 14)
    except Exception:
        return None, None


def default_params() -> dict:
    return {
        "width": 5.0e-3,
        "height": 1.0e-2,
        "gas_density": 2.5e25,
        "effective_thickness": 1.0e-6,
        "sim_fraction": 1.0e-12,
        "max_nuclei": 30000.0,
        "sigma_collision": 2.0e-20,
        "ionization_ev": 15.8,
        "source_x_ratio": 0.5,
        "source_y_ratio": 0.98,
        "initial_vx": 0.0,
        "initial_vy": 0.0,
        "E_y": 4.0e5,
        "dt": 8.0e-14,
        "steps_per_frame": 12.0,
        "emission_rate": 3.0e11,
        "seed": 42.0,
    }


def get_controls() -> list[dict]:
    return [
        {"key": "gas_density", "label": "Densite gaz (m^-3)", "min": 1e22, "max": 1e27, "scale": "log", "reset": True},
        {"key": "E_y", "label": "Champ Ey (V/m)", "min": 1e3, "max": 5e6, "scale": "log", "reset": False},
        {"key": "sigma_collision", "label": "Sigma collision (m^2)", "min": 1e-22, "max": 5e-18, "scale": "log", "reset": True},
        {"key": "steps_per_frame", "label": "Pas/frame", "min": 1.0, "max": 40.0, "scale": "linear", "reset": False},
        {"key": "emission_rate", "label": "Emission (s^-1)", "min": 1e9, "max": 2e12, "scale": "log", "reset": False},
        {"key": "source_x_ratio", "label": "Source x/largeur", "min": 0.05, "max": 0.95, "scale": "linear", "reset": False},
    ]


def build_rects() -> tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
    sim_rect = pygame.Rect(SIM_MARGIN, SIM_MARGIN, WINDOW_W - PANEL_W - 3 * SIM_MARGIN, WINDOW_H - 2 * SIM_MARGIN)
    panel_rect = pygame.Rect(sim_rect.right + SIM_MARGIN, SIM_MARGIN, PANEL_W, WINDOW_H - 2 * SIM_MARGIN)
    hist_rect = pygame.Rect(panel_rect.left + 18, panel_rect.bottom - 290, panel_rect.width - 36, 260)
    return sim_rect, panel_rect, hist_rect


def energy_from_velocity(v: np.ndarray) -> float:
    return 0.5 * M_ELECTRON * float(np.dot(v, v))


def sample_e2(available_energy: float, ionization_energy: float, rng: np.random.Generator) -> float:
    if available_energy <= 0.0:
        return 0.0
    a = 1.0 / ionization_energy
    b = 1.0 / (available_energy + ionization_energy)
    u = rng.uniform(0.0, 1.0)
    inv = a - u * (a - b)
    e2 = (1.0 / inv) - ionization_energy
    return float(np.clip(e2, 0.0, available_energy))


def rotate2d(u: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]], dtype=float)


def post_collision_velocities(v_incident: np.ndarray, e1: float, e2: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    p0_vec = M_ELECTRON * v_incident
    p0 = float(np.linalg.norm(p0_vec))
    if p0 <= 0.0:
        return None
    p1 = math.sqrt(max(0.0, 2.0 * M_ELECTRON * e1))
    p2 = math.sqrt(max(0.0, 2.0 * M_ELECTRON * e2))
    if p0 < abs(p1 - p2) or p0 > (p1 + p2):
        return None
    u0 = p0_vec / p0
    cos_theta1 = (p0 * p0 + p1 * p1 - p2 * p2) / (2.0 * p0 * p1)
    cos_theta1 = float(np.clip(cos_theta1, -1.0, 1.0))
    theta1 = math.acos(cos_theta1)
    sign = -1.0 if rng.uniform() < 0.5 else 1.0
    p1_vec = rotate2d(u0, sign * theta1) * p1
    p2_vec = p0_vec - p1_vec
    return p1_vec / M_ELECTRON, p2_vec / M_ELECTRON


def world_to_screen(x: float, y: float, params: dict, sim_rect: pygame.Rect) -> tuple[int, int]:
    sx = sim_rect.left + int((x / params["width"]) * sim_rect.width)
    sy = sim_rect.bottom - int((y / params["height"]) * sim_rect.height)
    return sx, sy


def make_electron(pos: np.ndarray, vel: np.ndarray) -> dict:
    return {"pos": pos.astype(float).copy(), "vel": vel.astype(float).copy(), "alive": True, "trail": [pos.astype(float).copy()]}


def to_slider(value: float, c: dict) -> float:
    if c["scale"] == "log":
        lo = math.log10(c["min"])
        hi = math.log10(c["max"])
        vv = math.log10(max(c["min"], min(c["max"], value)))
        return (vv - lo) / (hi - lo)
    return (value - c["min"]) / (c["max"] - c["min"])


def from_slider(t: float, c: dict) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    if c["scale"] == "log":
        lo = math.log10(c["min"])
        hi = math.log10(c["max"])
        return 10 ** (lo + t * (hi - lo))
    return c["min"] + t * (c["max"] - c["min"])


def clamp_params(params: dict) -> None:
    for c in get_controls():
        k = c["key"]
        params[k] = float(np.clip(params[k], c["min"], c["max"]))


def build_spatial_grid(nuclei: np.ndarray, cell_size: float) -> dict:
    grid: dict[tuple[int, int], list[int]] = {}
    if len(nuclei) == 0:
        return grid
    inv = 1.0 / cell_size
    for i, (x, y) in enumerate(nuclei):
        key = (int(x * inv), int(y * inv))
        grid.setdefault(key, []).append(i)
    return grid


def generate_gas(params: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict, float, float]:
    area = params["width"] * params["height"]
    eff_dens_2d = params["gas_density"] * params["effective_thickness"]
    expected = eff_dens_2d * area * params["sim_fraction"]
    n_nuclei = int(rng.poisson(expected))
    n_nuclei = min(n_nuclei, int(params["max_nuclei"]))
    x = rng.uniform(0.0, params["width"], size=n_nuclei)
    y = rng.uniform(0.0, params["height"], size=n_nuclei)
    nuclei = np.column_stack((x, y)) if n_nuclei > 0 else np.zeros((0, 2), dtype=float)
    ionized = np.zeros(n_nuclei, dtype=bool)
    sigma_sim = params["sigma_collision"] / params["sim_fraction"]
    collision_radius = math.sqrt(sigma_sim / PI)
    grid = build_spatial_grid(nuclei, max(collision_radius * 2.5, 1.0e-6))
    return nuclei, ionized, grid, collision_radius, sigma_sim


def reset_state(params: dict) -> dict:
    clamp_params(params)
    rng = np.random.default_rng(int(params["seed"]))
    nuclei, ionized, grid, collision_radius, sigma_sim = generate_gas(params, rng)
    src = np.array([params["source_x_ratio"] * params["width"], params["source_y_ratio"] * params["height"]], dtype=float)
    primary = make_electron(src, np.array([params["initial_vx"], params["initial_vy"]], dtype=float))
    return {
        "rng": rng,
        "nuclei": nuclei,
        "ionized": ionized,
        "grid": grid,
        "collision_radius": collision_radius,
        "sigma_sim": sigma_sim,
        "electrons": [primary],
        "impacts_x": [],
        "collisions": 0,
        "ionizations": 0,
        "impacts": 0,
        "time": 0.0,
        "paused": False,
        "selected_control": 0,
        "slider_rects": {},
        "dragging_key": None,
    }


def find_collision_index(pos: np.ndarray, state: dict) -> int | None:
    nuclei = state["nuclei"]
    ionized = state["ionized"]
    if len(nuclei) == 0:
        return None
    r = state["collision_radius"]
    cell_size = max(r * 2.5, 1.0e-6)
    cx = int(pos[0] / cell_size)
    cy = int(pos[1] / cell_size)
    r2 = r * r
    best, best_d2 = None, float("inf")
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for idx in state["grid"].get((cx + dx, cy + dy), []):
                if ionized[idx]:
                    continue
                d = nuclei[idx] - pos
                d2 = float(d[0] * d[0] + d[1] * d[1])
                if d2 <= r2 and d2 < best_d2:
                    best, best_d2 = idx, d2
    return best


def spawn_primaries(state: dict, params: dict) -> None:
    rng = state["rng"]
    n = int(rng.poisson(params["emission_rate"] * params["dt"]))
    if n <= 0:
        return
    src = np.array([params["source_x_ratio"] * params["width"], params["source_y_ratio"] * params["height"]], dtype=float)
    for _ in range(n):
        state["electrons"].append(make_electron(src, np.array([params["initial_vx"], params["initial_vy"]], dtype=float)))


def step_physics(state: dict, params: dict) -> None:
    if state["paused"]:
        return
    rng = state["rng"]
    ay = (-E_CHARGE / M_ELECTRON) * params["E_y"]
    accel = np.array([0.0, ay], dtype=float)
    ei = params["ionization_ev"] * EV_TO_J

    for _ in range(int(params["steps_per_frame"])):
        spawn_primaries(state, params)
        state["time"] += params["dt"]

        for e in state["electrons"]:
            if not e["alive"]:
                continue
            e["vel"] += accel * params["dt"]
            e["pos"] += e["vel"] * params["dt"]
            tr = e["trail"]
            tr.append(e["pos"].copy())
            if len(tr) > 20:
                tr.pop(0)

            x, y = float(e["pos"][0]), float(e["pos"][1])
            if y <= 0.0:
                e["alive"] = False
                if 0.0 <= x <= params["width"]:
                    state["impacts"] += 1
                    state["impacts_x"].append(x)
                    if len(state["impacts_x"]) > MAX_IMPACTS_STORED:
                        state["impacts_x"] = state["impacts_x"][-MAX_IMPACTS_STORED:]
                continue
            if x < 0.0 or x > params["width"] or y > params["height"]:
                e["alive"] = False
                continue

            idx = find_collision_index(e["pos"], state)
            if idx is None:
                continue
            state["collisions"] += 1

            e0 = energy_from_velocity(e["vel"])
            if e0 < ei:
                continue

            available = e0 - ei
            solution, e2_final = None, 0.0
            for _ in range(24):
                e2 = sample_e2(available, ei, rng)
                e1 = available - e2
                trial = post_collision_velocities(e["vel"], e1, e2, rng)
                if trial is not None:
                    solution, e2_final = trial, e2
                    break
            if solution is None:
                continue

            state["ionized"][idx] = True
            state["ionizations"] += 1
            v1, v2 = solution
            e["vel"] = v1
            if e2_final > 0.0:
                state["electrons"].append(make_electron(e["pos"], v2))


def draw_text(surface: pygame.Surface, font: object | None, text: str, x: int, y: int, color=(230, 230, 230)) -> None:
    if font is None:
        draw_bitmap_text(surface, text, x, y, color, scale=2)
    else:
        surface.blit(font.render(text, True, color), (x, y))


def draw_bitmap_text(
    surface: pygame.Surface,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    scale: int = 2,
) -> None:
    cursor_x = x
    for ch in text.upper():
        glyph = BITMAP_5X7.get(ch, BITMAP_5X7[" "])
        for row_i, row in enumerate(glyph):
            for col_i, bit in enumerate(row):
                if bit == "1":
                    surface.fill(
                        color,
                        (
                            cursor_x + col_i * scale,
                            y + row_i * scale,
                            scale,
                            scale,
                        ),
                    )
        cursor_x += 6 * scale


def draw_simulation(screen: pygame.Surface, sim_rect: pygame.Rect, params: dict, state: dict) -> None:
    pygame.draw.rect(screen, (20, 22, 30), sim_rect, border_radius=6)
    for i, (x, y) in enumerate(state["nuclei"]):
        sx, sy = world_to_screen(float(x), float(y), params, sim_rect)
        color = (70, 100, 145) if not state["ionized"][i] else (210, 135, 45)
        screen.fill(color, (sx, sy, 1, 1))
    pygame.draw.line(screen, (235, 235, 235), (sim_rect.left, sim_rect.bottom), (sim_rect.right, sim_rect.bottom), 2)
    src = (params["source_x_ratio"] * params["width"], params["source_y_ratio"] * params["height"])
    ssx, ssy = world_to_screen(src[0], src[1], params, sim_rect)
    pygame.draw.circle(screen, (255, 255, 255), (ssx, ssy), 5)
    for e in state["electrons"]:
        if len(e["trail"]) > 1:
            pts = [world_to_screen(float(p[0]), float(p[1]), params, sim_rect) for p in e["trail"]]
            pygame.draw.lines(screen, (245, 90, 90), False, pts, 1)
        if e["alive"]:
            ex, ey = world_to_screen(float(e["pos"][0]), float(e["pos"][1]), params, sim_rect)
            pygame.draw.circle(screen, (255, 130, 130), (ex, ey), 2)


def draw_histogram(surface: pygame.Surface, rect: pygame.Rect, impacts_x: list[float], width: float, font: object | None) -> None:
    pygame.draw.rect(surface, (14, 16, 24), rect, border_radius=6)
    pygame.draw.rect(surface, (70, 75, 95), rect, 1, border_radius=6)
    if len(impacts_x) < 3:
        draw_text(surface, font, "Histogramme: pas assez d'impacts", rect.left + 10, rect.top + 10)
        return
    data = np.array(impacts_x, dtype=float)
    bins = 46
    hist, edges = np.histogram(data, bins=bins, range=(0.0, width))
    centers = 0.5 * (edges[:-1] + edges[1:])
    hmax = max(1, int(hist.max()))
    mu = float(np.mean(data))
    sigma = float(np.std(data))
    y_gauss = None
    if sigma > 1e-12:
        bin_w = edges[1] - edges[0]
        pdf = (1.0 / (sigma * math.sqrt(2.0 * PI))) * np.exp(-0.5 * ((centers - mu) / sigma) ** 2)
        y_gauss = pdf * len(data) * bin_w
        hmax = max(hmax, int(np.max(y_gauss)))
    bw = rect.width / bins
    for i, v in enumerate(hist):
        x0 = rect.left + int(i * bw)
        x1 = rect.left + int((i + 1) * bw)
        h = int((v / hmax) * (rect.height - 34))
        y0 = rect.bottom - 22 - h
        pygame.draw.rect(surface, (95, 150, 245), (x0 + 1, y0, max(1, x1 - x0 - 2), h))
    if y_gauss is not None:
        pts = []
        for i, gy in enumerate(y_gauss):
            x = rect.left + int((i + 0.5) * bw)
            y = rect.bottom - 22 - int((gy / hmax) * (rect.height - 34))
            pts.append((x, y))
        if len(pts) > 1:
            pygame.draw.lines(surface, (255, 210, 80), False, pts, 2)
    draw_text(surface, font, "Impacts sur y=0", rect.left + 8, rect.top + 8)


def draw_control_panel(surface: pygame.Surface, panel_rect: pygame.Rect, params: dict, state: dict, controls: list[dict], font: object | None, small: object | None) -> None:
    pygame.draw.rect(surface, (18, 20, 28), panel_rect, border_radius=6)
    pygame.draw.rect(surface, (62, 66, 86), panel_rect, 1, border_radius=6)

    draw_text(surface, font, f"Impacts: {state['impacts']}", panel_rect.left + 14, panel_rect.top + 10, (255, 220, 90))

    y = panel_rect.top + 46
    x0 = panel_rect.left + 16
    bar_w = panel_rect.width - 32
    state["slider_rects"] = {}

    for i, c in enumerate(controls):
        selected = i == state["selected_control"]
        draw_text(surface, small, c["label"], x0, y, (235, 235, 235) if selected else (195, 195, 195))
        val = params[c["key"]]
        val_txt = f"{val:.3E}" if abs(val) < 1e-2 or abs(val) > 1e4 else f"{val:.5g}"
        draw_text(surface, small, val_txt, panel_rect.right - 126, y, (255, 210, 120) if selected else (170, 170, 170))

        track = pygame.Rect(x0, y + 16, bar_w, 10)
        fill_t = to_slider(val, c)
        knob_x = track.left + int(fill_t * track.width)
        pygame.draw.rect(surface, (60, 64, 84), track, border_radius=4)
        pygame.draw.rect(surface, (115, 170, 245) if selected else (95, 110, 150), (track.left, track.top, max(1, knob_x - track.left), track.height), border_radius=4)
        pygame.draw.circle(surface, (245, 245, 245), (knob_x, track.centery), 7)

        click_rect = pygame.Rect(track.left, y + 2, track.width, 26)
        state["slider_rects"][c["key"]] = click_rect
        y += 34

    draw_text(surface, small, "Souris: glisser curseur | Fleches gauche/droite: ajuster", x0, panel_rect.bottom - 54)
    draw_text(surface, small, "Haut/bas: changer parametre | R reset gaz | Espace pause", x0, panel_rect.bottom - 36)


def apply_slider_value(params: dict, control: dict, t: float) -> None:
    params[control["key"]] = from_slider(t, control)
    clamp_params(params)


def adjust_selected_control(params: dict, controls: list[dict], idx: int, direction: int) -> bool:
    c = controls[idx]
    val = params[c["key"]]
    span = (c["max"] - c["min"])
    if c["scale"] == "log":
        factor = 1.06 if direction > 0 else (1.0 / 1.06)
        val *= factor
    else:
        val += direction * span * 0.01
    params[c["key"]] = float(np.clip(val, c["min"], c["max"]))
    return bool(c["reset"])


def run_app() -> None:
    pygame.init()
    pygame.display.set_caption("Avalanche electronique 2D - Controle interactif")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()
    font, small = load_fonts()

    params = default_params()
    controls = get_controls()
    state = reset_state(params)
    sim_rect, panel_rect, hist_rect = build_rects()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    state["paused"] = not state["paused"]
                elif event.key == pygame.K_r:
                    state = reset_state(params)
                elif event.key == pygame.K_c:
                    state["impacts_x"] = []
                    state["impacts"] = 0
                elif event.key == pygame.K_UP:
                    state["selected_control"] = (state["selected_control"] - 1) % len(controls)
                elif event.key == pygame.K_DOWN:
                    state["selected_control"] = (state["selected_control"] + 1) % len(controls)
                elif event.key == pygame.K_LEFT:
                    if adjust_selected_control(params, controls, state["selected_control"], -1):
                        state = reset_state(params)
                elif event.key == pygame.K_RIGHT:
                    if adjust_selected_control(params, controls, state["selected_control"], +1):
                        state = reset_state(params)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    for i, c in enumerate(controls):
                        r = state["slider_rects"].get(c["key"])
                        if r is not None and r.collidepoint(mx, my):
                            state["selected_control"] = i
                            state["dragging_key"] = c["key"]
                            t = (mx - r.left) / r.width
                            apply_slider_value(params, c, t)
                            if c["reset"]:
                                state = reset_state(params)
                            break
                elif event.button == 4:
                    if adjust_selected_control(params, controls, state["selected_control"], +1):
                        state = reset_state(params)
                elif event.button == 5:
                    if adjust_selected_control(params, controls, state["selected_control"], -1):
                        state = reset_state(params)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                state["dragging_key"] = None

            elif event.type == pygame.MOUSEMOTION and state["dragging_key"] is not None:
                key = state["dragging_key"]
                c = next(cc for cc in controls if cc["key"] == key)
                r = state["slider_rects"].get(key)
                if r is not None:
                    t = (event.pos[0] - r.left) / r.width
                    apply_slider_value(params, c, t)
                    if c["reset"]:
                        state = reset_state(params)

        step_physics(state, params)

        screen.fill((9, 11, 16))
        draw_simulation(screen, sim_rect, params, state)
        draw_control_panel(screen, panel_rect, params, state, controls, font, small)
        draw_histogram(screen, hist_rect, state["impacts_x"], params["width"], small)

        if len(state["impacts_x"]) >= 3:
            mu = float(np.mean(state["impacts_x"]))
            sigma = float(np.std(state["impacts_x"]))
            draw_text(screen, small, f"fit gauss: mu={mu*1e3:.3f} mm, sigma={sigma*1e3:.3f} mm", panel_rect.left + 16, hist_rect.top - 20, (240, 210, 120))

        if state["paused"]:
            draw_text(screen, font, "PAUSE", sim_rect.left + 12, sim_rect.top + 10, (255, 180, 80))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()
