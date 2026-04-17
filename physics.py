"""
physics.py – Townsend avalanche simulation
Geometry: y=0 = anode (top), y=HEIGHT = cathode (bottom)
Electric field points downward (from anode + to cathode -)
Electrons drift DOWN (toward cathode), ions drift UP (toward anode)
"""

import math, random
from dataclasses import dataclass, field
from typing import List

# ── Physical constants (simulation units) ────────────────────────────────────
IONISATION_ENERGY_EV  = 15.7   # Argon first ionisation potential (eV)
EXCITATION_ENERGY_EV  = 11.5   # First excitation level (eV)

GAS_DENSITY   = 0.10
ELECTRON_MASS = 1.0
ION_MASS      = 73600

DEFAULT_VOLTAGE  = 300
GAP_HEIGHT_MM    = 10.0

ELECTRON_CHARGE  = 1.0
MAX_PARTICLES    = 8000
SECONDARY_PROB   = 0.35
INJECT_PERIOD    = 5000 #TODO

PHOTON_IONISE_PROB = 0.05

# Recombination: after this many steps an ionised atom becomes neutral again.
# Set to 0 to disable.
RECOMBINATION_STEPS = 80


@dataclass
class Particle:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    kind: str = 'electron'   # 'electron' | 'ion'
    alive: bool = True
    trail: list = field(default_factory=list)
    MAX_TRAIL: int = 18

    def push_trail(self):
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.MAX_TRAIL:
            self.trail.pop(0)


@dataclass
class Atom:
    x: float
    y: float
    ionised: bool = False
    excited: bool = False
    deexcite_timer: float = 0.0
    recombine_timer: int = 0       # steps until neutralisation
    recombine_flash: float = 0.0   # visual glow countdown (0-1)


class AvalancheSimulation:
    def __init__(self,
                 width: int = 800, height: int = 600,
                 n_atoms: int = 140,
                 voltage: float = DEFAULT_VOLTAGE,
                 gas: str = 'Ar',
                 recombination_steps: int = RECOMBINATION_STEPS,
                 seed: int = 42):

        self.W = width
        self.H = height
        self.voltage = voltage
        self.gas = gas
        self.recombination_steps = recombination_steps
        random.seed(seed)

        self.E_field = voltage / height

        self.atoms: List[Atom] = []
        self._place_atoms(n_atoms)

        self.particles: List[Particle] = []
        self.events: List[dict] = []

        self.snapshots: List[dict] = []
        self.dt = 0.016

        self._spawn_electron(self.W / 2, 8)

    # ── Geometry ──────────────────────────────────────────────────────────────

    def _place_atoms(self, n: int):
        margin = 30
        cols = int(math.sqrt(n * self.W / self.H)) + 1
        rows = int(n / cols) + 1
        dx = (self.W - 2 * margin) / cols
        dy = (self.H - 2 * margin) / rows
        placed = 0
        for r in range(rows):
            for c in range(cols):
                if placed >= n:
                    break
                jx = random.uniform(-dx * 0.3, dx * 0.3)
                jy = random.uniform(-dy * 0.3, dy * 0.3)
                x = margin + c * dx + dx / 2 + jx
                y = margin + r * dy + dy / 2 + jy
                self.atoms.append(Atom(x, y))
                placed += 1

    def _spawn_electron(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        if len(self.particles) < MAX_PARTICLES:
            p = Particle(x=x, y=y,
                         vx=vx + random.uniform(-0.3, 0.3),
                         vy=vy, kind='electron')
            self.particles.append(p)

    def _ke_ev(self, p: Particle) -> float:
        mass = ELECTRON_MASS if p.kind == 'electron' else ION_MASS
        return 0.5 * mass * (p.vx ** 2 + p.vy ** 2)

    # ── Core step ─────────────────────────────────────────────────────────────

    def step(self):
        for p in self.particles:
            if not p.alive:
                continue

            p.push_trail()

            if p.kind == 'electron':
                accel = self.E_field * ELECTRON_CHARGE / ELECTRON_MASS
                p.vy += accel
                p.vx += random.gauss(0, 0.05)
            elif p.kind == 'ion':
                accel = self.E_field * ELECTRON_CHARGE / ION_MASS * 400
                p.vy -= accel
                p.vx += random.gauss(0, 0.01)

            p.x += p.vx
            p.y += p.vy

            if p.x < 0 or p.x > self.W:
                p.vx *= -0.5
                p.x = max(0, min(self.W, p.x))

            if p.kind == 'electron' and p.y >= self.H:
                p.alive = False
                self.events.append({'type': 'detected', 'x': p.x, 'y': self.H})
                if random.random() < SECONDARY_PROB and len(self.particles) < MAX_PARTICLES:
                    self._spawn_electron(p.x, self.H - 5, vy=-0.5)
                continue

            if p.kind == 'ion' and p.y <= 0:
                p.alive = False
                continue

            if p.kind == 'ion' and p.y >= self.H:
                p.alive = False
                if random.random() < SECONDARY_PROB and len(self.particles) < MAX_PARTICLES:
                    self._spawn_electron(p.x, self.H - 5, vy=-1.0)
                continue

            if p.y < -50 or p.y > self.H + 50:
                p.alive = False
                continue

            if p.kind == 'electron':
                self._check_collisions(p)

        # ── Atom timers ───────────────────────────────────────────────────────
        for atom in self.atoms:

            if atom.excited:
                atom.deexcite_timer -= self.dt
                if atom.deexcite_timer <= 0:
                    atom.excited = False
                    if random.random() < PHOTON_IONISE_PROB:
                        target = self._find_nearby_atom(atom, radius=80)
                        if target and not target.ionised:
                            target.ionised = True
                            target.recombine_timer = (self.recombination_steps
                                                      + random.randint(-10, 10))
                            self._spawn_electron(target.x, target.y - 2, vy=0.5)

            # Recombination countdown
            if atom.ionised and self.recombination_steps > 0:
                atom.recombine_timer -= 1
                if atom.recombine_timer <= 0:
                    atom.ionised = False
                    atom.excited = False
                    atom.recombine_flash = 1.0
                    self.events.append({'type': 'recombined',
                                        'x': atom.x, 'y': atom.y})

            if atom.recombine_flash > 0:
                atom.recombine_flash = max(0.0, atom.recombine_flash - 0.08)

        self.particles = [p for p in self.particles if p.alive]

    def _check_collisions(self, electron: Particle):
        collision_radius = 10 + GAS_DENSITY * 20

        for atom in self.atoms:
            if atom.ionised:
                continue
            dx = electron.x - atom.x
            dy = electron.y - atom.y
            if math.hypot(dx, dy) > collision_radius:
                continue

            ke = self._ke_ev(electron)

            if ke >= IONISATION_ENERGY_EV:
                atom.ionised = True
                atom.recombine_timer = (self.recombination_steps
                                        + random.randint(-15, 15))
                atom.recombine_flash = 0.0
                self.events.append({'type': 'ionisation',
                                    'x': atom.x, 'y': atom.y, 'ke': ke})

                fraction_kept = random.uniform(0.3, 0.6)
                energy_left   = ke - IONISATION_ENERGY_EV
                v_inc = math.sqrt(max(0, 2 * energy_left * fraction_kept / ELECTRON_MASS))
                v_sec = math.sqrt(max(0, 2 * energy_left * (1 - fraction_kept) / ELECTRON_MASS))

                angle = random.uniform(-math.pi / 4, math.pi / 4)
                electron.vx = v_inc * math.sin(angle)
                electron.vy = v_inc * math.cos(angle) + 0.3

                angle2 = random.uniform(-math.pi / 3, math.pi / 3)
                self._spawn_electron(atom.x, atom.y,
                                     vx=v_sec * math.sin(angle2),
                                     vy=v_sec * math.cos(angle2) + 0.3)

                ion = Particle(x=atom.x + random.uniform(-2, 2),
                               y=atom.y,
                               vx=random.uniform(-0.05, 0.05),
                               vy=-0.02,
                               kind='ion')
                self.particles.append(ion)

            elif ke >= EXCITATION_ENERGY_EV:
                atom.excited = True
                atom.deexcite_timer = random.uniform(0.01, 0.15)
                energy_left = ke - EXCITATION_ENERGY_EV
                v = math.sqrt(max(0, 2 * energy_left / ELECTRON_MASS))
                angle = random.uniform(-math.pi / 6, math.pi / 6)
                electron.vx = v * math.sin(angle)
                electron.vy = max(0.1, v * math.cos(angle))

            else:
                angle = random.uniform(-math.pi / 8, math.pi / 8)
                spd = math.hypot(electron.vx, electron.vy)
                electron.vx = spd * math.sin(angle)
                electron.vy = max(0, spd * math.cos(angle))

            break

    def _find_nearby_atom(self, source: Atom, radius: float):
        candidates = [a for a in self.atoms
                      if not a.ionised and not a.excited
                      and math.hypot(a.x - source.x, a.y - source.y) < radius]
        return random.choice(candidates) if candidates else None

    def take_snapshot(self) -> dict:
        return {
            # atoms now include recombine_flash for renderer
            'atoms': [
                (a.x, a.y, a.ionised, a.excited, a.recombine_flash)
                for a in self.atoms
            ],
            'electrons': [(p.x, p.y, list(p.trail))
                          for p in self.particles if p.kind == 'electron'],
            'ions':      [(p.x, p.y, list(p.trail))
                          for p in self.particles if p.kind == 'ion'],
            'events':    list(self.events),
            'n_ionised':   sum(1 for a in self.atoms if a.ionised),
            'n_electrons': sum(1 for p in self.particles if p.kind == 'electron'),
        }

    def run_full(self, n_steps: int = 400):
        self.snapshots = []
        for step_n in range(n_steps):
            self.snapshots.append(self.take_snapshot())
            if step_n>0 and step_n % INJECT_PERIOD == 0 and len(self.particles) < 20:
                x0 = random.uniform(self.W * 0.2, self.W * 0.8)
                self._spawn_electron(x0, random.uniform(5, 20))
            self.step()
        self.snapshots.append(self.take_snapshot())
        return self.snapshots
