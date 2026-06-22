"""
Centralised configuration for the LG-beam fog / turbulence simulation.
All physical quantities are in SI units throughout.
"""

from dataclasses import dataclass, field
from math import pi
from typing import List


# ── Grid ─────────────────────────────────────────────────────────────────────

@dataclass
class GridConfig:
    N:          int   = 128
    L_m:        float = 32e-3      # physical extent [m]  — must contain diverged beam
    wavelength: float = 1550e-9    # [m]  telecom C-band
    dx:         float = field(init=False)
    k:          float = field(init=False)

    def __post_init__(self):
        self.dx = self.L_m / self.N          # pixel pitch [m]
        self.k  = 2 * pi / self.wavelength   # wavenumber [rad/m]


# ── Beam ─────────────────────────────────────────────────────────────────────

@dataclass
class BeamConfig:
    w0:   float = 1.5e-3   # beam waist [m]
    l_in: int   = 1        # input OAM topological charge
    p_in: int   = 0        # input radial index


# ── Fog + Turbulence ──────────────────────────────────────────────────────────

@dataclass
class FogConfig:
    # Mie / fog parameters
    r_eff:    float = 10e-6    # droplet radius [m]
    nd_min:   float = 10e6    # minimum droplet density [m^-3]  (~628 m visibility)
    nd_max:   float = 150e6   # maximum droplet density [m^-3]  (~42 m visibility)

    # Propagation geometry
    path_m:   float = 20.0    # total path length [m]
    n_screens: int  = 10      # number of thin screens

    # Kolmogorov turbulence (Option C)
    r0_min:   float = 3e-4    # Fried parameter minimum [m]  — strong turbulence
    r0_max:   float = 5e-3    # Fried parameter maximum [m]  — weak turbulence
    L0:       float = 10.0    # outer scale [m]

    dz:       float = field(init=False)   # screen spacing [m]

    def __post_init__(self):
        self.dz = self.path_m / self.n_screens


# ── OAM Basis ────────────────────────────────────────────────────────────────

@dataclass
class BasisConfig:
    l_max:   int        = 5
    p_modes: List[int]  = field(default_factory=lambda: [0])
    modes:   list       = field(init=False)
    n_modes: int        = field(init=False)

    def __post_init__(self):
        self.modes   = [(p, l)
                        for p in self.p_modes
                        for l in range(-self.l_max, self.l_max + 1)]
        self.n_modes = len(self.modes)


# ── Dataset ───────────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    n_samples: int = 1800
    save_dir:  str = "data/"


# ── Module-level singletons (imported everywhere) ────────────────────────────

grid    = GridConfig()
beam    = BeamConfig()
fog_cfg = FogConfig()
basis   = BasisConfig()
dataset = DatasetConfig()
