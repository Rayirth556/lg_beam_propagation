from dataclasses import dataclass, field
from math import pi
from typing import List, Tuple

"""
grid = 2d spacial sampling of a physical plane

l = azimuthal quantum number (controls the helical/spiral phase structure of the beam)
- l = 0(plain gaussian, no spiral)
- l = 1(single helix)
- l = -1(helix in opposite directions)
p = radial index(controls how many concentric rings the beam has)
- p = 0 → one bright ring (most common)
- p = 1 → two concentric rings
- p = 2 → three rings
"""

@dataclass
class GridConfig:
    N: int = 128 ## grid is 128x128 pixels
    L_m: float = 32e-3 ## physical size of grid(field of view)
    wavelength: float = 1550e-9 ## laser wavelength

    @property
    def dx(self) -> float:    ## Computes the physical size of one pixel.
        return self.L_m / self.N


    ## k is the optical wave number
    @property
    def k(self) -> float:     ## k = 2*pi/wavelength
        return 2 * pi / self.wavelength


@dataclass
class BeamConfig:
    w0: float = 1.5e-3          ## beam waist radius [m]
    l_in: int = 1               ## single-mode input (used if l_modes is empty)
    p_in: int = 0               ## radial index of input beam
    l_modes: List[int] = field(default_factory=lambda: [1, 3, 5])  ## multiplexed OAM modes




## This class defines the fog medium through which the LG beam propagates
@dataclass
class FogConfig:
    r_eff: float = 10e-6   ## effective droplet radius (10 µm)
    nd_min: float = 10e6   ## minimum droplet number density [m^-3]
    nd_max: float = 20e6   ## maximum droplet number density [m^-3]
                           ##   validated: keeps mean capture ≥ 0.90 at 250m
    r0_min: float = 2e-2   ## Fried coherence length min [m] = 20mm — validated at 250m
    r0_max: float = 10e-2  ## Fried coherence length max [m] = 100mm
    path_m: float = 250.0  ## propagation distance [m] — validated max with fog + turbulence
    n_screens: int = 3     ## phase screens: dz = 250/3 ≈ 83m per screen

    @property
    def dz(self) -> float:
        return self.path_m / self.n_screens   ## Distance between screens (20/10 = 2m)

"""
- onfiguration is for the OAM decomposition basis
An OAM mode is a light beam whose phase twists around its center

"""
@dataclass
class BasisConfig:
    l_max: int = 8  ## l∈[−8,8], 17 different OAM values
    p_modes: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])  ## radial modes — fog scatters power into p>0

    @property
    def modes(self) -> List[Tuple[int, int]]:        ## This generates all basis modes.
        return [(p, l) for p in self.p_modes for l in range(-self.l_max, self.l_max + 1)]

    @property
    def n_modes(self) -> int:          ## ## Counts how many basis functions exist.
        return len(self.modes)


@dataclass
class DatasetConfig:
    n_samples: int = 5000
    save_dir: str = "data/"


grid = GridConfig()
beam = BeamConfig()
fog_cfg = FogConfig()
basis = BasisConfig()
dataset = DatasetConfig()
