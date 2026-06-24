import numpy as np
from scipy.special import genlaguerre
from math import sqrt, pi
from typing import List, Tuple



"""

- This function generates the Laguerre-Gaussian beam.

"""
def generate_lg(l: int, p: int, w0: float, R: np.ndarray, PHI: np.ndarray, dx: float) -> np.ndarray:
    """Normalized LG_p^l field. integral |E|^2 dx dy = 1."""
    al = abs(l)
    Lp = genlaguerre(p, al)

    r_norm = R * sqrt(2) / w0
    # Normalization factor for LG_p^l
    # ||E||^2 = 1 requires norm^2 * integral = 1
    from math import factorial
    norm = sqrt(2 * factorial(p) / (pi * factorial(p + al))) / w0

    E = (norm
         * r_norm**al
         * np.exp(-R**2 / w0**2)
         * Lp(r_norm**2)
         * np.exp(1j * l * PHI))
    return E.astype(np.complex128)


def generate_multiplexed_lg(l_modes: List[int], p: int, w0: float,
                             R: np.ndarray, PHI: np.ndarray, dx: float) -> np.ndarray:
    """
    Sum multiple LG modes with equal power. Total power normalized to 1.

    Parameters
    ----------
    l_modes : list of azimuthal indices, e.g. [1, 3, 5]
    p       : radial index (same for all modes)
    w0      : beam waist [m]
    """
    E = np.zeros_like(R, dtype=np.complex128)
    for l in l_modes:
        E += generate_lg(l, p, w0, R, PHI, dx)
    # Normalize so total power = 1
    power = np.sum(np.abs(E)**2) * dx**2
    return E / np.sqrt(power)


def precompute_basis(modes: List[Tuple[int, int]], w0: float,
                     R: np.ndarray, PHI: np.ndarray, dx: float) -> List[np.ndarray]:
    """Return list of LG mode fields in the same order as modes."""
    return [generate_lg(l, p, w0, R, PHI, dx) for p, l in modes]
