import numpy as np
from scipy.special import genlaguerre
from math import sqrt, pi
from typing import List, Tuple


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


def precompute_basis(modes: List[Tuple[int, int]], w0: float,
                     R: np.ndarray, PHI: np.ndarray, dx: float) -> List[np.ndarray]:
    """Return list of LG mode fields in the same order as modes."""
    return [generate_lg(l, p, w0, R, PHI, dx) for p, l in modes]
