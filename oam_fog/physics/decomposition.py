import numpy as np
from typing import List


def oam_spectrum(E_out: np.ndarray, basis: List[np.ndarray], dx: float) -> np.ndarray:
    """Project E_out onto OAM basis and return normalized power spectrum."""
    powers = np.array([abs(np.sum(E_out * np.conj(mode)) * dx**2)**2 for mode in basis],
                      dtype=np.float64)
    total = powers.sum()
    return powers / total if total > 1e-30 else powers


def basis_capture(E_out: np.ndarray, basis: List[np.ndarray], dx: float) -> float:
    """Fraction of output power captured by the basis. Diagnostic: should be > 0.85."""
    total_output_power = np.sum(np.abs(E_out)**2) * dx**2
    captured = sum(abs(np.sum(E_out * np.conj(mode)) * dx**2)**2 for mode in basis)
    return float(captured / total_output_power) if total_output_power > 1e-30 else 0.0
