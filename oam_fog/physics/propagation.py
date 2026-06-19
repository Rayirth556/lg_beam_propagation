import numpy as np
from math import pi


def asm_transfer(z_m: float, grid_config) -> np.ndarray:
    """Angular spectrum transfer function for free-space propagation distance z_m."""
    N = grid_config.N
    dx = grid_config.dx
    k = grid_config.k

    fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    Fx, Fy = np.meshgrid(fx, fx, indexing='ij')

    kz2 = k**2 - (2 * pi * Fx)**2 - (2 * pi * Fy)**2
    kz = np.sqrt(np.maximum(kz2, 0.0))
    H = np.where(kz2 > 0, np.exp(1j * kz * z_m), 0.0)
    return H.astype(np.complex128)


def asm_step(E: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Propagate field E using precomputed angular spectrum transfer function H."""
    Efft = np.fft.fftshift(np.fft.fft2(E))
    return np.fft.ifft2(np.fft.ifftshift(Efft * H))
