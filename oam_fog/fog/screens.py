import numpy as np
from math import sqrt
from scipy.ndimage import gaussian_filter
from physics.propagation import asm_step

_BLUR_PX = 10.0  # spatial correlation length [pixels] (~2.5mm >> w0=1.5mm keeps radial structure intact)


def _correlated_noise(rng: np.random.Generator, N: int, blur_px: float) -> np.ndarray:
    """Unit-variance spatially-correlated Gaussian noise via blur + rescale."""
    raw = rng.standard_normal((N, N))
    if blur_px > 0:
        smoothed = gaussian_filter(raw, sigma=blur_px)
        std = smoothed.std()
        return smoothed / std if std > 0 else smoothed
    return raw


def generate_screen(fog_params: dict, N: int, rng: np.random.Generator) -> np.ndarray:
    """Complex NxN transmittance mask T(x,y) = A(x,y) * exp(i*phi(x,y)).
    Screens are spatially correlated at scale _BLUR_PX to keep scattered power
    within the OAM basis (avoids coupling to high-frequency modes outside l=-5..5)."""
    T_amp     = fog_params['T_amp']
    sigma_A   = fog_params['sigma_A']
    sigma_phi = fog_params['sigma_phi']

    mean_log = np.log(T_amp) - sigma_A**2 / 2
    log_amp  = mean_log + sigma_A * _correlated_noise(rng, N, _BLUR_PX)
    log_amp  = np.clip(log_amp, -10, 1)
    A        = np.exp(log_amp)

    phi = sigma_phi * _correlated_noise(rng, N, _BLUR_PX)
    return (A * np.exp(1j * phi)).astype(np.complex128)

def simulate(E_in: np.ndarray, fog_params: dict, H_step: np.ndarray,
             n_screens: int, rng: np.random.Generator) -> np.ndarray:
    """Split-step fog propagation through n_screens phase screens."""
    N = E_in.shape[0]
    E = E_in.copy()
    for _ in range(n_screens):
        E = E * generate_screen(fog_params, N, rng)
        E = asm_step(E, H_step)
    return E
