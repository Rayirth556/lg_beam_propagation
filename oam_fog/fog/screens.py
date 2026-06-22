"""
Fog screen generation.

Each screen = Beer-Lambert amplitude (from Mie) × Kolmogorov turbulence phase.

Amplitude A(x,y):
    Log-normal distributed, mean = T_amp (Beer-Lambert), std = sigma_A.
    Spatially uncorrelated (r_eff = 10 µm << dx = 250 µm).
    Source of amplitude attenuation — does NOT cause strong OAM mixing on its own.

Phase phi(x,y):
    Von Karman / Kolmogorov turbulence phase screen.
    Parameterised by Fried coherence radius r0 [m].
    This IS the primary driver of OAM mode mixing.
    For w0/r0 ~ 1–5, significant power transfers between OAM modes.

Reference: Schmidt (2010) 'Numerical Simulation of Optical Wave Propagation',
           SPIE Press — Chapter 9 (phase screen generation).
"""

import numpy as np
from physics.propagation import asm_step


# ─────────────────────────────────────────────────────────────────────────────
#  Kolmogorov turbulence phase screen
# ─────────────────────────────────────────────────────────────────────────────

def kolmogorov_screen(N: int, dx: float, r0: float,
                      rng, L0: float = 10.0) -> np.ndarray:
    """
    Generate a von Karman turbulence phase screen.

    PSD:  Φ_φ(f) = 0.023 · r0^(-5/3) · (f² + f0²)^(-11/6)   [rad² · m²]

    Generation (Schmidt 2010):
        del_f = 1 / (N · dx)
        phi   = Re{ ifft2(sqrt(Φ_φ) · cn) } · N² · del_f

    Physical sanity check:
        Structure function  D_φ(r0) ≈ 6.88 rad²

    Parameters
    ----------
    N   : grid size (N × N)
    dx  : pixel pitch [m]
    r0  : Fried coherence parameter [m]  — smaller = stronger turbulence
    rng : numpy random Generator
    L0  : outer scale [m]  (prevents DC divergence)

    Returns
    -------
    phi : real phase screen [rad], shape (N, N)
    """
    del_f = 1.0 / (N * dx)                          # frequency resolution [cycles/m]

    # Spatial frequency grid in natural FFT order [cycles/m]
    fx       = np.fft.fftfreq(N, d=dx)
    Fx, Fy   = np.meshgrid(fx, fx, indexing='ij')
    f2       = Fx**2 + Fy**2
    f0       = 1.0 / L0                              # outer-scale frequency [1/m]

    # Von Karman PSD [rad² · m²]
    PSD         = 0.023 * r0**(-5 / 3) * (f2 + f0**2)**(-11 / 6)
    PSD[0, 0]   = 0.0                                # remove piston (DC)

    # Complex Gaussian noise — unit variance per component
    cn  = (rng.standard_normal((N, N)) +
           1j * rng.standard_normal((N, N))) / np.sqrt(2)

    # Phase screen via IFFT with correct normalisation
    phi = np.real(np.fft.ifft2(np.sqrt(PSD) * cn)) * (N ** 2) * del_f

    return phi


# ─────────────────────────────────────────────────────────────────────────────
#  Combined fog screen
# ─────────────────────────────────────────────────────────────────────────────

def generate_screen(fog_params: dict, N: int, dx: float, r0: float,
                    rng, L0: float = 10.0) -> np.ndarray:
    """
    Generate one complex N×N transmittance mask:  T(x,y) = A(x,y) · exp(iφ(x,y))

    Amplitude  A(x,y) : Beer-Lambert mean + log-normal spatial fluctuation (Mie)
    Phase    phi(x,y) : Kolmogorov turbulence via von Karman PSD

    Parameters
    ----------
    fog_params : dict returned by MieModel.fog_params()
    N          : grid size
    dx         : pixel pitch [m]
    r0         : Fried coherence parameter [m]
    rng        : numpy random Generator
    L0         : turbulence outer scale [m]

    Returns
    -------
    T : complex screen, shape (N, N), dtype complex128
    """
    T_amp   = fog_params['T_amp']
    sigma_A = fog_params['sigma_A']

    # Amplitude: log-normal with bias-corrected mean = T_amp
    mean_log = np.log(max(T_amp, 1e-30)) - sigma_A**2 / 2.0
    log_amp  = rng.normal(mean_log, sigma_A, (N, N))
    A        = np.exp(np.clip(log_amp, -10.0, 1.0))

    # Phase: Kolmogorov turbulence
    phi = kolmogorov_screen(N, dx, r0, rng, L0)

    return (A * np.exp(1j * phi)).astype(np.complex128)


# ─────────────────────────────────────────────────────────────────────────────
#  Split-step propagation loop
# ─────────────────────────────────────────────────────────────────────────────

def simulate(E_in: np.ndarray,
             fog_params: dict,
             H_step: np.ndarray,
             n_screens: int,
             r0: float,
             dx: float,
             rng,
             L0: float = 10.0) -> np.ndarray:
    """
    Propagate E_in through n_screens of fog + turbulence.

    Each step:
        1. Apply screen: E ← E · T(x,y)   [amplitude attenuation + turbulence phase]
        2. Free-space:   E ← ASM_step(E, H_step)

    Parameters
    ----------
    E_in      : input complex field, shape (N, N)
    fog_params: dict from MieModel.fog_params() — provides T_amp, sigma_A
    H_step    : precomputed ASM transfer function for one DZ step
    n_screens : number of thin screens
    r0        : Fried coherence parameter [m]
    dx        : pixel pitch [m]
    rng       : numpy random Generator
    L0        : turbulence outer scale [m]

    Returns
    -------
    E_out : complex field at z = path_m, shape (N, N)
    """
    E = E_in.copy()
    N = E.shape[0]

    for _ in range(n_screens):
        screen = generate_screen(fog_params, N, dx, r0, rng, L0)
        E      = E * screen
        E      = asm_step(E, H_step)

    return E
