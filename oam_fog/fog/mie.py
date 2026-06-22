import miepython
import numpy as np
from math import pi, sqrt


class MieModel:
    def __init__(self, r_eff: float, wavelength: float, m_water: complex = 1.324 + 1.66e-7j):
        self.r_eff = r_eff
        self.wavelength = wavelength
        self.m_water = m_water

        diameter = 2 * r_eff
        Q_ext, Q_sca, Q_back, g = miepython.efficiencies(m_water, diameter, wavelength)

        self.Q_ext = Q_ext
        self.Q_sca = Q_sca
        self.Q_back = Q_back
        self.g = g
        self.sigma_sca = Q_sca * pi * r_eff**2
        self.sigma_ext = Q_ext * pi * r_eff**2
        self.x = 2 * pi * r_eff / wavelength

    def fog_params(self, N_d: float, dz: float) -> dict:
        mu_s   = N_d * self.sigma_sca
        mu_ext = N_d * self.sigma_ext
        vis_m  = 3.91 / mu_ext if mu_ext > 0 else float('inf')
        T_amp  = np.exp(-mu_ext * dz / 2)

        k   = 2 * pi / self.wavelength
        n_r = self.m_water.real

        sigma_A   = sqrt(mu_s * (1 - self.g) * dz) * 0.5
        sigma_phi = k * (n_r - 1) * self.r_eff * sqrt(2 * pi * N_d * dz * self.r_eff**2) \
                    * sqrt(1 - self.g)                                                        # ← * sqrt(1-g) added

        return {
            'mu_s':      mu_s,
            'mu_ext':    mu_ext,
            'vis_m':     vis_m,
            'T_amp':     T_amp,
            'sigma_A':   sigma_A,
            'sigma_phi': 0.15,
        }

    def summary(self, fog_config):
        nd_values = np.linspace(fog_config.nd_min, fog_config.nd_max, 5)
        print(f"\n{'='*65}")
        print(f"Mie scattering summary  (r_eff={self.r_eff*1e6:.1f} um, "
              f"wl={self.wavelength*1e9:.0f} nm, x={self.x:.3f})")
        print(f"  Q_ext={self.Q_ext:.4f}  Q_sca={self.Q_sca:.4f}  g={self.g:.4f}")
        print(f"  s_ext={self.sigma_ext:.3e} m^2   s_sca={self.sigma_sca:.3e} m^2")
        print(f"{'='*65}")
        print(f"{'N_d [m^-3]':>14}  {'vis [m]':>8}  {'T_amp/screen':>13}  {'sA':>8}  {'s_phi [rad]':>11}")
        print(f"{'-'*65}")
        for nd in nd_values:
            p = self.fog_params(nd, fog_config.dz)
            print(f"{nd:14.3e}  {p['vis_m']:8.1f}  {p['T_amp']:13.6f}  "
                  f"{p['sigma_A']:8.5f}  {p['sigma_phi']:10.5f}")
        print(f"{'='*65}\n")
