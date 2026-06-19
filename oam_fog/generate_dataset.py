"""Generate dataset of (output intensity image, OAM spectrum) pairs."""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import grid, beam, fog_cfg, basis, dataset
from physics.beam import generate_lg, precompute_basis
from physics.propagation import asm_transfer
from physics.decomposition import oam_spectrum, basis_capture
from fog.mie import MieModel
from fog.screens import simulate


def main():
    os.makedirs(dataset.save_dir, exist_ok=True)

    # 1. Build grid arrays
    x = (np.arange(grid.N) - grid.N // 2) * grid.dx
    X, Y = np.meshgrid(x, x, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X)

    # 2. Mie model
    mie = MieModel(fog_cfg.r_eff, grid.wavelength)
    mie.summary(fog_cfg)

    # 3. Precompute OAM basis
    print("Precomputing OAM basis...")
    basis_fields = precompute_basis(basis.modes, beam.w0, R, PHI, grid.dx)
    print(f"  {basis.n_modes} modes: l in [{-basis.l_max}, {basis.l_max}], p in {basis.p_modes}")

    # 4. Input beam
    E_in = generate_lg(beam.l_in, beam.p_in, beam.w0, R, PHI, grid.dx)

    # 5. Precompute ASM transfer function
    H_step = asm_transfer(fog_cfg.dz, grid)
    H_back = asm_transfer(-fog_cfg.path_m, grid)
    # 6. Sample N_d log-uniformly
    rng_master = np.random.default_rng(42)
    nd_samples = np.exp(
        rng_master.uniform(np.log(fog_cfg.nd_min), np.log(fog_cfg.nd_max), dataset.n_samples)
    )
    seeds = rng_master.integers(0, 2**31, size=dataset.n_samples)

    # Storage
    intensities = np.zeros((dataset.n_samples, grid.N, grid.N), dtype=np.float32)
    spectra = np.zeros((dataset.n_samples, basis.n_modes), dtype=np.float32)
    visibilities = np.zeros(dataset.n_samples, dtype=np.float32)
    captures = np.zeros(dataset.n_samples, dtype=np.float32)

    print(f"\nGenerating {dataset.n_samples} samples...")
    t0 = time.time()

    for i in range(dataset.n_samples):
        nd = nd_samples[i]
        fp = mie.fog_params(nd, fog_cfg.dz)
        rng_i = np.random.default_rng(int(seeds[i]))
        E_out = simulate(E_in, fp, H_step, fog_cfg.n_screens, rng_i)

        I = np.abs(E_out)**2                                        # ← keep E_out here
        I_max = I.max()
        intensities[i] = (I / I_max if I_max > 0 else I).astype(np.float32)

        E_dec = asm_step(E_out, H_back)                            # ← ADD this line

        spec = oam_spectrum(E_dec, basis_fields, grid.dx)          # ← E_out → E_dec
        spectra[i] = spec.astype(np.float32)

        cap = basis_capture(E_dec, basis_fields, grid.dx)          # ← E_out → E_dec
        captures[i] = float(cap)
        visibilities[i] = float(fp['vis_m'])

        if (i + 1) % 300 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (dataset.n_samples - i - 1) / rate
            print(f"  [{i+1:4d}/{dataset.n_samples}]  "
                  f"elapsed={elapsed:.1f}s  eta={eta:.1f}s  "
                  f"capture={cap:.3f}  vis={fp['vis_m']:.1f}m")

    # 9. Save arrays
    np.save(os.path.join(dataset.save_dir, "intensities.npy"), intensities)
    np.save(os.path.join(dataset.save_dir, "spectra.npy"), spectra)
    np.save(os.path.join(dataset.save_dir, "visibilities.npy"), visibilities)

    metadata = {
        "grid": {"N": grid.N, "L_m": grid.L_m, "dx": grid.dx,
                 "wavelength": grid.wavelength, "k": grid.k},
        "beam": {"w0": beam.w0, "l_in": beam.l_in, "p_in": beam.p_in},
        "fog": {"r_eff": fog_cfg.r_eff, "nd_min": fog_cfg.nd_min,
                "nd_max": fog_cfg.nd_max, "path_m": fog_cfg.path_m,
                "n_screens": fog_cfg.n_screens, "dz": fog_cfg.dz},
        "basis": {"l_max": basis.l_max, "p_modes": basis.p_modes,
                  "modes": basis.modes, "n_modes": basis.n_modes},
        "dataset": {"n_samples": dataset.n_samples, "save_dir": dataset.save_dir},
    }
    with open(os.path.join(dataset.save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # 10. Final summary
    total_time = time.time() - t0
    mean_capture = captures.mean()
    min_capture = captures.min()
    min_cap_idx = captures.argmin()
    mean_spec = spectra.mean(axis=0)
    top5_idx = np.argsort(mean_spec)[::-1][:5]

    print(f"\n{'='*60}")
    print(f"Dataset complete in {total_time:.1f}s")
    print(f"  intensities: {intensities.shape}  float32")
    print(f"  spectra:     {spectra.shape}  float32")
    print(f"  visibilities:{visibilities.shape}  float32")
    print(f"\nBasis capture:")
    print(f"  mean = {mean_capture:.4f}")
    print(f"  min  = {min_capture:.4f}  (sample {min_cap_idx}, vis={visibilities[min_cap_idx]:.1f}m)")
    if min_capture < 0.80:
        print("  WARNING: min capture < 0.80 — consider increasing l_max!")
    print(f"\nMean OAM spectrum (top 5 modes by power):")
    for idx in top5_idx:
        p_mode, l_mode = basis.modes[idx]
        print(f"  (p={p_mode}, l={l_mode:+d})  {mean_spec[idx]:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
