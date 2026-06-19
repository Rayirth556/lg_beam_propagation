"""Standalone verification script. Run before generate_dataset.py."""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from config import grid, beam, fog_cfg, basis
from physics.beam import generate_lg, precompute_basis
from physics.propagation import asm_transfer, asm_step
from physics.decomposition import oam_spectrum, basis_capture
from fog.mie import MieModel
from fog.screens import generate_screen, simulate

os.makedirs("outputs", exist_ok=True)


def _assert(cond: bool, msg: str):
    if not cond:
        print(f"  FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  PASS: {msg}")


def build_grid():
    x = (np.arange(grid.N) - grid.N // 2) * grid.dx
    X, Y = np.meshgrid(x, x, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X)
    return X, Y, R, PHI


# ---------------------------------------------------------------------------
# Check 1 — LG beam
# ---------------------------------------------------------------------------
def check1_lg_beam(R, PHI):
    print("\n[Check 1] LG beam generation")
    E = generate_lg(beam.l_in, beam.p_in, beam.w0, R, PHI, grid.dx)
    _assert(E.shape == (grid.N, grid.N), f"shape is {E.shape}")

    power = np.sum(np.abs(E)**2) * grid.dx**2
    _assert(abs(power - 1.0) < 0.01, f"power = {power:.6f} (should be 1.0 ± 1%)")

    # Phase winds 2π around circle at r = w0
    theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    r_circle = beam.w0
    xi = r_circle * np.cos(theta)
    yi = r_circle * np.sin(theta)
    # Sample phase via bilinear interpolation indices
    x_arr = (np.arange(grid.N) - grid.N // 2) * grid.dx
    xi_idx = np.interp(xi, x_arr, np.arange(grid.N))
    yi_idx = np.interp(yi, x_arr, np.arange(grid.N))
    xi_i = np.clip(xi_idx.astype(int), 0, grid.N - 1)
    yi_i = np.clip(yi_idx.astype(int), 0, grid.N - 1)
    phase_circle = np.angle(E[xi_i, yi_i])
    unwrapped = np.unwrap(phase_circle)
    total_wind = (unwrapped[-1] - unwrapped[0]) / (2 * np.pi)
    _assert(abs(total_wind - beam.l_in) < 0.1,
            f"phase winding = {total_wind:.3f} (should be {beam.l_in})")

    # Intensity ring peak at r = w0/sqrt(2) for p=0, |l|=1
    I = np.abs(E)**2
    # Radial profile
    r_flat = R.ravel()
    I_flat = I.ravel()
    sort_idx = np.argsort(r_flat)
    r_s = r_flat[sort_idx]
    I_s = I_flat[sort_idx]
    r_peak = r_s[np.argmax(I_s)]
    expected_peak = beam.w0 / np.sqrt(2 * (beam.p_in + 1) if beam.p_in > 0 else 2)
    _assert(abs(r_peak - expected_peak) < 0.3e-3,
            f"intensity ring peak at r={r_peak*1e3:.3f}mm (expected {expected_peak*1e3:.3f}mm)")

    # Save figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ext = grid.L_m * 1e3 / 2
    axes[0].imshow(I.T, origin='lower', extent=[-ext, ext, -ext, ext], cmap='hot')
    axes[0].set_title(f'LG intensity (l={beam.l_in}, p={beam.p_in})')
    axes[0].set_xlabel('x [mm]'); axes[0].set_ylabel('y [mm]')
    ph = axes[1].imshow(np.angle(E).T, origin='lower', extent=[-ext, ext, -ext, ext], cmap='hsv')
    axes[1].set_title('LG phase')
    axes[1].set_xlabel('x [mm]')
    plt.colorbar(ph, ax=axes[1], label='Phase [rad]')
    plt.tight_layout()
    plt.savefig('outputs/check1_lg_beam.png', dpi=120)
    plt.close()
    print("  Saved: outputs/check1_lg_beam.png")
    return E


# ---------------------------------------------------------------------------
# Check 2 — ASM free-space power conservation
# ---------------------------------------------------------------------------
def check2_asm(E_in):
    print("\n[Check 2] ASM free-space propagation")
    H = asm_transfer(fog_cfg.dz, grid)
    E = E_in.copy()
    P0 = np.sum(np.abs(E)**2) * grid.dx**2
    for step in range(fog_cfg.n_screens):
        E = asm_step(E, H)
        P = np.sum(np.abs(E)**2) * grid.dx**2
        rel_err = abs(P - P0) / P0
        _assert(rel_err < 0.001,
                f"step {step+1}: power={P:.6f}  rel_err={rel_err:.2e} (need < 0.1%)")
    print(f"  Power conserved over {fog_cfg.n_screens} steps.")


# ---------------------------------------------------------------------------
# Check 3 — OAM decomposition on input beam
# ---------------------------------------------------------------------------
def check3_decomposition(E_in, basis_fields):
    print("\n[Check 3] OAM decomposition on input beam")
    spec = oam_spectrum(E_in, basis_fields, grid.dx)
    cap = basis_capture(E_in, basis_fields, grid.dx)

    # Find l=1, p=0 index
    l1_idx = basis.modes.index((beam.p_in, beam.l_in))
    l1_power = spec[l1_idx]
    _assert(l1_power > 0.99, f"l=1 power = {l1_power:.6f} (should be > 99%)")
    _assert(cap > 0.99, f"basis_capture = {cap:.6f} (should be > 0.99)")

    print(f"  Full OAM spectrum:")
    for (p, l), s in zip(basis.modes, spec):
        if s > 1e-4:
            print(f"    (p={p}, l={l:+d}) = {s:.6f}")
    print(f"  basis_capture = {cap:.6f}")


# ---------------------------------------------------------------------------
# Check 4 — Single fog screen statistics
# ---------------------------------------------------------------------------
def check4_fog_screen():
    print("\n[Check 4] Single fog screen statistics")
    mie = MieModel(fog_cfg.r_eff, grid.wavelength)
    fp = mie.fog_params(100e6, fog_cfg.dz)

    T_amp = fp['T_amp']
    sigma_A = fp['sigma_A']

    rng = np.random.default_rng(0)
    n_trials = 200
    means, stds = [], []
    for _ in range(n_trials):
        screen = generate_screen(fp, grid.N, rng)
        amp = np.abs(screen)
        means.append(amp.mean())
        stds.append(amp.std())

    mean_amp = np.mean(means)
    mean_std = np.mean(stds)
    _assert(abs(mean_amp - T_amp) / T_amp < 0.05,
            f"mean|screen|={mean_amp:.5f}, T_amp={T_amp:.5f} (within 5%)")
    _assert(abs(mean_std - sigma_A) / (sigma_A + 1e-10) < 0.10,
            f"std|screen|={mean_std:.5f}, sigma_A={sigma_A:.5f} (within 10%)")


# ---------------------------------------------------------------------------
# Check 5 — End-to-end single sample
# ---------------------------------------------------------------------------
def check5_e2e(E_in, basis_fields, R, PHI):
    print("\n[Check 5] End-to-end single sample (N_d=100e6)")
    mie = MieModel(fog_cfg.r_eff, grid.wavelength)
    fp = mie.fog_params(100e6, fog_cfg.dz)
    H      = asm_transfer(fog_cfg.dz, grid)
    H_back = asm_transfer(-fog_cfg.path_m, grid)        # ← ADD
    rng = np.random.default_rng(7)

    E_out = simulate(E_in, fp, H, fog_cfg.n_screens, rng)
    E_dec = asm_step(E_out, H_back)                     # ← ADD

    spec = oam_spectrum(E_dec, basis_fields, grid.dx)   # ← E_out → E_dec
    cap  = basis_capture(E_dec, basis_fields, grid.dx)  # ← E_out → E_dec

    l1_idx = basis.modes.index((beam.p_in, beam.l_in))
    l1_power = spec[l1_idx]
    _assert(l1_power < 1.0, f"l=1 power = {l1_power:.4f} (should be < 100% after fog)")
    _assert(cap > 0.80, f"basis_capture = {cap:.4f} (should be > 0.80)")

    print(f"  OAM spectrum after fog:")
    for (p, l), s in zip(basis.modes, spec):
        if s > 0.01:
            print(f"    (p={p}, l={l:+d}) = {s:.4f}")
    print(f"  basis_capture = {cap:.4f}")

    # Save figure — I_out stays E_out (what the camera sees at z=20m)
    I_in  = np.abs(E_in)**2
    I_out = np.abs(E_out)**2
    ext   = grid.L_m * 1e3 / 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(I_in.T,  origin='lower', extent=[-ext, ext, -ext, ext], cmap='hot')
    axes[0].set_title('Input intensity')
    axes[1].imshow(I_out.T, origin='lower', extent=[-ext, ext, -ext, ext], cmap='hot')
    axes[1].set_title('Output intensity (fog)')

    l_vals = [l for p, l in basis.modes]
    axes[2].bar(l_vals, spec, color='steelblue')
    axes[2].set_xlabel('OAM mode l'); axes[2].set_ylabel('Power fraction')
    axes[2].set_title('OAM spectrum')
    axes[2].set_xticks(l_vals)
    plt.tight_layout()
    plt.savefig('outputs/check5_e2e.png', dpi=120)
    plt.close()
    print("  Saved: outputs/check5_e2e.png")


# ---------------------------------------------------------------------------
# Check 6 — Four fog densities side by side
# ---------------------------------------------------------------------------
def check6_four_densities(E_in, basis_fields):
    print("\n[Check 6] Four fog densities side by side")
    nd_list = [12e6, 50e6, 100e6, 140e6]
    mie    = MieModel(fog_cfg.r_eff, grid.wavelength)
    H      = asm_transfer(fog_cfg.dz, grid)
    H_back = asm_transfer(-fog_cfg.path_m, grid)              # ← ADD

    results = []
    for nd in nd_list:
        fp    = mie.fog_params(nd, fog_cfg.dz)
        rng   = np.random.default_rng(42)
        E_out = simulate(E_in, fp, H, fog_cfg.n_screens, rng)
        I     = np.abs(E_out)**2                              # ← keep E_out for image
        E_dec = asm_step(E_out, H_back)                      # ← ADD
        spec  = oam_spectrum(E_dec, basis_fields, grid.dx)   # ← E_out → E_dec
        cap   = basis_capture(E_dec, basis_fields, grid.dx)  # ← E_out → E_dec
        n_active = int(np.sum(spec > 0.02))
        results.append((nd, fp['vis_m'], I, spec, cap, n_active))
        print(f"  N_d={nd:.0e}: vis={fp['vis_m']:.1f}m  "
              f"l=1 power={spec[basis.modes.index((0,1))]:.3f}  "
              f"modes>2%={n_active}  capture={cap:.3f}")

    # Expected trend check
    l1_powers = [spec[basis.modes.index((0, 1))] for _, _, _, spec, _, _ in results]
    _assert(l1_powers[0] > l1_powers[-1],
            f"l=1 power should decrease with fog density: {l1_powers}")

    ext    = grid.L_m * 1e3 / 2
    l_vals = [l for p, l in basis.modes]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for col, (nd, vis, I, spec, cap, n_active) in enumerate(results):
        axes[0, col].imshow(I.T, origin='lower', extent=[-ext, ext, -ext, ext], cmap='hot')
        axes[0, col].set_title(f'N_d={nd:.0e}\nV={vis:.0f}m')
        axes[0, col].set_xlabel('x [mm]')
        if col == 0:
            axes[0, col].set_ylabel('y [mm]')

        axes[1, col].bar(l_vals, spec, color='steelblue')
        axes[1, col].set_xlabel('OAM mode l')
        axes[1, col].set_ylim(0, 1)
        if col == 0:
            axes[1, col].set_ylabel('Power fraction')
        axes[1, col].set_title(f'capture={cap:.2f}  >2%: {n_active}')
        axes[1, col].set_xticks(l_vals[::2])

    plt.suptitle('OAM mode mixing at four fog densities', fontsize=13)
    plt.tight_layout()
    plt.savefig('outputs/check6_four_densities.png', dpi=120)
    plt.close()
    print("  Saved: outputs/check6_four_densities.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("="*60)
    print("LG Beam / Fog Propagation — Verification Suite")
    print("="*60)

    X, Y, R, PHI = build_grid()
    basis_fields = precompute_basis(basis.modes, beam.w0, R, PHI, grid.dx)

    passed = 0
    failed = 0
    checks = [
        lambda: check1_lg_beam(R, PHI),
        lambda: check2_asm(generate_lg(beam.l_in, beam.p_in, beam.w0, R, PHI, grid.dx)),
        lambda: check3_decomposition(
            generate_lg(beam.l_in, beam.p_in, beam.w0, R, PHI, grid.dx), basis_fields),
        check4_fog_screen,
        lambda: check5_e2e(
            generate_lg(beam.l_in, beam.p_in, beam.w0, R, PHI, grid.dx), basis_fields, R, PHI),
        lambda: check6_four_densities(
            generate_lg(beam.l_in, beam.p_in, beam.w0, R, PHI, grid.dx), basis_fields),
    ]

    for i, check in enumerate(checks, 1):
        try:
            check()
            passed += 1
        except AssertionError as e:
            print(f"  *** Check {i} FAILED: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  *** Check {i} ERROR: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All checks passed — safe to run generate_dataset.py")
    else:
        print("Fix failures before running generate_dataset.py")
    print("="*60)
    return failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
