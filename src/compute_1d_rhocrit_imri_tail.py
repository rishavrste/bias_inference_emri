"""
Compute shift-aggregated sigmas and 1D rho_crit for the IMRI_TAIL 1PA case.

For each of the 25 IMRI_TAIL grid points, 9 Fisher matrices are loaded from:
    fishers/mle/IMRI_TAIL_1pa/shift_{s}/Fishers/Fisher_{i}.h5
    (computed by fisher_grid_optimized_shifts_1pa_imri_tail_mle.py)

Parameter set (K=8 intrinsic, no extrinsic):
    ['m1', 'm2', 'a', 'p0', 'e0', 'Phi_phi0', 'Phi_r0', 'chi2']

Outputs (in bias_inference_emri/data/):
    sigmas/sigmas_1pa_K8_imri_tail_all.npy      (25, 9, 8)
    sigmas/sigmas_1pa_K8_imri_tail_median.npy   (25, 8)
    sigmas/sigmas_1pa_K8_imri_tail_nominal.npy  (25, 8)
    rhocrit/rhocrit_1d_1pa_K8_imri_tail.npy    (25,)
    rhocrit/rhocrit_1d_imri_tail_summary.txt

Usage:
    python compute_1d_rhocrit_imri_tail.py
"""
import os
import numpy as np
import h5py

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SRC_DIR, '..', 'data')
SIGMA_DIR   = os.path.join(DATA_DIR, 'sigmas')
RHOCRIT_DIR = os.path.join(DATA_DIR, 'rhocrit')

FISHER_BASE      = '/scratch/josh.mat/plot_scripts/data/fishers'
FISHER_IMRI_TAIL = os.path.join(FISHER_BASE, 'mle', 'IMRI_TAIL_1pa')

SIGNAL_FILE = '/scratch/josh.mat/plot_scripts/data/signal/signal_parameter_array_IMRI_TAIL.npy'
REC_FILE    = '/scratch/josh.mat/plot_scripts/data/recovered/mle/recovered_parameter_array_IMRI_TAIL_1pa.npy'

# Column indices (same layout as EMRI arrays)
SIG_COLS = [0, 1, 2, 3, 4, 11, 13, 16]  # m1,m2,a,p0,e0,Phi_phi0,Phi_r0,chi2
REC_COLS = [0, 1, 2, 3, 4, 11, 13, 16]

PARAM_NAMES = ['m1', 'm2', 'a', 'p0', 'e0', 'Phi_phi0', 'Phi_r0', 'chi2']
N_SHIFTS    = 9
N_GRID      = 25
SNR0        = 20.0
R1_99       = 2.5758   # norm.ppf(0.995)


def invert_log_fisher(Fisher_transformed, m1, m2):
    K  = Fisher_transformed.shape[0]
    J  = np.eye(K); J[0, 0] = m1; J[1, 1] = m2
    ev = np.linalg.eigvalsh(Fisher_transformed)
    cond = ev.max() / max(ev.min(), 1e-300)
    if not (ev > 0).all():
        return None, cond
    C_log  = np.linalg.inv(Fisher_transformed)
    C_phys = J @ C_log @ J
    return np.sqrt(np.diag(C_phys)), cond


def load_sigmas_for_point(fisher_base, idx, m1, m2):
    K         = len(PARAM_NAMES)
    sigma_all = np.full((N_SHIFTS, K), np.nan)
    for s in range(N_SHIFTS):
        fpath = os.path.join(fisher_base, f'shift_{s}', 'Fishers', f'Fisher_{idx}.h5')
        if not os.path.exists(fpath):
            continue
        try:
            with h5py.File(fpath, 'r') as f:
                Ft = f['Fisher_transformed'][:]
            sigmas, cond = invert_log_fisher(Ft, m1, m2)
            if sigmas is None:
                print(f'    [SKIP] shift_{s} idx={idx}: not PD  cond={cond:.2e}')
                continue
            sigma_all[s] = sigmas
            if s == 4:
                print(f'    nominal: cond={cond:.2e}  σ_m1={sigmas[0]:.3e}  '
                      f'σ_a={sigmas[2]:.3e}  σ_chi2={sigmas[7]:.3e}')
        except Exception as exc:
            print(f'    [WARN] shift_{s} idx={idx}: {exc}')
    return sigma_all


def main():
    os.makedirs(SIGMA_DIR,   exist_ok=True)
    os.makedirs(RHOCRIT_DIR, exist_ok=True)

    signal  = np.load(SIGNAL_FILE)
    rec     = np.load(REC_FILE)
    K       = len(PARAM_NAMES)

    sigma_all = np.full((N_GRID, N_SHIFTS, K), np.nan)

    for i in range(N_GRID):
        m1 = float(rec[i, REC_COLS[0]])
        m2 = float(rec[i, REC_COLS[1]])
        a_inj  = float(signal[i, 2])
        e0_inj = float(signal[i, 4])
        print(f'\n[{i:2d}/24]  a_inj={a_inj:+.2f}  e0_inj={e0_inj:.2f}')
        sigma_all[i] = load_sigmas_for_point(FISHER_IMRI_TAIL, i, m1, m2)

    NOMINAL = 4
    sigma_med = np.nanmedian(sigma_all, axis=1)   # (25, 8)
    sigma_nom = sigma_all[:, NOMINAL, :]           # (25, 8)

    np.save(os.path.join(SIGMA_DIR, 'sigmas_1pa_K8_imri_tail_all.npy'),     sigma_all)
    np.save(os.path.join(SIGMA_DIR, 'sigmas_1pa_K8_imri_tail_median.npy'),  sigma_med)
    np.save(os.path.join(SIGMA_DIR, 'sigmas_1pa_K8_imri_tail_nominal.npy'), sigma_nom)
    print(f'\nSaved sigma arrays to {SIGMA_DIR}/')

    # 1D rho_crit
    rhocrit = np.full(N_GRID, np.nan)
    summary = os.path.join(RHOCRIT_DIR, 'rhocrit_1d_imri_tail_summary.txt')
    with open(summary, 'w') as fout:
        hdr = (f'1D rho_crit — IMRI_TAIL 1PA intrinsic-only Fisher at MLE recovery point\n'
               f'Shift aggregation: {N_SHIFTS} shifts, median sigma\n'
               f'r_{{1,0.99}} = {R1_99:.4f},  SNR0 = {SNR0}\n'
               f'K=8: {PARAM_NAMES}\n')
        fout.write(hdr); print(hdr)

        abbrev = {'m1':'m1','m2':'m2','a':'a','p0':'p0','e0':'e0',
                  'Phi_phi0':'Φφ','Phi_r0':'Φr','chi2':'χ2'}
        col_hdr = '  '.join(f'd_{abbrev[p]:4s}' for p in PARAM_NAMES)
        hdr2 = f"\n{'idx':>3}  {'a':>6}  {'e0':>5}  ||  {col_hdr}  {'max_d':>8}  {'ρcrit':>8}\n"
        fout.write(hdr2 + '-'*len(hdr2.split('\n')[1]) + '\n')
        print(hdr2, end='')

        for i in range(N_GRID):
            a_inj  = float(signal[i, 2])
            e0_inj = float(signal[i, 4])
            delta  = np.array([rec[i, REC_COLS[k]] - signal[i, SIG_COLS[k]]
                               for k in range(K)])
            s = sigma_med[i]
            d = np.where(s > 0, np.abs(delta) / s, np.nan)
            d_max = np.nanmax(d)
            rc    = SNR0 * R1_99 / d_max if np.isfinite(d_max) and d_max > 0 else np.nan
            rhocrit[i] = rc

            def fmt(v, w=8):
                return f'{v:{w}.3f}' if np.isfinite(v) else ' '*(w-3)+'nan'
            line = (f'{i:3d}  {a_inj:+5.2f}  {e0_inj:.2f}  ||  '
                    + '  '.join(fmt(v) for v in d)
                    + f'  {fmt(d_max)}  {fmt(rc)}\n')
            fout.write(line); print(line, end='')

    np.save(os.path.join(RHOCRIT_DIR, 'rhocrit_1d_1pa_K8_imri_tail.npy'), rhocrit)
    print(f'\n1PA IMRI_TAIL rho_crit: min={np.nanmin(rhocrit):.1f}  '
          f'max={np.nanmax(rhocrit):.1f}  median={np.nanmedian(rhocrit):.1f}')
    print(f'Summary: {summary}')


if __name__ == '__main__':
    main()
