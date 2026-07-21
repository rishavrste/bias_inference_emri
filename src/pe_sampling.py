#!/usr/bin/env python3
"""
Posterior sampling at the MLE point for bias-inference validation.

Runs 6 combinations: EMRI/IMRI/IMRI_TAIL × 0PA/1PA.
Samples the template log-likelihood around the MLE point found by inference.py
and compares the resulting posterior to the Fisher-matrix prediction.

Parameter space (0PA, ndim=12):
  [m1, m2, a, p0, e0, Phi_phi0, Phi_r0, dist, qS, phiS, qK, phiK]

Parameter space (1PA, ndim=13):
  [m1, m2, a, p0, e0, Phi_phi0, Phi_r0, chi2, dist, qS, phiS, qK, phiK]

Phi_theta0 and Y0 (xI0) are fixed at their signal values throughout.

Prior:
  - Intrinsic (m1..e0, chi2): Fisher sigma × PSR centred on signal
  - Phases (Phi_phi0, Phi_r0): [0, 2π]
  - chi2: [-1, 1]
  - dist: signal ± PSR/SNR × signal
  - Sky angles (qS, phiS, qK, phiK): signal ± 0.5 rad

Usage:
  python pe_sampling.py --type IMRI_TAIL --run-type 1pa_vs_2pa --point 0
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from misc import calculate_log_likelihood, compute_fisher_parallelotope
from inference import prepare_true_waveform, build_waveform_response

try:
    from parismc import Sampler, SamplerConfig
except ImportError:
    sys.path.insert(0, os.path.expanduser('~/git_repos/parismc'))
    from parismc import Sampler, SamplerConfig

import cupy as cp

# Column indices in the 17-column parameter array
_COL = {
    'm1': 0, 'm2': 1, 'a': 2, 'p0': 3, 'e0': 4, 'Y0': 5,
    'dist': 6, 'qS': 7, 'phiS': 8, 'qK': 9, 'phiK': 10,
    'Phi_phi0': 11, 'Phi_theta0': 12, 'Phi_r0': 13,
    'dt': 14, 'T': 15, 'chi2': 16,
}

# PE parameter ordering (chi2 appears only for 1PA)
_PARAMS_0PA = ['m1', 'm2', 'a', 'p0', 'e0', 'Phi_phi0', 'Phi_r0',
               'dist', 'qS', 'phiS', 'qK', 'phiK']
_PARAMS_1PA = ['m1', 'm2', 'a', 'p0', 'e0', 'Phi_phi0', 'Phi_r0', 'chi2',
               'dist', 'qS', 'phiS', 'qK', 'phiK']


def _load_fisher_sigma(cache_dir: str, point: int, is_1pa: bool, snr: float) -> np.ndarray:
    """Return Fisher 1σ widths [m1, m2, a, p0, e0 (, chi2)] at given SNR."""
    tag = 'm1_m2_a_p0_e0_chi2' if is_1pa else 'm1_m2_a_p0_e0'
    path = os.path.join(cache_dir, f'fisher_{point:04d}_2pa_{tag}.npy')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Fisher cache not found: {path}')
    d = np.load(path, allow_pickle=True).item()
    F = np.asarray(d['F'], dtype=float)
    snr_model = float(d['snr_model'])
    scale = (snr / max(snr_model, 1e-30)) ** 2
    Finv = np.linalg.inv(F * scale)
    return np.sqrt(np.diag(Finv))


def _build_bounds(signal_row: np.ndarray, sigma_intr: np.ndarray, snr: float,
                  psr: float, param_names: list) -> tuple:
    """Return (bounds_lo, bounds_hi) arrays in theta space."""
    lo, hi = {}, {}

    # Intrinsic: Fisher ± PSR centred on signal
    for i, p in enumerate(['m1', 'm2', 'a', 'p0', 'e0']):
        c = _COL[p]
        lo[p] = signal_row[c] - psr * sigma_intr[i]
        hi[p] = signal_row[c] + psr * sigma_intr[i]
    lo['a'] = max(lo['a'], -0.99);  hi['a'] = min(hi['a'], 0.99)
    lo['p0'] = max(lo['p0'], 1.0)
    lo['e0'] = max(lo['e0'], 1e-4); hi['e0'] = min(hi['e0'], 0.9)

    # Phases: flat over full circle
    lo['Phi_phi0'] = 0.0;  hi['Phi_phi0'] = 2.0 * np.pi
    lo['Phi_r0']   = 0.0;  hi['Phi_r0']   = 2.0 * np.pi

    # chi2 (1PA only): hard physical bounds
    if 'chi2' in param_names:
        sigma_chi2 = float(sigma_intr[5])
        lo['chi2'] = max(-1.0, signal_row[_COL['chi2']] - psr * sigma_chi2)
        hi['chi2'] = min( 1.0, signal_row[_COL['chi2']] + psr * sigma_chi2)
        # always cover full range; chi2 is usually poorly constrained
        lo['chi2'] = -1.0
        hi['chi2'] =  1.0

    # Distance: PSR × (dist/SNR) about the signal distance
    d_sig = signal_row[_COL['dist']]
    sigma_dist = d_sig / snr
    lo['dist'] = max(d_sig - psr * sigma_dist, 1e-3)
    hi['dist'] = d_sig + psr * sigma_dist

    # Sky angles: generous ±0.5 rad window around signal
    for p in ['qS', 'qK']:
        v = signal_row[_COL[p]]
        lo[p] = max(v - 0.5, 0.0)
        hi[p] = min(v + 0.5, np.pi)
    for p in ['phiS', 'phiK']:
        v = signal_row[_COL[p]]
        lo[p] = v - 0.5
        hi[p] = v + 0.5

    lo_arr = np.array([lo[p] for p in param_names])
    hi_arr = np.array([hi[p] for p in param_names])
    return lo_arr, hi_arr


def main():
    parser = argparse.ArgumentParser(description='PE posterior sampling at MLE point')
    parser.add_argument('--type', required=True, choices=['EMRI', 'IMRI', 'IMRI_TAIL'],
                        help='Grid type')
    parser.add_argument('--run-type', required=True,
                        choices=['0pa_vs_2pa', '1pa_vs_2pa'],
                        help='Template PA order vs 2PA signal')
    parser.add_argument('--point', type=int, required=True,
                        help='Grid point index (0-24)')
    parser.add_argument('--prior-sigma-range', type=float, default=5.0,
                        help='Prior half-width in Fisher sigmas for intrinsic params')
    parser.add_argument('--n-seed', type=int, default=100,
                        help='Number of PARIS seeds (independent chains)')
    parser.add_argument('--num-iterations', type=int, default=100_000,
                        help='Maximum PARIS iterations')
    parser.add_argument('--stop-dlogz', type=float, default=0.01,
                        help='Stop when |ΔlogZ| < this value')
    parser.add_argument('--savepath', default=None,
                        help='Output directory (auto-generated if not set)')
    args = parser.parse_args()

    is_1pa = (args.run_type == '1pa_vs_2pa')
    pa_tag = '1pa' if is_1pa else '0pa'
    param_names = _PARAMS_1PA if is_1pa else _PARAMS_0PA
    ndim = len(param_names)

    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    data_dir = os.path.join(repo_root, 'data')
    fisher_cache_dir = os.path.join(data_dir, 'fisher_cache', args.type)

    if args.savepath is None:
        args.savepath = os.path.join(
            '/scratch/josh.mat/opt_grid/pe_results',
            f'{args.type}_{pa_tag}_pt{args.point:02d}'
        )
    os.makedirs(args.savepath, exist_ok=True)

    print(f'PE sampling: {args.type} {args.run_type} pt{args.point}')
    print(f'ndim={ndim}  params={param_names}')
    print(f'savepath={args.savepath}')

    # Load parameter arrays
    signal_arr = np.load(os.path.join(data_dir, f'signal_parameter_array_{args.type}.npy'))
    mle_arr = np.load(
        os.path.join(data_dir, f'result_parameter_array_{args.type}_{pa_tag}_mle.npy')
    )
    signal_row = signal_arr[args.point]
    mle_row    = mle_arr[args.point]

    # Fixed signal quantities (never sampled)
    Y0_sig        = float(signal_row[_COL['Y0']])
    Phi_theta0_sig = float(signal_row[_COL['Phi_theta0']])
    dt_sig        = float(signal_row[_COL['dt']])
    T_sig         = float(signal_row[_COL['T']])
    chi2_sig      = float(signal_row[_COL['chi2']])

    # Build 2PA signal context (true waveform + PSD)
    emri_kw_signal = {
        'T': T_sig, 'dt': dt_sig, 'chi2': chi2_sig,
        'evolve_1PA': True, 'evolve_primary': False, 'evolve_2PA': True,
    }
    add_kw_signal = {
        'chi2': chi2_sig,
        'evolve_1PA': True, 'evolve_primary': False, 'evolve_2PA': True,
    }
    print('Building 2PA signal waveform...')
    ctx = prepare_true_waveform(
        signal_row[:14], emri_kw_signal, add_kw_signal,
        add_noise=False, use_gpu=True, nchannels=2
    )
    snr = float(ctx['snr'])
    print(f'Signal SNR = {snr:.4f}')

    # Template evolution flags
    if is_1pa:
        tmpl_evolve = {'evolve_1PA': True, 'evolve_2PA': False, 'evolve_primary': False}
    else:
        tmpl_evolve = {'evolve_1PA': False, 'evolve_2PA': False, 'evolve_primary': False}

    # Fixed dict for likelihood evaluation
    fixed = {
        'waveform_true_fft': ctx['waveform_true_fft'],
        'waveform_response':  ctx['waveform_response'],
        'PSD':                ctx['PSD_funcs'],
        'dt':                 ctx['dt'],
        'T':                  ctx['T'],
        'N_fiducial':         ctx['N_fiducial'],
        'delta_f':            ctx['delta_f'],
        'use_gpu':            True,
    }

    # Fisher sigmas for intrinsic params from precomputed cache
    print('Loading Fisher matrix from cache...')
    sigma_intr = _load_fisher_sigma(fisher_cache_dir, args.point, is_1pa, snr)
    print(f'Fisher sigma: {dict(zip(["m1","m2","a","p0","e0","chi2"][:len(sigma_intr)], sigma_intr))}')

    # Prior bounds
    bounds_lo, bounds_hi = _build_bounds(
        signal_row, sigma_intr, snr, args.prior_sigma_range, param_names
    )
    print('\nPrior bounds:')
    for i, p in enumerate(param_names):
        print(f'  {p:10s}: [{bounds_lo[i]:.6g}, {bounds_hi[i]:.6g}]')

    # MLE theta in PE parameter space
    mle_theta = np.array([mle_row[_COL[p]] for p in param_names])
    print(f'\nMLE theta: {dict(zip(param_names, mle_theta))}')

    # MLE in unit hypercube
    span = bounds_hi - bounds_lo
    mle_u = np.clip((mle_theta - bounds_lo) / span, 0.01, 0.99)
    print(f'MLE in unit hypercube: {np.round(mle_u, 4)}')

    # ------------------------------------------------------------------ #
    # Log-density (physical-space batch, shape [N, ndim] → [N])
    # The sampler calls: log_density(prior_transform(u_batch))
    # ------------------------------------------------------------------ #
    def log_density(theta_batch: np.ndarray) -> np.ndarray:
        if theta_batch.ndim == 1:
            theta_batch = theta_batch[np.newaxis, :]
        results = np.full(len(theta_batch), -np.inf)
        for k, theta in enumerate(theta_batch):
            m1, m2, a, p0, e0, Phi_phi0, Phi_r0 = theta[:7]
            if is_1pa:
                chi2_val, dist, qS, phiS, qK, phiK = theta[7:]
            else:
                dist, qS, phiS, qK, phiK = theta[7:]
                chi2_val = 0.0
            ak = {'chi2': chi2_val, **tmpl_evolve}
            try:
                ll = calculate_log_likelihood(
                    m1, m2, a, p0, e0, Y0_sig, dist, qS, phiS, qK, phiK,
                    Phi_phi0, Phi_theta0_sig, Phi_r0, ak,
                    **fixed,
                )
            except Exception as exc:
                print(f'  [WARN] log_likelihood failed for sample {k}: {exc}')
                ll = -np.inf
            results[k] = ll
        return results

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return bounds_lo + u * span

    # ------------------------------------------------------------------ #
    # PARIS sampler
    # ------------------------------------------------------------------ #
    config = SamplerConfig(
        merge_confidence=0.9,
        alpha=5000,
        trail_size=int(1e3),
        boundary_limiting=True,
        use_beta=True,
        integral_num=int(1e5),
        gamma=500,
        use_pool=False,
    )

    init_cov_list = [np.eye(ndim) * 1e-10] * args.n_seed

    sampler = Sampler(
        ndim=ndim,
        n_seed=args.n_seed,
        log_density_func=log_density,
        init_cov_list=init_cov_list,
        prior_transform=prior_transform,
        config=config,
    )

    # Seed tightly around MLE in unit-hypercube space
    np.random.seed(42)
    scatter = 1e-7
    seeds = mle_u + np.random.randn(args.n_seed - 1, ndim) * scatter
    seeds = np.vstack([seeds, mle_u])
    seeds = np.clip(seeds, 0.0, 1.0)

    print('\nEvaluating log-density at seed points...')
    seeds_logL = log_density(prior_transform(seeds))
    print(f'Seed log-L range: [{seeds_logL.min():.2f}, {seeds_logL.max():.2f}]')
    print(f'MLE seed log-L: {seeds_logL[-1]:.4f}')

    print(f'\nStarting PARIS (ndim={ndim}, n_seed={args.n_seed}, '
          f'max_iter={args.num_iterations})...')
    sampler.run_sampling(
        num_iterations=args.num_iterations,
        savepath=args.savepath,
        print_iter=100,
        external_lhs_points=seeds,
        external_lhs_log_densities=seeds_logL,
        stop_dlogZ=args.stop_dlogz,
    )

    # ------------------------------------------------------------------ #
    # Extract and save results
    # ------------------------------------------------------------------ #
    print('\nExtracting results...')
    samples_u, weights = sampler.get_samples_with_weights(flatten=True)
    samples = prior_transform(samples_u)
    ess = float(1.0 / np.sum(weights ** 2))

    print(f'Total samples: {len(samples)}')
    print(f'Effective sample size: {ess:.1f}')

    weighted_mean = np.average(samples, weights=weights, axis=0)
    weighted_cov  = np.cov(samples.T, aweights=weights)

    print('\nFisher vs PE marginal sigmas:')
    intr_params = ['m1', 'm2', 'a', 'p0', 'e0'] + (['chi2'] if is_1pa else [])
    for j, p in enumerate(intr_params):
        idx = param_names.index(p)
        pe_sig = float(np.sqrt(weighted_cov[idx, idx]))
        fish_sig = float(sigma_intr[j])
        print(f'  {p:6s}: Fisher={fish_sig:.4g}  PE={pe_sig:.4g}  ratio={pe_sig/max(fish_sig,1e-30):.3f}')

    signal_theta = np.array([signal_row[_COL[p]] for p in param_names])

    np.savez(
        os.path.join(args.savepath, 'pe_results.npz'),
        samples=samples,
        samples_unit=samples_u,
        weights=weights,
        weighted_mean=weighted_mean,
        weighted_cov=weighted_cov,
        param_names=np.array(param_names),
        signal_theta=signal_theta,
        mle_theta=mle_theta,
        sigma_fisher=sigma_intr,
        bounds_lo=bounds_lo,
        bounds_hi=bounds_hi,
        ess=ess,
        grid_type=args.type,
        run_type=args.run_type,
        point=args.point,
    )
    print(f'\nResults saved to {args.savepath}/pe_results.npz')


if __name__ == '__main__':
    main()
