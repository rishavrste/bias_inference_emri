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
  - Intrinsic (m1..e0): Fisher sigma × PSR centred on MLE (Fisher at MLE, template PA)
  - Phases (Phi_phi0, Phi_r0): [mle_phase − π, mle_phase + π] (full circle centred on MLE)
  - chi2 (1PA only): Fisher sigma × PSR centred on MLE chi2, clipped to [-1, 1]
  - dist: MLE dist ± (PSR/SNR) × MLE dist
  - Sky angles (qS, phiS, qK, phiK): MLE ± 0.5 rad

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

from parismc import Sampler, SamplerConfig

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

# ---------------------------------------------------------------------------
# Module-level globals — set by main() so that _log_density and
# _prior_transform are top-level functions and can be pickled by name
# (parismc.Sampler calls save_state() via pickle at end of run_sampling).
# ---------------------------------------------------------------------------
_PE_IS_1PA: bool = False
_PE_Y0: float = 1.0
_PE_PHI_THETA0: float = 0.0
_PE_TMPL_EVOLVE: dict = {}
_PE_FIXED: dict = {}
_PE_BOUNDS_LO: np.ndarray = None
_PE_SPAN: np.ndarray = None


def _log_density(theta_batch: np.ndarray) -> np.ndarray:
    """Log-likelihood at physical-space parameters; shape (N, ndim) → (N,)."""
    if theta_batch.ndim == 1:
        theta_batch = theta_batch[np.newaxis, :]
    results = np.full(len(theta_batch), -np.inf)
    for k, theta in enumerate(theta_batch):
        m1, m2, a, p0, e0, Phi_phi0, Phi_r0 = theta[:7]
        if _PE_IS_1PA:
            chi2_val, dist, qS, phiS, qK, phiK = theta[7:]
        else:
            dist, qS, phiS, qK, phiK = theta[7:]
            chi2_val = 0.0
        ak = {'chi2': chi2_val, **_PE_TMPL_EVOLVE}
        try:
            ll = calculate_log_likelihood(
                m1, m2, a, p0, e0, _PE_Y0, dist, qS, phiS, qK, phiK,
                Phi_phi0, _PE_PHI_THETA0, Phi_r0, ak,
                **_PE_FIXED,
            )
        except Exception as exc:
            print(f'  [WARN] log_likelihood failed for sample {k}: {exc}')
            ll = -np.inf
        results[k] = ll
    return results


def _prior_transform(u: np.ndarray) -> np.ndarray:
    """Map unit hypercube [0,1]^ndim to physical parameter space."""
    return _PE_BOUNDS_LO + u * _PE_SPAN


def _get_mle_fisher_sigma(mle_row: np.ndarray, point: int, is_1pa: bool,
                           type_name: str, snr: float,
                           use_gpu: bool = True) -> tuple:
    """Compute (or load cached) Fisher 1σ widths at the MLE point, template PA order.

    Includes initial phases (Phi_phi0, Phi_r0) in params_to_infer so that the
    Fisher gives guidance on phase prior widths — avoiding the flat-phase problem.

    Returns (sigma_arr, params_to_infer) so the caller knows param ordering.
    Cache lives in data/fisher_cache/{type_name}_mle/.
    """
    pa_tag = '1pa' if is_1pa else '0pa'
    if is_1pa:
        params_to_infer = ['m1', 'm2', 'a', 'p0', 'e0', 'chi2', 'Phi_phi0', 'Phi_r0']
    else:
        params_to_infer = ['m1', 'm2', 'a', 'p0', 'e0', 'Phi_phi0', 'Phi_r0']
    params_tag = '_'.join(params_to_infer)

    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    cache_dir = os.path.join(repo_root, 'data', 'fisher_cache', f'{type_name}_mle')
    cache_path = os.path.join(cache_dir, f'fisher_{point:04d}_{pa_tag}_{params_tag}.npy')

    if not os.path.exists(cache_path):
        print(f'[MLE FISHER] Cache miss — computing at MLE pt{point} ({pa_tag})...')
        T   = float(mle_row[_COL['T']])
        dt  = float(mle_row[_COL['dt']])
        chi2 = float(mle_row[_COL['chi2']]) if is_1pa else 0.0
        ctx = {
            'waveform_response': build_waveform_response(T=T, dt=dt, use_gpu=use_gpu),
            'waveform_true_fft': np.zeros((2, 1)),  # shape[0] = nchannels = 2
            'T': T, 'dt': dt, 'chi2': chi2,
        }
        add_kw = {
            'chi2': chi2,
            'evolve_1PA': is_1pa,
            'evolve_primary': False,
            'evolve_2PA': False,
        }
        compute_fisher_parallelotope(
            ctx=ctx,
            fisher_params=mle_row[:14],
            params_to_infer=params_to_infer,
            additional_kwargs=add_kw,
            use_gpu=use_gpu,
            _TARGET_SNR=1.0,
            prior_sigma_range=5.0,
            using_evec=False,
            cache_dir=cache_dir,
            cache_index=point,
        )

    d = np.load(cache_path, allow_pickle=True).item()
    F = np.asarray(d['F'], dtype=float)
    snr_model = float(d['snr_model'])
    scale = (snr / max(snr_model, 1e-30)) ** 2
    return np.sqrt(np.diag(np.linalg.inv(F * scale))), params_to_infer


def _build_bounds(mle_row: np.ndarray, sigma_dict: dict, snr: float,
                  psr: float, param_names: list) -> tuple:
    """Return (bounds_lo, bounds_hi) arrays centred on the MLE point.

    sigma_dict maps parameter name → Fisher 1σ width (from _get_mle_fisher_sigma).
    """
    lo, hi = {}, {}

    # Intrinsic: Fisher ± PSR centred on MLE
    for p in ['m1', 'm2', 'a', 'p0', 'e0']:
        c = _COL[p]
        lo[p] = mle_row[c] - psr * sigma_dict[p]
        hi[p] = mle_row[c] + psr * sigma_dict[p]
    lo['a']  = max(lo['a'],  -0.99); hi['a']  = min(hi['a'],  0.99)
    lo['p0'] = max(lo['p0'],  1.0)
    lo['e0'] = max(lo['e0'],  1e-4); hi['e0'] = min(hi['e0'], 0.9)

    # Phases: exactly one full 2π period centred on the MLE.
    # PSR×sigma always exceeds π for our configurations, so the Fisher-guided
    # window would be wider than 2π. Using [mle−π, mle+π] gives the same prior
    # volume as [0, 2π] while placing the MLE at mle_u = 0.5 for efficient seeding.
    for p in ['Phi_phi0', 'Phi_r0']:
        c = _COL[p]
        lo[p] = mle_row[c] - np.pi
        hi[p] = mle_row[c] + np.pi

    # chi2 (1PA only): Fisher ± PSR centred on MLE chi2, clipped to physical limits
    if 'chi2' in param_names:
        chi2_mle = float(mle_row[_COL['chi2']])
        lo['chi2'] = max(chi2_mle - psr * sigma_dict['chi2'], -1.0)
        hi['chi2'] = min(chi2_mle + psr * sigma_dict['chi2'],  1.0)

    # Distance: PSR × (dist/SNR) centred on MLE distance
    d_mle = float(mle_row[_COL['dist']])
    sigma_dist = d_mle / snr
    lo['dist'] = max(d_mle - psr * sigma_dist, 1e-3)
    hi['dist'] = d_mle + psr * sigma_dist

    # Sky angles: generous ±0.5 rad window centred on MLE
    for p in ['qS', 'qK']:
        v = float(mle_row[_COL[p]])
        lo[p] = max(v - 0.5, 0.0)
        hi[p] = min(v + 0.5, np.pi)
    for p in ['phiS', 'phiK']:
        v = float(mle_row[_COL[p]])
        lo[p] = v - 0.5
        hi[p] = v + 0.5

    lo_arr = np.array([lo[p] for p in param_names])
    hi_arr = np.array([hi[p] for p in param_names])
    return lo_arr, hi_arr


def main():
    global _PE_IS_1PA, _PE_Y0, _PE_PHI_THETA0, _PE_TMPL_EVOLVE
    global _PE_FIXED, _PE_BOUNDS_LO, _PE_SPAN

    parser = argparse.ArgumentParser(description='PE posterior sampling at MLE point')
    parser.add_argument('--type', required=True, choices=['EMRI', 'IMRI', 'IMRI_TAIL'],
                        help='Grid type')
    parser.add_argument('--run-type', required=True,
                        choices=['0pa_vs_2pa', '1pa_vs_2pa'],
                        help='Template PA order vs 2PA signal')
    parser.add_argument('--point', type=int, required=True,
                        help='Grid point index (0-24)')
    parser.add_argument('--prior-sigma-range', type=float, default=50.0,
                        help='Prior half-width in Fisher sigmas for intrinsic params')
    parser.add_argument('--n-seed', type=int, default=20,
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
    Y0_sig         = float(signal_row[_COL['Y0']])
    Phi_theta0_sig = float(signal_row[_COL['Phi_theta0']])
    dt_sig         = float(signal_row[_COL['dt']])
    T_sig          = float(signal_row[_COL['T']])
    chi2_sig       = float(signal_row[_COL['chi2']])

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

    # Template evolution flags (0PA: no post-adiabatic corrections; 1PA: first order only)
    if is_1pa:
        tmpl_evolve = {'evolve_1PA': True, 'evolve_2PA': False, 'evolve_primary': False}
    else:
        tmpl_evolve = {'evolve_1PA': False, 'evolve_2PA': False, 'evolve_primary': False}

    # Context dict for likelihood evaluation (no waveform generation here)
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

    # Fisher sigmas at MLE point, using template PA order (includes phases)
    print('Computing/loading Fisher at MLE point...')
    sigma_arr, fisher_params = _get_mle_fisher_sigma(
        mle_row, args.point, is_1pa, args.type, snr
    )
    sigma_dict = dict(zip(fisher_params, sigma_arr))
    print(f'Fisher sigma: {sigma_dict}')

    # param_labels: Fisher params that are also PE params (all Fisher params are in param_names)
    param_labels = [p for p in fisher_params if p in param_names]
    sigma_labels = np.array([sigma_dict[p] for p in param_labels])

    # Prior bounds centred on MLE
    bounds_lo, bounds_hi = _build_bounds(
        mle_row, sigma_dict, snr, args.prior_sigma_range, param_names
    )
    print('\nPrior bounds:')
    for i, p in enumerate(param_names):
        print(f'  {p:10s}: [{bounds_lo[i]:.6g}, {bounds_hi[i]:.6g}]')

    span = bounds_hi - bounds_lo

    # Set module-level globals so _log_density and _prior_transform can be pickled
    _PE_IS_1PA     = is_1pa
    _PE_Y0         = Y0_sig
    _PE_PHI_THETA0 = Phi_theta0_sig
    _PE_TMPL_EVOLVE = tmpl_evolve
    _PE_FIXED      = fixed
    _PE_BOUNDS_LO  = bounds_lo
    _PE_SPAN       = span

    # MLE theta in PE parameter space and in unit hypercube
    mle_theta = np.array([mle_row[_COL[p]] for p in param_names])
    mle_u = np.clip((mle_theta - bounds_lo) / span, 0.01, 0.99)
    print(f'\nMLE theta: {dict(zip(param_names, mle_theta))}')
    print(f'MLE in unit hypercube: {np.round(mle_u, 4)}')

    # PARIS sampler
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
        log_density_func=_log_density,
        init_cov_list=init_cov_list,
        prior_transform=_prior_transform,
        config=config,
    )

    # Seed around MLE in unit-hypercube space.
    # scatter=0.01 puts seeds at ±0.5σ from MLE (prevents immediate PARIS chain collapse).
    np.random.seed(42)
    scatter = 0.01
    seeds = mle_u + np.random.randn(args.n_seed - 1, ndim) * scatter
    seeds = np.vstack([seeds, mle_u])
    seeds = np.clip(seeds, 0.0, 1.0)

    print('\nEvaluating log-density at seed points...')
    seeds_logL = _log_density(_prior_transform(seeds))
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

    # Extract and save results
    # get_samples_with_weights applies prior_transform internally — returns physical space
    print('\nExtracting results...')
    samples, weights = sampler.get_samples_with_weights(flatten=True)
    samples_unit = np.clip((samples - bounds_lo) / span, 0.0, 1.0)
    ess = float(np.sum(weights) ** 2 / np.sum(weights ** 2))

    print(f'Total samples: {len(samples)}')
    print(f'Effective sample size: {ess:.1f}')

    w_norm = weights / weights.sum()
    weighted_mean = np.average(samples, weights=w_norm, axis=0)
    weighted_cov  = np.cov(samples.T, aweights=w_norm)

    print('\nFisher vs PE marginal sigmas:')
    for j, p in enumerate(param_labels):
        idx = param_names.index(p)
        pe_sig   = float(np.sqrt(weighted_cov[idx, idx]))
        fish_sig = float(sigma_labels[j])
        print(f'  {p:10s}: Fisher={fish_sig:.4g}  PE={pe_sig:.4g}  ratio={pe_sig/max(fish_sig,1e-30):.3f}')

    signal_theta = np.array([signal_row[_COL[p]] for p in param_names])

    np.savez(
        os.path.join(args.savepath, 'pe_results.npz'),
        samples=samples,
        samples_unit=samples_unit,
        weights=weights,
        weighted_mean=weighted_mean,
        weighted_cov=weighted_cov,
        param_names=np.array(param_names),
        fisher_params=np.array(param_labels),
        signal_theta=signal_theta,
        mle_theta=mle_theta,
        sigma_fisher=sigma_labels,
        bounds_lo=bounds_lo,
        bounds_hi=bounds_hi,
        ess=ess,
    )
    print(f'\nResults saved to {args.savepath}/pe_results.npz')


if __name__ == '__main__':
    main()
