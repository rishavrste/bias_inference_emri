#!/usr/bin/env python3
"""
Re-extract PE results from saved sampler_state.pkl files.

Fixes two bugs in the original pe_sampling.py extraction:
  1. Double prior_transform: get_samples_with_weights already applies the
     transform internally; the original code applied it again.
  2. ESS formula: weights are unnormalized; correct formula is
     sum(w)^2 / sum(w^2), not 1/sum(w^2).

Overwrites each pe_results.npz in-place with corrected values.
No GPU needed — reads only stored sampler state.
"""

import sys
import os
import pickle

import numpy as np

# pe_sampling must be importable so that the pickled function references
# (__main__._log_density, __main__._prior_transform) resolve correctly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe_sampling

import __main__
__main__._log_density    = pe_sampling._log_density
__main__._prior_transform = pe_sampling._prior_transform

RESULTS_BASE = '/scratch/josh.mat/opt_grid/pe_results'

RUNS = [
    'EMRI_0pa_pt00',
    'EMRI_1pa_pt00',
    'IMRI_0pa_pt00',
    'IMRI_1pa_pt00',
    'IMRI_TAIL_0pa_pt00',
    'IMRI_TAIL_1pa_pt00',
]

for run in RUNS:
    base = os.path.join(RESULTS_BASE, run)
    npz_path = os.path.join(base, 'pe_results.npz')
    pkl_path = os.path.join(base, 'sampler_state.pkl')

    print(f'\n=== {run} ===')

    if not os.path.exists(pkl_path):
        print(f'  [SKIP] no sampler_state.pkl')
        continue

    # Load bounds from the saved npz (these were stored correctly)
    orig = np.load(npz_path, allow_pickle=True)
    bounds_lo = orig['bounds_lo']
    bounds_hi = orig['bounds_hi']
    span      = bounds_hi - bounds_lo

    # Set module-level globals so _prior_transform works when called by PARIS
    pe_sampling._PE_BOUNDS_LO = bounds_lo
    pe_sampling._PE_SPAN      = span

    # Unpickle sampler and extract samples (prior_transform applied internally)
    with open(pkl_path, 'rb') as f:
        sampler = pickle.load(f)

    samples, weights = sampler.get_samples_with_weights(flatten=True)

    # Verify samples are in the expected physical range
    for i, p in enumerate(orig['param_names']):
        lo, hi = bounds_lo[i], bounds_hi[i]
        frac_out = np.mean((samples[:, i] < lo) | (samples[:, i] > hi))
        if frac_out > 0.01:
            print(f'  [WARN] {p}: {100*frac_out:.1f}% samples outside prior bounds')

    # ESS with unnormalized importance weights
    ess = float(np.sum(weights) ** 2 / np.sum(weights ** 2))

    w_norm = weights / weights.sum()
    weighted_mean = np.average(samples, weights=w_norm, axis=0)
    weighted_cov  = np.cov(samples.T, aweights=w_norm)

    samples_unit = np.clip((samples - bounds_lo) / span, 0.0, 1.0)

    param_names  = orig['param_names']
    signal_theta = orig['signal_theta']
    mle_theta    = orig['mle_theta']
    sigma_fisher = orig['sigma_fisher']

    np.savez(
        npz_path,
        samples=samples,
        samples_unit=samples_unit,
        weights=weights,
        weighted_mean=weighted_mean,
        weighted_cov=weighted_cov,
        param_names=param_names,
        signal_theta=signal_theta,
        mle_theta=mle_theta,
        sigma_fisher=sigma_fisher,
        bounds_lo=bounds_lo,
        bounds_hi=bounds_hi,
        ess=ess,
    )

    param_labels = [p for p in ['m1','m2','a','p0','e0','chi2'] if p in list(param_names)]
    print(f'  n_samples={len(samples)}  ESS={ess:.1f}')
    print(f'  Fisher vs PE marginal sigmas:')
    for j, p in enumerate(param_labels):
        idx = list(param_names).index(p)
        pe_sig   = float(np.sqrt(weighted_cov[idx, idx]))
        fish_sig = float(sigma_fisher[j])
        print(f'    {p:6s}: Fisher={fish_sig:.4g}  PE={pe_sig:.4g}  ratio={pe_sig/max(fish_sig,1e-30):.3f}')

    print(f'  Saved to {npz_path}')

print('\nDone.')
