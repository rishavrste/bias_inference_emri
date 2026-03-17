import os
import json
import time
import argparse
from typing import Tuple, Optional
from scipy.signal.windows import tukey
import numpy as np
import cupy as cp
# FEW / waveform & noise
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens
from stableemrifisher.utils import generate_PSD, inner_product
from stableemrifisher.fisher import StableEMRIFisher
from stableemrifisher.utils import generate_PSD, inner_product

def _is_pos_def(mat: np.ndarray) -> bool:
    """Return True if matrix is positive-definite via Cholesky test."""
    try:
        np.linalg.cholesky(mat)
        return True
    except np.linalg.LinAlgError:
        return False

def check_and_clip_prior(priors_range, param_names):
    """Check that prior ranges are consistent with physical bounds and reference values.
    Make sure the param_names are decorated with the same pattern as in the reference dict (e.g., 'm1', 'm2', 'a', etc.) for proper checking."""

    # --- Physical bounds ---
    EMRI_param_ranges = {
        "m1": (0, None),
        "m2": (0, None),
        "a": (-0.999, 0.999),
        "p0": (0, None),
        "e0": (0, 1),
        "xI0": (-1, 1),
        "dist": (0, None),
        "qS": (-np.pi, np.pi),
        "phiS": (-np.pi, np.pi),
        "qK": (-np.pi, np.pi),
        "phiK": (-np.pi, np.pi),
        "Phi_phi0": (-np.pi, np.pi),
        "Phi_theta0": (-np.pi, np.pi),
        "Phi_r0": (-np.pi, np.pi),
    }

    # --- Angular width limits ---
    angular_width_limits = {
        "phiS": 2*np.pi,
        "phiK": 2*np.pi,
        "Phi_phi0": 2*np.pi,
        "Phi_theta0": 2*np.pi,
        "Phi_r0": 2*np.pi,
        "qS": 2*np.pi,
        "qK": 2*np.pi,
    }

    checked_ranges = []

    for i, base_name in enumerate(param_names):
        low, high = priors_range[i]

        # Enforcing physical bounds without circular bound
        if base_name in EMRI_param_ranges:
            ref_low, ref_high = EMRI_param_ranges[base_name]

            if base_name not in angular_width_limits:
                if ref_low is not None:
                    low = max(low, ref_low)
                if ref_high is not None:
                    high = min(high, ref_high)

        # Angular Widths
        if base_name in angular_width_limits:
            max_width = angular_width_limits[base_name]
            width = high - low

            if width > max_width:
                center = 0.5 * (low + high)
                low = center - 0.5 * max_width
                high = center + 0.5 * max_width

        checked_ranges.append([low, high])
    return checked_ranges


def _clip_physical_params_intrinsic(theta: np.ndarray) -> np.ndarray:
    """Clip mapped physical parameters to minimal physical ranges.

    Indices: 0:m1, 1:m2, 2:a, 3:p0, 4:e0, [5:chi2]
    - m1, m2 > 0
    - a in [-0.9999, 0.9999]
    - e0 in (0, 1)
    - chi2 in [-0.99, 0.99] (only when pa_template == '1PA')
    """
    x = np.asarray(theta, dtype=float).copy()
    if x.ndim == 1:
        if x.shape[0] >= 1:
            x[0] = max(x[0], 1e-30)
        if x.shape[0] >= 2:
            x[1] = max(x[1], 1e-30)
        if x.shape[0] >= 3:
            x[2] = np.clip(x[2], -0.9999, 0.9999)
        if x.shape[0] >= 5:
            x[4] = np.clip(x[4], 1e-8, 1 - 1e-8)
   
        return x
    else:
        x[:, 0] = np.maximum(x[:, 0], 1e-30)
        x[:, 1] = np.maximum(x[:, 1], 1e-30)
        if x.shape[1] >= 3:
            x[:, 2] = np.clip(x[:, 2], -0.9999, 0.9999)
        if x.shape[1] >= 5:
            x[:, 4] = np.clip(x[:, 4], 1e-8, 1 - 1e-8)
        return x
    

def compute_fisher_parallelotope(build_waveform_response,
                                 ctx: dict,
                                 fisher_params: list,
                                 params_to_infer: list,
                                 additional_kwargs: dict,
                                 use_gpu: bool = True,
                                 _TARGET_SNR: float = 20.0,
                                 prior_sigma_range: float = 20,
                                 using_evec: bool = False) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Build Fisher-based local prior around ``theta0``.

    When ``using_evec`` is True we recover the original Fisher ellipsoid (axes
    given by covariance eigenvectors). When False we form a simple axis-aligned
    box whose half-widths are ``prior_sigma_range * sqrt(diag(F^{-1}))``.

    Returns ``(Q, b, meta)`` where ``Q`` is the basis matrix (orthonormal when
    ``using_evec`` is True, identity otherwise), ``b`` holds the half-lengths,
    and ``meta`` carries diagnostics including ``using_evec``.
    """
    # Prepare waveform generator
    dim = len(params_to_infer)
    if 'waveform_response' in ctx and ctx['waveform_response'] is not None:
        waveform_response = ctx['waveform_response']
    else:
        waveform_response = build_waveform_response(T=ctx['T'], dt=ctx['dt'], use_gpu=use_gpu)

    channels = [A1TDISens, E1TDISens, T1TDISens]
    noise_kwargs = [{"sens_fn": ch} for ch in channels]

    # Parameter names and flags (use 5D derivatives as in old_method; chi2 fixed via add_param_args when 1PA)
    param_names = params_to_infer

    add_param_args =  additional_kwargs
    sef_kwargs = {
        'EMRI_waveform_gen': waveform_response,
        'param_names': param_names,
        'der_order': 6,
        'Ndelta': 12,
        'use_gpu': bool(use_gpu),
        'noise_model': get_sensitivity,
        'channels': channels,
        'noise_kwargs': noise_kwargs,
        'add_param_args': add_param_args,
        'deltas': None,
    }
    # Fisher parameter vector (only 5 core dims positionally)
    fisher_params = fisher_params

    # Initialize SEF and compute Fisher
    try:
        # Follow old_method.py style: pass T, dt as keyword args
        sef = StableEMRIFisher(*fisher_params, T=float(ctx['T']), dt=float(ctx['dt']), **sef_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"StableEMRIFisher init failed with kwargs T,dt: {e} | "
            f"len(fisher_params)={len(fisher_params)}, param_names={param_names}"
        )
    try:
        # Initialize internal PSD/noise and window via SEF API
        sef.SNRcalc_SEF()
        if not hasattr(sef, 'deltas') or sef.deltas is None:
            sef.Fisher_Stability()
        F = np.asarray(sef.FisherCalc(), dtype=float)
        print(f"[FISHER] {repr(F)}")
        print(f"[FISHER_IS_PD] {_is_pos_def(F)}")
        F_inv = np.linalg.inv(F)
        print(f"[FISHER_INV] {repr(F_inv)}")
        print(f"[FISHER_INV_IS_PD] {_is_pos_def(F_inv)}")
        F_std = np.sqrt(np.diag(F_inv))
        print(f"[FISHER_STD] {repr(F_std)}")
    except Exception as e:
        raise RuntimeError(f"Fisher computation failed: {e}")

    emri_flags = {"T": ctx['T'], "dt": ctx['dt'], '1PA': False, 'evolve_primary': False, '2PA': False}

    waveform_tmpl = waveform_response(*fisher_params, **emri_flags)
    PSD_funcs = generate_PSD(waveform=waveform_tmpl, dt=float(ctx['dt']), noise_PSD=get_sensitivity,
                             channels=channels, noise_kwargs=noise_kwargs, use_gpu=bool(use_gpu))
    snr_model = float(np.sqrt(inner_product(waveform_tmpl, waveform_tmpl, PSD_funcs, float(ctx['dt']), use_gpu=bool(use_gpu))))

    # Scale Fisher to target SNR
    scale = (_TARGET_SNR / max(snr_model, 1e-30)) ** 2
    print('F: scale',scale)

    if not using_evec:
        sigma_diag = F_std
        print(f"[DIAG_STD] {repr(sigma_diag)}")
        eigvals = sigma_diag ** 2
        print(f"[DIAG_EIGVALS] {repr(eigvals)}")
        b = prior_sigma_range * sigma_diag
        print(f"[DIAG_B] {repr(b)}")
        meta = {
            'diag_sigma': sigma_diag.tolist(),
            'eigvals': eigvals.tolist(),
            'snr_model': float(snr_model),
            'scale_applied': float(scale),
            'using_evec': False,
        }
        return np.eye(dim), b, meta

    F_scaled = F * scale

    # Regularize and invert to covariance
    dim = F_scaled.shape[0]
    reg = 1e-20 * np.trace(F_scaled) / max(dim, 1)       
    F_scaled = F_scaled + reg * np.eye(dim)     
    print(f"[F_SCALED] {repr(F_scaled)}")
    print(f"[F_SCALED_IS_PD] {_is_pos_def(F_scaled)}")
    try:
        cov = np.linalg.inv(F_scaled)
    except np.linalg.LinAlgError:
        print("[WARN] F_scaled inversion failed; using pseudoinverse")
        cov = np.linalg.pinv(F_scaled)
    print(f"[COV_RAW] {repr(cov)}")
    print(f"[COV_RAW_IS_PD] {_is_pos_def(cov)}")
    cov = 0.5 * (cov + cov.T)
    print(f"[COV_SYM] {repr(cov)}")
    print(f"[COV_SYM_IS_PD] {_is_pos_def(cov)}")
    cov_diag_std = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    print(f"[COV_STD] {repr(cov_diag_std)}")
    evals_raw, evecs = np.linalg.eigh(cov)
    print(f"[COV_EVALS_RAW] {repr(evals_raw)}")
    print(f"[COV_EVECS] {repr(evecs)}")
    eigvals_eigvalsh = np.linalg.eigvalsh(cov)
    print(f"[COV_EIGVALSH] {repr(eigvals_eigvalsh)}")
    evals = evals_raw
    print(f"[COV_EVALS] {repr(evals)}")
    sigma_diag = cov_diag_std
    print(f"[COV_SIGMA_DIAG] {repr(sigma_diag)}")
    b = prior_sigma_range * np.sqrt(evals)
    print(f"[FISHER_EVALS] {repr(evals)}")
    print(f"[FISHER_B] {repr(b)}")

    meta = {
        'diag_sigma': sigma_diag.tolist(),
        'eigvals': evals.tolist(),
        'snr_model': float(snr_model),
        'scale_applied': float(scale),
        'using_evec': True,
        #'reg_added': float(reg),
    }
    return evecs, b, meta

def covariance_from_fisher_parallelotope(Q: np.ndarray, b: np.ndarray, prior_sigma_range: float = 20) -> np.ndarray:
    """Construct covariance matrix from Fisher-parallelotope outputs (Q, b).

    Given that b = 10 * sqrt(eigvals), we recover eigvals = (b/10)^2 and return Q diag(eigvals) Q^T.
    """
    b = np.asarray(b, dtype=float)
    eigvals = (b / prior_sigma_range) ** 2
    cov = Q @ (np.diag(eigvals)) @ Q.T
    cov = 0.5 * (cov + cov.T)
    return cov


def compute_fft_with_windowing(waveform, dt, N,use_gpu=False,n_channels=3):
    if use_gpu:
        try:
            xp = cp
        except ImportError:
            print("[WARN] CuPy not available, falling back to NumPy")
            xp = np
    else:
        xp = np

    window = xp.asarray(tukey(N, 0.01))
    waveform_windowed = waveform * window
    waveform_f = xp.asarray([xp.fft.rfft(waveform_windowed[i]) * dt for i in range(n_channels)])
    return waveform_f

def inner_prod(signal_1_f, signal_2_f, PSD, delta_f, xp=np):
    """
    Compute noise-weighted inner product using BBHx's standard normalization.

    Uses: ⟨a|b⟩ = 4·Δf·Re[Σ a(f)·b*(f) / Sn(f)]

    Parameters
    ----------
    signal_1_f : array-like
        First signal in frequency domain
    signal_2_f : array-like
        Second signal in frequency domain
    PSD : array-like
        Power spectral density Sn(f)
    delta_f : float
        Frequency spacing (Hz)
    xp : module, optional
        Array module (numpy or cupy). Default: numpy

    Returns
    -------
    float
        Inner product value ⟨signal_1|signal_2⟩
    """
    return 4 * delta_f * xp.real(xp.sum(signal_1_f * signal_2_f.conj() / PSD))

def inner_prod(signal_1_f, signal_2_f, PSD, delta_f, xp=np):
    """
    Compute noise-weighted inner product using BBHx's standard normalization.

    Uses: ⟨a|b⟩ = 4·Δf·Re[Σ a(f)·b*(f) / Sn(f)]

    Parameters
    ----------
    signal_1_f : array-like
        First signal in frequency domain
    signal_2_f : array-like
        Second signal in frequency domain
    PSD : array-like
        Power spectral density Sn(f)
    delta_f : float
        Frequency spacing (Hz)
    xp : module, optional
        Array module (numpy or cupy). Default: numpy

    Returns
    -------
    float
        Inner product value ⟨signal_1|signal_2⟩
    """
    return 4 * delta_f * xp.real(xp.sum(signal_1_f * signal_2_f.conj() / PSD))

def inner_prod_without_phase(signal_1_f, signal_2_f, PSD, delta_f, xp=np):
    return 4 * delta_f * xp.abs(xp.sum(signal_1_f * signal_2_f.conj() / PSD))

def timemax_correlation(h1, h2,dt, PSD, xp=np):

    # FFT with dt scaling
    H1 = xp.array([xp.fft.rfft(h1[k]) * dt for k in range(2)])
    H2 = xp.array([xp.fft.rfft(h2[k]) * dt for k in range(2)])
    # print("H1 shape: ", H1.shape
    #       ,"H2 shape: ", H2.shape)
    # print("PSD shape: ", PSD.shape)

    Y = xp.zeros_like(H1)
    for i in range(2):
        Y[i,1:] = H1[i,1:] * xp.conj(H2[i,1:]) / (0.5 * PSD[i])  # Avoid DC component
    # IFFT to time domain with proper normalization
    S =xp.array([xp.fft.irfft(Y[i]) / dt for i in range(2)])
    # Return maximum correlation
    return  xp.max(xp.abs(S))

#use detection SNR and also use max phase if phase_max is True
def calculate_detection_snr_0pa_vs_1pa(m1, m2, a, p0, e0, Y0, dist, qS,phiS, qK, phiK, 
                    Phi_phi0, Phi_theta0, Phi_r0,add_kwargs,
                    maximize_phase=False,
                    **fixed):
    xp = cp if fixed['use_gpu'] else np
    waveform_response = fixed['waveform_response']
    wave_params = [m1, m2, a, p0, e0, Y0, dist, qS,phiS, qK, phiK, 
                    Phi_phi0, Phi_theta0, Phi_r0,add_kwargs['chi2'],add_kwargs['evolve_1PA'],add_kwargs['evolve_primary'],
                    add_kwargs['evolve_2PA'], add_kwargs['deviation_included'],add_kwargs['dev_1'],add_kwargs['dev_2']]
    emri_kwargs =  {"T": fixed['T'], "dt": fixed['dt'],'chi2': add_kwargs['chi2'],'evolve_1PA': add_kwargs['evolve_1PA'],
                    'evolve_primary': add_kwargs['evolve_primary'],'evolve_2PA': add_kwargs['evolve_2PA'],'deviation_included': add_kwargs['deviation_included'],
               'dev_1': add_kwargs['dev_1'], 'dev_2': add_kwargs['dev_2']}
    h = waveform_response(*wave_params, **emri_kwargs)
    PSD = fixed['PSD']
    h_f = compute_fft_with_windowing(h, fixed['dt'], fixed['N'], use_gpu=fixed['use_gpu'], n_channels=3)
    optimal_snr = xp.sqrt(inner_prod(h_f, h_f, PSD, fixed['delta_f'], xp=np))


    if (maximize_phase):
        num = inner_prod_without_phase(h, h, PSD, fixed['dt'], window=None, fmin=None, fmax=None, use_gpu=fixed['use_gpu'])
    else:
        num = inner_prod(fixed['waveform_true_fft'], h_f, PSD, fixed['delta_f'], xp=np)
    snr = num / optimal_snr
    return float(snr)

def load_startingpoint_param_array():
    pass

def generate_colored_noise(PSD,delta_f,delta_t):
    pass