import os
import json
import time
import argparse
from typing import Tuple, Optional
import re
from unittest import case

import numpy as np
from scipy.optimize import minimize

#few and SEF imports
from few.waveform import GenerateEMRIWaveform
from few.waveform.waveform import SuperKludgeWaveform

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import KerrEccEqFlux
from few.utils.geodesic import get_fundamental_frequencies
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
from few.utils.constants import MTSUN_SI

from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity, A2TDISens, E2TDISens, T2TDISens
from stableemrifisher.utils import generate_PSD, inner_product
from stableemrifisher.fisher import StableEMRIFisher
import matplotlib.pyplot as plt

from config_paris import Config, ObjectiveTracker
from misc import *
import parismc
try:
    import cupy as cp
    xp=cp
except:
    xp=np
    print("CuPy not found, using NumPy instead. For GPU acceleration, please install CuPy.")
# -----------------------------
# PARIS global context (picklable functions require module scope)
# -----------------------------
_PARIS_REF_CENTER = None          # type: Optional[np.ndarray]
_PARIS_SPREAD_SCALE = None        # type: Optional[float]
_PARIS_OBJECTIVE = None           # type: Optional[callable]
_PARIS_TARGET_KIND = None         # type: Optional[str]  # 'optimal_snr', 'optimal_snr_phase_max', 'phase_match', 'time_max'
_PARIS_TEMPERATURE = 1.0          # type: float  # divide score by this to flatten landscape (>1 = more exploratory)
_TARGET_SNR = None
# Fisher-parallelotope affine prior (primary for this script)
_PARIS_AFFINE_CENTER = None       # type: Optional[np.ndarray]
_PARIS_AFFINE_Q = None            # type: Optional[np.ndarray]
_PARIS_AFFINE_B = None            # type: Optional[np.ndarray]
_PARIS_DIM = None                 # type: Optional[int]
_PARIS_USE_ELLIPSE = True

cfg = Config()
# Target SNR for Fisher scaling
base_dir = cfg.basedir

def paris_prior_transform(u):
    """Prior transform using Fisher-parallelotope when configured.

    If affine parameters are set: theta = center + Q @ (b * t), with t = 2u-1.
    Otherwise falls back to multiplicative band around _PARIS_REF_CENTER.
    """
    u = np.asarray(u, dtype=float)
    if _PARIS_AFFINE_CENTER is not None and _PARIS_AFFINE_Q is not None and _PARIS_AFFINE_B is not None:
        center = _PARIS_AFFINE_CENTER
        Q = _PARIS_AFFINE_Q
        b = _PARIS_AFFINE_B
        dim = Q.shape[0]

        def map_one(u1):
            t = 2.0 * np.asarray(u1)[:dim] - 1.0
            return center + Q @ (b * t)

        if u.ndim == 1:
            theta = map_one(u)
            # if theta.shape[0] == 5:
            return _clip_physical_params_intrinsic(theta)
            # else:
            #     return check_and_clip_prior(theta, params_to_infer)
        else:
            out = np.zeros((u.shape[0], dim), dtype=float)
            for i in range(u.shape[0]):
                out[i] = map_one(u[i])
            return _clip_physical_params_intrinsic(out)
            # else:
            #     return check_and_clip_prior(out, params_to_infer)

    # Legacy multiplicative band (fallback)
    ref = _PARIS_REF_CENTER
    s = _PARIS_SPREAD_SCALE
    u = np.asarray(u)
    return ref * (1 - s + u * 2 * s)


def paris_inverse_prior_transform(params):
    """Inverse of paris_prior_transform.

    For affine Fisher mapping: t = diag(1/b) Q^T (theta - center), u = 0.5*(t+1).
    Falls back to multiplicative inverse if affine not configured.
    """
    theta = np.asarray(params, dtype=float)
    if _PARIS_AFFINE_CENTER is not None and _PARIS_AFFINE_Q is not None and _PARIS_AFFINE_B is not None:
        center = _PARIS_AFFINE_CENTER
        Q = _PARIS_AFFINE_Q
        b = _PARIS_AFFINE_B
        inv_b = 1.0 / b

        def inv_one(th):
            d = np.asarray(th) - center
            t = (Q.T @ d) * inv_b
            return 0.5 * (t + 1.0)

        if theta.ndim == 1:
            return inv_one(theta)
        out = np.zeros_like(theta, dtype=float)
        for i in range(theta.shape[0]):
            out[i] = inv_one(theta[i])
        return out

    ref = _PARIS_REF_CENTER
    s = _PARIS_SPREAD_SCALE
    return (theta / ref - (1 - s)) / (2 * s)


def paris_log_density(params):
    """Top-level log-density (score) wrapper without early-stop.

    - Receives physical parameters (after prior_transform) per parismc contract.
    - Returns a scalar/array of scores (larger is better).
    """
    params = np.asarray(params)

    def eval_one(x):
        try:
            val = float(_PARIS_OBJECTIVE(x)) / _PARIS_TEMPERATURE
        except Exception:
            return float('-inf')
        return val

    if params.ndim == 1:
        return eval_one(params)

    out = np.zeros(params.shape[0], dtype=float)
    for i in range(params.shape[0]):
        out[i] = eval_one(params[i])
    return out

def build_waveform_response(T: float, dt: float, use_gpu: bool = False) -> ResponseWrapper:
    """Create a LISA ResponseWrapper consistent with existing modules."""

    sum_kwargs = dict(pad_output=False, odd_len=True)
    
    waveform_model = GenerateEMRIWaveform(SuperKludgeWaveform, sum_kwargs=sum_kwargs, return_list=False,use_gpu=use_gpu)

    t0 = 10000.0
    tdi_gen = "2nd generation"
    order = 20
    index_lambda = 8  # phiS
    index_beta = 7    # qS

    response = ResponseWrapper(
        waveform_gen=waveform_model,
        Tobs=T,
        t0=t0,
        dt=dt,
        index_lambda=index_lambda,
        index_beta=index_beta,
        flip_hx=True,
        is_ecliptic_latitude=False,
        remove_garbage="zero",
        orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
        force_backend = "cuda12x" if use_gpu else "cpu",
        order=order,
        tdi=tdi_gen,
        tdi_chan="AET")

    print("[INFO] Finished loading modules and building ResponseWrapper")
    return response

def prepare_true_waveform(signal_row: np.ndarray, emri_kwargs: dict, add_kwargs: dict,add_noise: bool=False, use_gpu: bool = False,seed: Optional[int] = 0,nchannels: int = 3) -> dict:
    """
    Build fiducial 2PA waveform, PSD, and FFT from a signal parameter row.
    signal_row columns:
      [m1, m2, a, p0, e0, Y0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
    """
    (
        m1, m2, a, p0, e0, Y0,
        dist, qS, phiS, qK, phiK,
        Phi_phi0, Phi_theta0, Phi_r0
    ) = signal_row

    waveform_response = build_waveform_response(T=emri_kwargs['T'], dt=emri_kwargs['dt'], use_gpu=use_gpu)

    chi2 = add_kwargs.get('chi2')
    add_kwargs['evolve_1PA'] = True
    add_kwargs['evolve_2PA'] = True
    evolve_1PA = add_kwargs.get('evolve_1PA',True)
    evolve_primary = add_kwargs.get('evolve_primary', False)
    evolve_2PA = add_kwargs.get('evolve_2PA',True)
    dt = emri_kwargs['dt']
    T=emri_kwargs['T']
    noise= None

    wave_params = [
        m1, m2, a, p0, e0, Y0,
        dist, qS, phiS, qK, phiK,
        Phi_phi0, Phi_theta0, Phi_r0,chi2, evolve_1PA, evolve_primary, evolve_2PA
    ]

    waveform_true = xp.array(waveform_response(*wave_params, **emri_kwargs))[0:nchannels,:]  # Shape (3, N) for A, E, T channels
    print("[INFO] Finished generating true waveform")
    
    channels = [A2TDISens, E2TDISens, T2TDISens]
    if nchannels == 3:
        noise_kwargs = [{"sens_fn": ch} for ch in channels]
    elif nchannels == 2:
        noise_kwargs = [{"sens_fn": ch} for ch in channels[:2]]
    else:
        raise ValueError(f"Unsupported number of channels: {nchannels}. Only 2 (A,E) or 3 (A,E,T) are supported.")
    PSD_funcs = generate_PSD(
        waveform=waveform_true,
        dt=dt,
        noise_PSD=get_sensitivity,
        channels=channels,
        noise_kwargs=noise_kwargs,
        use_gpu=use_gpu,
    )

    # Verify SNR level (grid builder normalized dist to target PA2 SNR already)
    PSD_funcs_ = xp.array(PSD_funcs)

    # print("type check - waveform_true:", type(waveform_true), "PSD_funcs:", type(PSD_funcs_), "dt:", type(dt))
    print("shape check - waveform_true:", xp.shape(waveform_true), "PSD_funcs:", xp.shape(PSD_funcs_))
    snr_2 = inner_product(waveform_true, waveform_true, PSD_funcs_, dt, use_gpu=use_gpu)
    #if hasattribute get u
    snr = np.sqrt(snr_2.get()) if hasattr(snr_2, "get") else np.sqrt(snr_2)
    print(f"[TRUE] SNR: {snr:.6f}")     
    for i in range(PSD_funcs_.shape[0]):    
       #check snr of each channel
         snr_i_2 = inner_product(waveform_true[i:i+1,], waveform_true[i:i+1,], PSD_funcs_[i:i+1,:], dt, use_gpu=use_gpu)
         snr_i = np.sqrt(snr_i_2.get()) if hasattr(snr_i_2, "get") else np.sqrt(snr_i_2)
         print(f"[TRUE] SNR - Channel {i}: {snr_i:.6f}")
    
         
    N_fiducial = len(waveform_true[0])
    freq = np.fft.rfftfreq(N_fiducial, dt)
    delta_f = freq[1] - freq[0]
    print(f"[INFO] Frequency resolution delta_f: {delta_f:.6e} Hz, Number of frequency bins: {len(freq)}")
    print(f"freq_max: {freq[-1]:.6f} Hz and Freq_min: {freq[1]:.6f} Hz")

    if add_noise:
        waveform_true_fft_without_noise = compute_fft_with_windowing(waveform_true, dt, N_fiducial, use_gpu=use_gpu, n_channels=nchannels)
        plot_time_series_from_fft(waveform_true_fft_without_noise, dt, title="True Waveform without Noise (Time Domain)")
        waveform_true_fft,noise = add_noise_func(waveform_true_fft_without_noise,PSD_funcs_,delta_f, dt,n_channels= nchannels,seed=cfg.seed)
        print("[INFO] Added noise to true waveform FFT\n")
        print("shape of waveform_true_fft after noise addition:", xp.shape(waveform_true_fft))
        plot_time_series_from_fft(waveform_true_fft, dt, title="True Waveform with Noise (Time Domain)")
        noise_dict = check_noise_model_consistency(PSD_funcs_,delta_f,dt,n_channels=nchannels,temp_signal=waveform_true,seed=seed)
        print(f"Noise consistency check results: {noise_dict}")
        
    else:
        print("[INFO] No noise added to true waveform FFT\n")
        waveform_true_fft = compute_fft_with_windowing(waveform_true, dt, N_fiducial, use_gpu=use_gpu, n_channels=nchannels)
        waveform_true_fft_without_noise = xp.copy(waveform_true_fft)  # Keep a copy of the clean FFT for later use
    
    print("[INFO] Finished preparing true waveform (GPU)")
    

    return {
        'm1': m1, 'm2': m2, 'a': a, 'p0': p0, 'e0': e0, 'Y0': Y0,
        'dist': dist, 'qS': qS, 'phiS': phiS, 'qK': qK, 'phiK': phiK,
        'Phi_phi0': Phi_phi0, 'Phi_theta0': Phi_theta0, 'Phi_r0': Phi_r0,
        'dt': dt, 'T': T, 'chi2': chi2,
        'waveform_response': waveform_response,
        'PSD_funcs': PSD_funcs_,
        'waveform_true_fft': waveform_true_fft,
        'waveform_true_fft_without_noise': waveform_true_fft_without_noise,
        'N_fiducial': N_fiducial,
        'snr': snr,
        'delta_f': freq[1]-freq[0],
        'freq': freq,
        'noise_dict': noise_dict if add_noise else None,
        'noise': noise if add_noise else None,
    }



def objective_factory(target_func: str,
                      ctx: dict,
                      phase_max: bool = False,
                      use_gpu_for_snr: bool = True,
                      use_1PA:bool = False,
                      with_phase: bool = False,
                      add_kwargs: dict = None) -> callable:
    """
    Build a score(theta) where larger is better for all targets.
    - 'optimal_snr' and 'optimal_snr_phase_max': score = optimal SNR (maximize)
    - 'phase_match': score = -phase_diff_metric (maximize score => minimize phase diff)
    """
    # Only needed for SNR-based objective
    if target_func in ('optimal_snr', 'optimal_snr_phase_max','time_max','chi2_match'):
        fixed = {
            'waveform_response': ctx['waveform_response'],
            'PSD': ctx['PSD_funcs'],
            'dt': ctx['dt'],
            'T': ctx['T'],
            'N_fiducial': ctx['N_fiducial'],
            'waveform_true_fft': ctx['waveform_true_fft'],
            'xp': np,'delta_f': ctx['delta_f'],
            'use_gpu': bool(use_gpu_for_snr),
            'nchannels': ctx['waveform_true_fft'].shape[0],
        }
    nchannels = ctx['waveform_true_fft'].shape[0]
    def score_optimal_snr(theta: np.ndarray) -> float:
        if with_phase == False:
            if not use_1PA:
                m1, m2, a, p0, e0 = theta
                add_kwargs['evolve_1PA'] = False
                add_kwargs['evolve_2PA'] = False
  
                val = calculate_detection_snr(
                    m1, m2, a, p0, e0, ctx['Y0'],ctx['dist'],ctx['qS'],ctx['phiS'], ctx['qK'], ctx['phiK'], 
                    ctx['Phi_phi0'], ctx['Phi_theta0'], ctx['Phi_r0'],add_kwargs,
                    maximize_phase=bool(phase_max),
                    **fixed,)
            else:
                m1,m2,a,p0,e0,chi2 = theta
                add_kwargs['evolve_1PA'] = True
                add_kwargs['evolve_2PA'] = False
                add_kwargs['chi2'] = chi2
                val = calculate_detection_snr(
                    m1, m2, a, p0, e0, ctx['Y0'],ctx['dist'],ctx['qS'],ctx['phiS'], ctx['qK'], ctx['phiK'], 
                    ctx['Phi_phi0'], ctx['Phi_theta0'], ctx['Phi_r0'],add_kwargs,
                    maximize_phase=bool(phase_max),
                    **fixed,)
            
                

        else:
            if not use_1PA:
                m1, m2, a, p0, e0,Phi_phi0,Phi_r0 = theta
                add_kwargs['evolve_1PA'] = False
                add_kwargs['evolve_2PA'] = False
           
                val = calculate_detection_snr(
                    m1, m2, a, p0, e0, ctx['Y0'],ctx['dist'],ctx['qS'],ctx['phiS'], ctx['qK'], ctx['phiK'], 
                    Phi_phi0, ctx['Phi_theta0'], Phi_r0,add_kwargs,
                    maximize_phase=bool(phase_max),
                    **fixed)
            else:
                m1, m2, a, p0, e0,Phi_phi0,Phi_r0,chi2 = theta
                add_kwargs['evolve_1PA'] = True
                add_kwargs['evolve_2PA'] = False
                add_kwargs['chi2'] = chi2
                val = calculate_detection_snr(
                    m1, m2, a, p0, e0, ctx['Y0'],ctx['dist'],ctx['qS'],ctx['phiS'], ctx['qK'], ctx['phiK'], 
                    Phi_phi0, ctx['Phi_theta0'], Phi_r0,add_kwargs,
                    maximize_phase=bool(phase_max),
                    **fixed,)

        return float(val)
    
    def score_time_max(theta: np.ndarray) -> float:
        if with_phase == False:
            if not use_1PA:
                m1, m2, a, p0, e0 = theta
                add_kwargs['evolve_1PA'] = False
                add_kwargs['evolve_2PA'] = False
  
                val = calculate_time_max(
                    m1, m2, a, p0, e0, ctx['Y0'],ctx['dist'],ctx['qS'],ctx['phiS'], ctx['qK'], ctx['phiK'], 
                    ctx['Phi_phi0'], ctx['Phi_theta0'], ctx['Phi_r0'],add_kwargs,
                    maximize_phase=bool(phase_max),
                    **fixed,)
            else:
                m1,m2,a,p0,e0,chi2 = theta
                add_kwargs['evolve_1PA'] = True
                add_kwargs['evolve_2PA'] = False
                add_kwargs['chi2'] = chi2
                val = calculate_time_max(
                    m1, m2, a, p0, e0, ctx['Y0'],ctx['dist'],ctx['qS'],ctx['phiS'], ctx['qK'], ctx['phiK'], 
                    ctx['Phi_phi0'], ctx['Phi_theta0'], ctx['Phi_r0'],add_kwargs,
                    maximize_phase=bool(phase_max),
                    **fixed,)
            
                

        else:
            if not use_1PA:
                m1, m2, a, p0, e0,Phi_phi0,Phi_r0 = theta
                add_kwargs['evolve_1PA'] = False
                add_kwargs['evolve_2PA'] = False
           
                val = calculate_time_max(
                    m1, m2, a, p0, e0, ctx['Y0'],ctx['dist'],ctx['qS'],ctx['phiS'], ctx['qK'], ctx['phiK'], 
                    Phi_phi0, ctx['Phi_theta0'], Phi_r0,add_kwargs,
                    maximize_phase=bool(phase_max),
                    **fixed)
            else:
                m1, m2, a, p0, e0,Phi_phi0,Phi_r0,chi2 = theta
                add_kwargs['evolve_1PA'] = True
                add_kwargs['evolve_2PA'] = False
                add_kwargs['chi2'] = chi2
                val = calculate_time_max(
                    m1, m2, a, p0, e0, ctx['Y0'],ctx['dist'],ctx['qS'],ctx['phiS'], ctx['qK'], ctx['phiK'], 
                    Phi_phi0, ctx['Phi_theta0'], Phi_r0,add_kwargs,
                    maximize_phase=bool(phase_max),
                    **fixed,)

        return float(val)
    
    def score_chi2_match(theta: np.ndarray) -> float:
        if with_phase == False:
            if not use_1PA:
                m1, m2, a, p0, e0 = theta
                add_kwargs['evolve_1PA'] = False
                add_kwargs['evolve_2PA'] = False
  
                val = chi2_match(
                    m1, m2, a, p0, e0, ctx['Y0'],ctx['dist'],ctx['qS'],ctx['phiS'], ctx['qK'], ctx['phiK'], 
                    ctx['Phi_phi0'], ctx['Phi_theta0'], ctx['Phi_r0'],add_kwargs,
                    maximize_phase=bool(phase_max),
                    **fixed,)
            else:
                m1,m2,a,p0,e0,chi2 = theta
                add_kwargs['evolve_1PA'] = True
                add_kwargs['evolve_2PA'] = False
                add_kwargs['chi2'] = chi2
                val = chi2_match(
                    m1, m2, a, p0, e0, ctx['Y0'],ctx['dist'],ctx['qS'],ctx['phiS'], ctx['qK'], ctx['phiK'], 
                    ctx['Phi_phi0'], ctx['Phi_theta0'], ctx['Phi_r0'],add_kwargs,
                    maximize_phase=bool(phase_max),
                    **fixed,)
            
                

        else:
            if not use_1PA:
                m1, m2, a, p0, e0,Phi_phi0,Phi_r0 = theta
                add_kwargs['evolve_1PA'] = False
                add_kwargs['evolve_2PA'] = False
  
                val = chi2_match(
                    m1, m2, a, p0, e0, ctx['Y0'],ctx['dist'],ctx['qS'],ctx['phiS'], ctx['qK'], ctx['phiK'], 
                    Phi_phi0, ctx['Phi_theta0'], Phi_r0,add_kwargs,
                    maximize_phase=bool(phase_max),
                    **fixed)
            else:
                m1, m2, a, p0, e0,Phi_phi0,Phi_r0,chi2 = theta
                add_kwargs['evolve_1PA'] = True
                add_kwargs['evolve_2PA'] = False
                add_kwargs['chi2'] = chi2
                val = chi2_match(
                    m1, m2, a, p0, e0, ctx['Y0'],ctx['dist'],ctx['qS'],ctx['phiS'], ctx['qK'], ctx['phiK'], 
                    Phi_phi0, ctx['Phi_theta0'], Phi_r0,add_kwargs,
                    maximize_phase=bool(phase_max),
                    **fixed,)

        return float(val)


    if target_func in ('optimal_snr', 'optimal_snr_phase_max'):
        return score_optimal_snr
    elif target_func == 'time_max':
        return score_time_max
    elif target_func == 'chi2_match':
        return score_chi2_match
    else:
        raise ValueError(f"Unknown target_func: {target_func}")
        

def nelder_mead_optimize(theta0: np.ndarray, objective, maxiter: int = 3000, xatol: float = 1e-10, fatol: float = 1e-12):
    res = minimize(
        objective,
        theta0,
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'maxfev': 15000, 'xatol': xatol, 'fatol': fatol,'adaptive': True},
    )
    return res

from scipy.optimize import differential_evolution
def differential_evolution_optimize(theta0: np.ndarray, objective, maxiter: int = 1000, tol: float = 1e-4, atol: float = 1e-5,x0: Optional[np.ndarray] = None,
                                    fisher_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,init='sobol',seed: Optional[int] = 42,
                                    popsize: int = 15, callback=None, workers: int = 1):
    from concurrent.futures import ThreadPoolExecutor
    if fisher_bounds is not None:
        bounds = fisher_bounds
    else:
        bounds = [(x*(1-1e-3), x*(1+1e-3)) for x in theta0]  # Define bounds around initial guess
    if theta0.shape[0] == 6 or theta0.shape[0] == 8:
        print("Earlier Bound are :", bounds)
        print("\nApplying special bounds for chi2\n")
        bounds[-1] = (-1, 1)
        print("Later Bound are :", bounds)
    de_kwargs = dict(
        func=objective,
        bounds=bounds,
        maxiter=maxiter,
        tol=tol,
        atol=atol,
        x0=theta0,
        seed=seed,
        init=init,
        popsize=popsize,
        callback=callback,
    )
    if workers == 1:
        res = differential_evolution(**de_kwargs)
    else:
        # Use ThreadPoolExecutor (threads share memory, no pickling of closures
        # needed, safe with CUDA contexts unlike multiprocessing fork).
        max_w = None if workers == -1 else workers
        print(f"[DE] Using ThreadPoolExecutor(max_workers={max_w}) for parallel population evaluation")
        with ThreadPoolExecutor(max_workers=max_w) as executor:
            res = differential_evolution(**de_kwargs, workers=executor.map)
    return res

def run_paris(ndim: int,
              prior_center: np.ndarray,
              score_func,
              spread_scale: float,
              savepath: str,
              seed_cloud: int = 10000,
              seed_jitter: float = 1e-10,
              paris_seed: int = 10,
              target_kind: str = None,
              lhs_save_dir: Optional[str] = None,
              affine_Q: Optional[np.ndarray] = None,
              affine_b: Optional[np.ndarray] = None,
              use_ellipse: bool = True):
    """Run PARIS sampler maximizing a score with a local prior around starting point.

    Notes on pickling:
    - parismc pickles the sampler, including log_density_func and prior_transform.
    - We therefore use top-level functions (paris_log_density/prior_transform) and
      set their behavior via module-level globals.
    """
    
    os.makedirs(savepath, exist_ok=True)

    # Configure global context for top-level callables
    global _PARIS_REF_CENTER, _PARIS_SPREAD_SCALE, _PARIS_OBJECTIVE, _PARIS_TARGET_KIND, _PARIS_EARLY_STOP_HIT
    global _PARIS_AFFINE_CENTER, _PARIS_AFFINE_Q, _PARIS_AFFINE_B, _PARIS_DIM
    global _PARIS_USE_ELLIPSE, _PARIS_TEMPERATURE
    _PARIS_REF_CENTER = np.asarray(prior_center, dtype=float).copy()
    _PARIS_SPREAD_SCALE = float(spread_scale)
    _PARIS_OBJECTIVE = score_func
    _PARIS_TARGET_KIND = target_kind
    _PARIS_EARLY_STOP_HIT = False
    _PARIS_USE_ELLIPSE = bool(use_ellipse)
    _PARIS_TEMPERATURE = float(paris_conf.get('paris_temperature', 1.0))

    # Configure Fisher-affine prior if provided
    if affine_Q is not None and affine_b is not None:
        _PARIS_AFFINE_CENTER = np.asarray(prior_center, dtype=float).copy()
        _PARIS_AFFINE_Q = np.asarray(affine_Q, dtype=float).copy()
        _PARIS_AFFINE_B = np.asarray(affine_b, dtype=float).copy()
        _PARIS_DIM = int(_PARIS_AFFINE_Q.shape[0])
    else:
        _PARIS_AFFINE_CENTER = None
        _PARIS_AFFINE_Q = None
        _PARIS_AFFINE_B = None
        _PARIS_DIM = None

    # Initialize sampler
    n_seed = paris_seed
    sigma = 1e-4
    init_cov_list = [sigma**2 * np.eye(ndim) for _ in range(n_seed)]
    config = parismc.SamplerConfig(
        merge_confidence=0.9,
        alpha=1000,
        #latest_prob_index=1000,
        trail_size=int(1e3),
        boundary_limiting=True,
        use_beta=True,
        integral_num=int(1e5),
        gamma=100,
        exclude_scale_z=np.inf,
        use_pool=False,
        cov_jitter=seed_jitter,  
      #  n_pool=36,
    )

    sampler = parismc.Sampler(
        ndim=ndim,
        n_seed=n_seed,
        log_density_func=paris_log_density,
        init_cov_list=init_cov_list,
        prior_transform=paris_prior_transform,
        config=config,
    )

    # Seed cloud in full unit cube; rely on Fisher-affine prior_transform for mapping
    unit_center = np.full(ndim, 0.5)
    center_val = float(paris_log_density(paris_prior_transform(unit_center.reshape(1, -1)))[0])

    point_blocks = [unit_center.reshape(1, -1)]
    log_blocks = [np.array([center_val])]

    n_samples = max(0, int(seed_cloud) - 1)
    if n_samples > 0:
        try:
            from smt.sampling_methods import LHS
        except ImportError as exc:
            raise RuntimeError(
                "smt.sampling_methods.LHS is required for PARIS seeding; install `smt` or ``pip install smt``"
            ) from exc

        xlimits = np.column_stack([
            np.zeros(ndim, dtype=float),
            np.ones(ndim, dtype=float),
        ])
        sampling = LHS(xlimits=xlimits)
        lhs_points = np.clip(sampling(n_samples), 0.0, 1.0)

        if lhs_points.ndim == 1:  # defensive: ensure (n, ndim)
            lhs_points = lhs_points.reshape(1, -1)

        # Require Fisher-affine mapping to be present
        if _PARIS_AFFINE_Q is None or _PARIS_AFFINE_B is None:
            raise RuntimeError("Fisher-affine prior is required for LHS seeding but is not set.")

        n_before = lhs_points.shape[0]
        if _PARIS_USE_ELLIPSE:
            # Fisher-ellipse truncation: keep points with ||t|| <= 1 where t = 2u-1
            t = 2.0 * lhs_points - 1.0
            keep_mask = np.sum(t * t, axis=1) <= 1.0
            lhs_points = lhs_points[keep_mask]
            n_after = lhs_points.shape[0]
            if n_after <= 0:
                raise RuntimeError("No LHS points remain after Fisher-ellipse truncation.")
            print(f"[LHS] before: {n_before}, after ellipse: {n_after}")
        else:
            print(f"[LHS] before: {n_before}, after ellipse: {n_before} (box)")

        print(f"[LHS] Sampled {lhs_points.shape[0]} points in unit cube; transforming to physical space with prior_transform")
        theta_points = paris_prior_transform(lhs_points)
        theta_min = np.min(theta_points, axis=0)
        theta_max = np.max(theta_points, axis=0)
        print(f"[LHS] theta_min: {repr(theta_min)}")
        print(f"[LHS] theta_max: {repr(theta_max)}")        


        lhs_vals = paris_log_density(paris_prior_transform(lhs_points))
        point_blocks.append(lhs_points)
        log_blocks.append(np.asarray(lhs_vals, dtype=float).reshape(-1))

    external_lhs_points = np.vstack(point_blocks)
    external_lhs_log_densities = np.concatenate(log_blocks)

    if lhs_save_dir:
        
        os.makedirs(lhs_save_dir, exist_ok=True)
        np.save(os.path.join(lhs_save_dir, 'lhs_points.npy'), np.asarray(external_lhs_points, dtype=float))
        np.save(os.path.join(lhs_save_dir, 'lhs_log_densities.npy'), np.asarray(external_lhs_log_densities, dtype=float))
        print(f"[INFO] Saved LHS points shape: {external_lhs_points.shape}, log-densities shape: {external_lhs_log_densities.shape}")

    max_idx = int(np.argmax(external_lhs_log_densities))
    max_point = external_lhs_points[max_idx]
    fallback_point = paris_prior_transform(max_point)
    fallback_score = float(external_lhs_log_densities[max_idx])
    sampler._fallback_best_point = np.asarray(fallback_point, dtype=float)
    sampler._fallback_best_score = fallback_score

    try:
        sampler.run_sampling(
            num_iterations=cfg.paris_niterations,
            savepath=savepath,
            print_iter=100,
            external_lhs_points=external_lhs_points,
            external_lhs_log_densities=external_lhs_log_densities,
        )
    except Exception as exc:
        print(f"[WARN] PARIS sampling failed: {exc}")
    return sampler, paris_prior_transform, external_lhs_points


def _ts():
    """Return a compact wall-clock timestamp string for log lines."""
    return time.strftime('[%H:%M:%S]')


def main(signal_param_array,
        dt,T,chi2,
        run_type,
        parameter_selected,base_dir,param_names_to_infer,
        target_func, optimizer,
        n_channels, startingpoints_file,
        include_noise,
        prior_sigma_range,using_evec,
        paris_conf,
        seed,cfg, use_gpu=True,
        cache_dir=None, grid_index=None,
        refine_only=False):
    

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    match = re.search(r'(\d+)(?=\D*$)', startingpoints_file)
    if match:
        id = int(match.group(1))
    else:
        raise ValueError("Could not extract id from startingpoints")
    try:
        starting_point = load_startingpoint_param_array(startingpoints_file)
        print(f"Loaded starting point from {startingpoints_file}: {starting_point}")
    except:
        print("Starting Point is None")
        starting_point = None

    # Initialise result_array to the starting point so that if the optimizer
    # crashes before writing its own result, we return the initial guess rather
    # than raising a NameError.
    result_array = starting_point.copy() if starting_point is not None else {}

    emri_kwargs = {"T": T, "dt": dt,'chi2': chi2,'evolve_1PA': True,'evolve_primary': False,'evolve_2PA': True}
    add_kwargs = {'chi2': chi2,'evolve_1PA': True,'evolve_primary': False,'evolve_2PA': True}
    # Fisher is always computed at 2PA (the true signal), independent of the
    # template PA order used in the optimisation objective.
    fisher_add_kwargs = {'chi2': chi2,'evolve_1PA': True,'evolve_primary': False,'evolve_2PA': True}
        
    print(f"\n{_ts()} {'='*55}")
    print(f"{_ts()} RUN CONFIGURATION")
    print(f"{_ts()} {'='*55}")
    print(f"{_ts()} TYPE={run_type}  template={parameter_selected}  optimizer={optimizer}")
    print(f"{_ts()} target={target_func}  nchannels={n_channels}  noise={include_noise}")
    print(f"{_ts()} params_to_infer={param_names_to_infer}")
    print(f"{_ts()} T={T:.3f}yr  dt={dt}s  chi2={chi2}")
    print(f"{_ts()} prior_sigma_range={prior_sigma_range}  using_evec={using_evec}")
    print(f"{_ts()} {'='*55}\n")

    print(f"{_ts()} Generating true 2PA waveform...")
    ctx = prepare_true_waveform(signal_param_array, emri_kwargs, add_kwargs,add_noise=include_noise, use_gpu=use_gpu,seed=seed,nchannels=n_channels)

    snr_2 = inner_prod(ctx['waveform_true_fft_without_noise'], ctx['waveform_true_fft_without_noise'], ctx['PSD_funcs'], ctx['delta_f'], xp=xp)
    snr = np.sqrt(snr_2.get()) if hasattr(snr_2, "get") else np.sqrt(snr_2)
    _TARGET_SNR = snr
    print(f"{_ts()} True signal SNR: {snr:.4f}")
        
    temp_dict = {'waveform_true_fft': ctx['waveform_true_fft'], 'waveform_true_fft_without_noise': ctx['waveform_true_fft_without_noise'],
                  'PSD': ctx['PSD_funcs'], 'dt': ctx['dt'], 'T': ctx['T'],
                     'N_fiducial': ctx['N_fiducial'], 'delta_f': ctx['delta_f'], 'use_gpu': use_gpu,
                      'waveform_response': ctx['waveform_response'],'xp': cp if use_gpu else np}
    
    for k in ['m1', 'm2', 'a', 'p0', 'e0', 'Y0', 'dist', 'qS', 'phiS', 'qK', 'phiK', 'Phi_phi0', 'Phi_theta0', 'Phi_r0', 'chi2','dt', 'T']:
        assert k in ctx, f"Missing {k} in 1PA context"

    # Initial theta from startingpoint array if available, else from signal row

    if run_type == '0pa_vs_2pa' and parameter_selected == "intrinsic":
        
        if starting_point is not None:
            theta0 = np.array([starting_point['m1'], starting_point['m2'], starting_point['a'], starting_point['p0'], starting_point['e0']], dtype=float)
        else:
            theta0 = np.array([ctx['m1'], ctx['m2'], ctx['a'], ctx['p0'], ctx['e0']], dtype=float)
        add_kwargs['evolve_1PA']=False
        add_kwargs['evolve_2PA']=False

        initial_overlap =calculate_detection_overlap(
                    m1=starting_point['m1'], m2=starting_point['m2'], a=starting_point['a'], p0=starting_point['p0'], e0=starting_point['e0'], Y0=ctx['Y0'], dist=ctx['dist'], qS=ctx['qS'], phiS=ctx['phiS'], qK=ctx['qK'], phiK=ctx['phiK'],
                    Phi_phi0=ctx['Phi_phi0'], Phi_theta0=ctx['Phi_theta0'], Phi_r0=ctx['Phi_r0'], add_kwargs=add_kwargs,
                    maximize_phase=False,
                    **temp_dict)
        print("Current  Overlap:", initial_overlap )
        ndim = 5

    elif run_type == '0pa_vs_2pa' and parameter_selected == "intrinsic_phase":
        add_kwargs['evolve_1PA'] = False
        add_kwargs['evolve_2PA'] = False
        if starting_point is not None:
            theta0 = np.array([starting_point['m1'], starting_point['m2'], starting_point['a'], starting_point['p0'],
                                starting_point['e0'], starting_point['Phi_phi0'], starting_point['Phi_r0']], dtype=float)
        else:
            theta0 = np.array([ctx['m1'], ctx['m2'], ctx['a'], ctx['p0'], ctx['e0'], ctx['Phi_phi0'], ctx['Phi_r0']], dtype=float)


        initial_overlap= calculate_detection_overlap(
                    m1=starting_point['m1'], m2=starting_point['m2'], a=starting_point['a'], p0=starting_point['p0'], e0=starting_point['e0'], Y0=ctx['Y0'], dist=ctx['dist'], qS=ctx['qS'], phiS=ctx['phiS'], qK=ctx['qK'], phiK=ctx['phiK'],
                    Phi_phi0=starting_point['Phi_phi0'], Phi_theta0=ctx['Phi_theta0'], Phi_r0=starting_point['Phi_r0'], add_kwargs=add_kwargs,
                    maximize_phase=False,
                    **temp_dict)
        print("Current  Overlap:",initial_overlap)
        ndim = 7

    elif run_type == '1pa_vs_2pa' and parameter_selected == "intrinsic":
        add_kwargs['evolve_1PA'] = True
        add_kwargs['evolve_2PA'] = False
        if starting_point is not None:
            theta0 = np.array([starting_point['m1'], starting_point['m2'], starting_point['a'], starting_point['p0'],
                                starting_point['e0'],starting_point['chi2']], dtype=float)
        else:
            theta0 = np.array([ctx['m1'], ctx['m2'], ctx['a'], ctx['p0'], ctx['e0'],ctx['chi2']], dtype=float)

        add_kwargs['chi2'] = theta0[-1]

        initial_overlap= calculate_detection_overlap(
                    m1=starting_point['m1'], m2=starting_point['m2'], a=starting_point['a'], p0=starting_point['p0'], e0=starting_point['e0'], Y0=ctx['Y0'], dist=ctx['dist'], qS=ctx['qS'], phiS=ctx['phiS'], qK=ctx['qK'], phiK=ctx['phiK'],
                    Phi_phi0=ctx['Phi_phi0'], Phi_theta0=ctx['Phi_theta0'], Phi_r0=ctx['Phi_r0'], add_kwargs=add_kwargs,
                    maximize_phase=False,
                    **temp_dict)
        print("Current  Overlap:",initial_overlap)
        ndim = 6

    
    elif run_type == '1pa_vs_2pa' and parameter_selected == "intrinsic_phase":
        add_kwargs['evolve_1PA'] = True
        add_kwargs['evolve_2PA'] = False
        if starting_point is not None:
            theta0 = np.array([starting_point['m1'], starting_point['m2'], starting_point['a'], starting_point['p0'],
                                starting_point['e0'], starting_point['Phi_phi0'], starting_point['Phi_r0'],starting_point['chi2']], dtype=float)
        else:
            theta0 = np.array([ctx['m1'], ctx['m2'], ctx['a'], ctx['p0'], ctx['e0'], ctx['Phi_phi0'], ctx['Phi_r0'],ctx['chi2']], dtype=float)

        add_kwargs['chi2'] = theta0[-1]

        initial_overlap= calculate_detection_overlap(
                    m1=starting_point['m1'], m2=starting_point['m2'], a=starting_point['a'], p0=starting_point['p0'], e0=starting_point['e0'], Y0=ctx['Y0'], dist=ctx['dist'], qS=ctx['qS'], phiS=ctx['phiS'], qK=ctx['qK'], phiK=ctx['phiK'],
                    Phi_phi0=starting_point['Phi_phi0'], Phi_theta0=ctx['Phi_theta0'], Phi_r0=starting_point['Phi_r0'], add_kwargs=add_kwargs,
                    maximize_phase=False,
                    **temp_dict)
        print("Current  Overlap:",initial_overlap)
        ndim = 8

    else:  
        raise ValueError(f"Unsupported run_type {run_type} with parameter_selected {parameter_selected}")
    
    # Objective setup with tracker for fallback support
    #no 1PA for analysis manifold
    #del temp_dict
    
    phase_max_flag = (target_func == 'optimal_snr_phase_max')
    
    raw_objective = objective_factory(
        target_func=target_func,
        ctx=ctx,
        phase_max=phase_max_flag,
        use_1PA= run_type == '1pa_vs_2pa',
        with_phase = parameter_selected == "intrinsic_phase",
        add_kwargs=add_kwargs,
    )

    tracker = ObjectiveTracker(theta0)
    def tracked_objective(theta: np.ndarray) -> float:
        val = float(raw_objective(np.asarray(theta, dtype=float)))
        tracker.update(theta, val)
        return val
    try:
        tracked_objective(theta0)
    except Exception as exc:
        print(f"[WARN] Initial objective evaluation failed at theta0: {exc}")
        tracker._best_score = float('-inf')
        
    objective = tracked_objective
    result = None
    if optimizer == 'nelder-mead':
            try:
                # Constrain search to remain within relative deviation of original (ctx-based) parameters
                tol = 1e-6 #1e-8 1pa emri #1e-6
                theta_ref = theta0.copy()
                def bounded_objective(theta: np.ndarray) -> float:
                    score_val = objective(theta)
                    return -float(score_val)
                print(f"Starting Nelder-Mead optimization with initial theta: {theta0}")
                result = nelder_mead_optimize(
                    theta0,
                    bounded_objective,
                    maxiter=cfg.nm_maxiter,
                    xatol=cfg.nm_xatol,
                    fatol=cfg.nm_fatol,
                )
                best_score = -float(result.fun)
                tracker.update(result.x, best_score)
    
                # Per-index output directory named with best score and optimized point
                _opt_vals = result.x
                
                # _vals_str = '_'.join(f"{v:.6e}" for v in _opt_vals)
                idx_dir = os.path.join(base_dir, f"nelder_mead_{target_func}_run_id_{id}")
               # idx_dir = os.path.join(nealder_mead_dir, f"{best_score:.12g}_{_vals_str}")
                os.makedirs(idx_dir, exist_ok=True)
                
                out_name = os.path.join(idx_dir, f"opt_nelder-mead_{target_func}_{timestamp}_id_{id}.json")
                out = {
                    'optimizer': 'nelder-mead',
                    'target_func': target_func,
                    'theta0': theta0.tolist(),
                    'x': result.x.tolist(),
                    'fun': float(result.fun),
                    'best_score': best_score,
                    'success': bool(result.success),
                    'snr_ref_1pa': float(ctx.get('snr', np.nan)),
                    # 'initial_overlap': float(initial_overlap),
                    # 'final_overlap': float(final_overlap),
                }
                with open(out_name, 'w') as f:
                    json.dump(out, f, indent=2)
                
                print(f"Saved result: {out_name}")
                print(f"[RESULT] Best loss (=-score): {out['fun']:.6e}")
                print(f"[RESULT] Best score: {out['best_score']:.6e}")
                print(f"[RESULT] Best point: {out['x']}")
                if not result.success:
                    print(f"[WARN] Nelder-Mead optimization did not converge: {result.message}")
    
                match ndim:
                    case 5:
                        print(f"Optimized (m1, m2, a, p0, e0): {result.x}")
                        result_array = starting_point.copy()
                        starting_point_keys = ['m1', 'm2', 'a', 'p0', 'e0']
                        for i, key in enumerate(starting_point_keys):
                            result_array[key] = result.x[i]
                        print(f"Optimized parameters as array: {result_array}")
                        np.save(os.path.join(idx_dir, f"results_nelder_mead_{id+1}_time_{timestamp}.npy"), result_array)
                        np.save(os.path.join(idx_dir, f"starting_point_{id+1}.npy"), result_array)
                    case 6:
                        print(f"Optimized (m1, m2, a, p0, e0, chi2): {result.x}")
                        result_array = starting_point.copy()
                        starting_point_keys = ['m1', 'm2', 'a', 'p0', 'e0', 'chi2']
                        for i, key in enumerate(starting_point_keys):
                            result_array[key] = result.x[i]
                        print(f"Optimized parameters as array: {result_array}")
                        np.save(os.path.join(idx_dir, f"results_nelder_mead_{id+1}_time_{timestamp}.npy"), result_array)
                        np.save(os.path.join(idx_dir, f"starting_point_{id+1}.npy"), result_array)

                    case 7:
                        print(f"Optimized (m1, m2, a, p0, e0, Phi_phi0, Phi_r0): {result.x}")
                        result_array = starting_point.copy()
                        starting_point_keys = ['m1', 'm2', 'a', 'p0', 'e0', 'Phi_phi0', 'Phi_r0']
                        for i, key in enumerate(starting_point_keys):
                            result_array[key] = result.x[i]
                        print(f"Optimized parameters as array: {result_array}")
                        np.save(os.path.join(idx_dir, f"results_nelder_mead_{id+1}_time_{timestamp}.npy"), result_array)
                        np.save(os.path.join(idx_dir, f"starting_point_{id+1}.npy"), result_array)

                    case 8:
                        print(f"Optimized (m1, m2, a, p0, e0, Phi_phi0, Phi_r0, chi2): {result.x}")
                        result_array = starting_point.copy()
                        starting_point_keys = ['m1', 'm2', 'a', 'p0','e0', 'Phi_phi0','Phi_r0','chi2']
                        for i, key in enumerate(starting_point_keys):
                            result_array[key] = result.x[i]
                        print(f"Optimized parameters as array: {result_array}")
                        np.save(os.path.join(idx_dir, f"results_nelder_mead_{id+1}_time_{timestamp}.npy"), result_array)
                        np.save(os.path.join(idx_dir, f"starting_point_{id+1}.npy"), result_array)

                    case _:
                        print(f"Optimized parameters: {result.x}")

                add_kwargs['chi2']=result_array['chi2']

                final_overlap = calculate_detection_overlap(
                    result_array['m1'], result_array['m2'], result_array['a'], result_array['p0'], result_array['e0'], ctx['Y0'],ctx['dist'],ctx['qS'],ctx['phiS'], ctx['qK'], ctx['phiK'], 
                    result_array['Phi_phi0'], ctx['Phi_theta0'], result_array['Phi_r0'],add_kwargs,
                    maximize_phase=False,
                    **temp_dict)
                
                print("Overlap of the best point:", final_overlap)
                out = {
                    'optimizer': 'nelder-mead',
                    'target_func': target_func,
                    'theta0': theta0.tolist(),
                    'x': result.x.tolist(),
                    'fun': float(result.fun),
                    'best_score': best_score,
                    'success': bool(result.success),
                    'snr_ref_1pa': float(ctx.get('snr', np.nan)),
                    'initial_overlap': float(initial_overlap),
                    'final_overlap': float(final_overlap),
                }
                
                Config.save_results_with_config(
                                        cfg=cfg,
                                        results=out,
                                        save_dir=idx_dir,
                                        filename_prefix=f"opt_nelder_mead_{target_func}_id_{id}"
                                    )
    
            except Exception as exc:
                print(f"[ERROR] Nelder-Mead optimization failed: {exc}")

            return result_array

    
    elif optimizer == 'differential_evolution':
                # Constrain search to remain within relative deviation of original (ctx-based) parameters
            tol = 1e-8 #1e-8 1pa emri #1e-6
            # Centre DE bounds on the TRUE SIGNAL (ctx) so DE can escape stuck basins
            # regardless of where the warm-start ended up.  theta0 (warm-start) may be
            # outside these bounds when it is a stuck PARIS result.
            _signal_keys = ['m1', 'm2', 'a', 'p0', 'e0']
            if parameter_selected == 'intrinsic_phase':
                _signal_keys = ['m1', 'm2', 'a', 'p0', 'e0', 'Phi_phi0', 'Phi_r0']
            if run_type == '1pa_vs_2pa':
                _signal_keys = [k for k in _signal_keys] + ['chi2']
                if parameter_selected != 'intrinsic_phase':
                    _signal_keys = ['m1', 'm2', 'a', 'p0', 'e0', 'chi2']
            theta_signal = np.array([ctx[k] for k in _signal_keys[:ndim]], dtype=float)
            theta_ref = theta_signal
            print(f"DE bounds centred on signal: {dict(zip(_signal_keys[:ndim], theta_signal))}")
            print(f"  (warm-start theta0={theta0})")
            try:
                import cupy as cP
                USE_GPU = True
            except ImportError:
                USE_GPU = False

            try:

                Q, b, fisher_meta = compute_fisher_parallelotope(
                    ctx=ctx,
                    params_to_infer= param_names_to_infer,
                    fisher_params=signal_param_array,
                    use_gpu=USE_GPU,
                    prior_sigma_range=float(prior_sigma_range),
                    using_evec=using_evec,
                    additional_kwargs=fisher_add_kwargs,
                    _TARGET_SNR= _TARGET_SNR,
                    build_waveform_response= build_waveform_response,
                    cache_dir=cache_dir,
                    cache_index=grid_index,
                )
                diag_sigma_fisher = np.asarray(fisher_meta['diag_sigma'])
                # Pad phase dimensions with uniform ±pi if intrinsic_phase
                if len(diag_sigma_fisher) < ndim:
                    diag_sigma_full = np.full(ndim, np.pi / float(prior_sigma_range))
                    diag_sigma_full[:len(diag_sigma_fisher)] = diag_sigma_fisher
                else:
                    diag_sigma_full = diag_sigma_fisher[:ndim]
                bounds = []
                for i in range(ndim):
                    bounds.append((theta_ref[i] - diag_sigma_full[i]*prior_sigma_range, theta_ref[i] + diag_sigma_full[i]*prior_sigma_range))

                print("Fisher-based bounds for optimization:")
                for i, (lower, upper) in enumerate(bounds):
                    print(f"  {param_names_to_infer[i]}: [{lower:.6e}, {upper:.6e}]")

                def bounded_objective(theta: np.ndarray) -> float:
                    try:
                        score_val = objective(theta)
                        return -float(score_val)
                    except Exception:
                        return np.inf  # invalid waveform — treat as worst score
                
                _de_gen1 = [0]
                _de_t1_start = time.time()
                def _de_stage1_callback(xk, convergence):
                    _de_gen1[0] += 1
                    g = _de_gen1[0]
                    if g % 50 == 0:
                        elapsed = (time.time() - _de_t1_start) / 60
                        score = -float(bounded_objective(xk))
                        print(f"{_ts()} [DE stage-1] gen={g:4d}/{cfg.de_maxiter}  "
                              f"best_overlap={score:.6e}  convergence={convergence:.4f}  "
                              f"elapsed={elapsed:.1f}min")
                    return False

                result = differential_evolution_optimize(
                    theta0=theta_signal,
                    objective=bounded_objective,
                    fisher_bounds=bounds,
                    maxiter=cfg.de_maxiter,
                    tol=tol,
                    seed=seed,
                    popsize=cfg.de_popsize,
                    callback=_de_stage1_callback,
                    workers=cfg.de_workers,
                )
                best_score = -float(result.fun)
                tracker.update(result.x, best_score)
    
                # Per-index output directory named with best score and optimized point
                
                _opt_vals = result.x
                
                # _vals_str = '_'.join(f"{v:.6e}" for v in _opt_vals)
                idx_dir = os.path.join(base_dir, f"differential_evolution_{target_func}_run_id_{id}")
               # idx_dir = os.path.join(nealder_mead_dir, f"{best_score:.12g}_{_vals_str}")
                os.makedirs(idx_dir, exist_ok=True)
                
                out_name = os.path.join(idx_dir, f"opt_differential_evolution_{target_func}_{timestamp}_id_{id}.json")
                out = {
                    'optimizer': 'differential_evolution',
                    'target_func': target_func,
                    'theta0': theta0.tolist(),
                    'x': result.x.tolist(),
                    'fun': float(result.fun),
                    'best_score': best_score,
                    'success': bool(result.success),
                    'snr_ref_1pa': float(ctx.get('snr', np.nan)),
                     'fisher_bounds': bounds,
     
                }
                with open(out_name, 'w') as f:
                    json.dump(out, f, indent=2)
                
                print(f"Saved result: {out_name}")
                print(f"[RESULT] Best loss (=-score): {out['fun']:.6e}")
                print(f"[RESULT] Best score: {out['best_score']:.6e}")
                print(f"[RESULT] Best point: {out['x']}")
                if not result.success:
                    print(f"[WARN] Differential Evolution optimization did not converge: {result.message}")

                match ndim:
                    case 5:
                        print(f"Optimized (m1, m2, a, p0, e0): {result.x}")
                        result_array = starting_point.copy()
                        starting_point_keys = ['m1', 'm2', 'a', 'p0', 'e0']
                        for i, key in enumerate(starting_point_keys):
                            result_array[key] = result.x[i]
                        print(f"Optimized parameters as array: {result_array}")
                        np.save(os.path.join(idx_dir, f"results_differential_evolution_{id+1}_time_{timestamp}.npy"), result_array)
                        np.save(os.path.join(idx_dir, f"starting_point_{id+1}.npy"), result_array)
                    case 6:
                        print(f"Optimized (m1, m2, a, p0, e0, chi2): {result.x}")
                        result_array = starting_point.copy()
                        starting_point_keys = ['m1', 'm2', 'a', 'p0', 'e0', 'chi2']
                        for i, key in enumerate(starting_point_keys):
                            result_array[key] = result.x[i]
                        print(f"Optimized parameters as array: {result_array}")
                        np.save(os.path.join(idx_dir, f"results_differential_evolution_time_{timestamp}.npy"), result_array)
                        np.save(os.path.join(idx_dir, f"starting_point_{id+1}.npy"), result_array)

                    case 7:
                        print(f"Optimized (m1, m2, a, p0, e0, Phi_phi0, Phi_r0): {result.x}")
                        result_array = starting_point.copy()
                        starting_point_keys = ['m1', 'm2', 'a', 'p0', 'e0', 'Phi_phi0', 'Phi_r0']
                        for i, key in enumerate(starting_point_keys):
                            result_array[key] = result.x[i]
                        print(f"Optimized parameters as array: {result_array}")
                        np.save(os.path.join(idx_dir, f"results_differential_evolution_{id+1}_time_{timestamp}.npy"), result_array)
                        np.save(os.path.join(idx_dir, f"starting_point_{id+1}.npy"), result_array)

                    case 8:
                        print(f"Optimized (m1, m2, a, p0, e0, Phi_phi0, Phi_r0, chi2): {result.x}")
                        result_array = starting_point.copy()
                        starting_point_keys = ['m1', 'm2', 'a', 'p0', 'e0', 'Phi_phi0', 'Phi_r0','chi2']
                        for i, key in enumerate(starting_point_keys):
                            result_array[key] = result.x[i]
                        print(f"Optimized parameters as array: {result_array}")
                        np.save(os.path.join(idx_dir, f"results_differential_evolution_{id+1}_time_{timestamp}.npy"), result_array)
                        np.save(os.path.join(idx_dir, f"starting_point_{id+1}.npy"), result_array)

                    case _:
                        print(f"Optimized parameters: {result.x}")


                add_kwargs['chi2']=result_array['chi2']

                final_overlap = calculate_detection_overlap(
                    result_array['m1'], result_array['m2'], result_array['a'], result_array['p0'], result_array['e0'], ctx['Y0'],ctx['dist'],ctx['qS'],ctx['phiS'], ctx['qK'], ctx['phiK'], 
                    result_array['Phi_phi0'], ctx['Phi_theta0'], result_array['Phi_r0'],add_kwargs,
                    maximize_phase=False,
                    **temp_dict)
                print("Overlap of the best point:", final_overlap)
                out = {
                    'optimizer': 'differential_evolution',
                    'target_func': target_func,
                    'theta0': theta0.tolist(),
                    'x': result.x.tolist(),
                    'fun': float(result.fun),
                    'best_score': best_score,
                    'success': bool(result.success),
                    'snr_ref_1pa': float(ctx.get('snr', np.nan)),
                    'fisher_meta': bool(fisher_meta),
                    'fisher_bounds': bounds,
                    'initial_overlap': float(initial_overlap),
                    'final_overlap': float(final_overlap),
                }
                
                Config.save_results_with_config(
                                        cfg=cfg,
                                        results=out,
                                        save_dir=idx_dir,
                                        filename_prefix=f"opt_differential_evolution_{target_func}_id_{id}"
                                    )

                # ---------------------------
                # Post-DE refinement: narrow DE then Nelder-Mead (with phases)
                # Mirrors the refine_after_paris block in the PARIS path.
                # ---------------------------
                print(f"\n{_ts()} {'='*55}")
                print(f"{_ts()} STAGE 2 (REFINE): Differential Evolution "
                      f"({cfg.de_refine_maxiter} gen, popsize={cfg.de_refine_popsize})")
                print(f"{_ts()} {'='*55}")
                _t_de_refine_start = time.time()
                try:
                    _de_best = np.asarray(result.x, dtype=float)
                    _rpr = cfg.refine_prior_sigma_range
                    _refine_bounds = [(
                        _de_best[i] - diag_sigma_fisher[i] * _rpr,
                        _de_best[i] + diag_sigma_fisher[i] * _rpr
                    ) for i in range(ndim)]
                    _refine_with_phase = (parameter_selected == 'intrinsic')
                    if _refine_with_phase:
                        _phi0_init = float(result_array['Phi_phi0'])
                        _phir0_init = float(result_array['Phi_r0'])
                        _de_best_r = np.append(_de_best, [_phi0_init, _phir0_init])
                        _refine_bounds_r = _refine_bounds + [(0.0, 2*np.pi), (0.0, 2*np.pi)]
                        def _neg_obj_r(theta):
                            try:
                                return -float(calculate_detection_overlap(
                                    theta[0], theta[1], theta[2], theta[3], theta[4],
                                    ctx['Y0'], ctx['dist'], ctx['qS'], ctx['phiS'],
                                    ctx['qK'], ctx['phiK'],
                                    theta[5], ctx['Phi_theta0'], theta[6],
                                    add_kwargs, maximize_phase=False, **temp_dict))
                            except Exception:
                                return np.inf
                    else:
                        _de_best_r = _de_best
                        _refine_bounds_r = _refine_bounds
                        def _neg_obj_r(theta):
                            try:
                                return -float(objective(theta))
                            except Exception:
                                return np.inf
                    _refine_val = -_neg_obj_r(_de_best_r)
                    _de_refine_result = differential_evolution_optimize(
                        theta0=_de_best_r,
                        objective=_neg_obj_r,
                        fisher_bounds=_refine_bounds_r,
                        maxiter=cfg.de_refine_maxiter,
                        seed=seed,
                        init='latinhypercube',
                        popsize=cfg.de_refine_popsize,
                        workers=cfg.de_workers,
                    )
                    _de_refine_elapsed = (time.time() - _t_de_refine_start) / 60
                    if -_de_refine_result.fun > _refine_val:
                        _de_best_r = np.asarray(_de_refine_result.x, dtype=float)
                        _refine_val = -_de_refine_result.fun
                        print(f"{_ts()} [REFINE] DE improved overlap to {_refine_val:.6e}  "
                              f"({_de_refine_elapsed:.1f} min, {_de_refine_result.nfev} evals)")
                    else:
                        print(f"{_ts()} [REFINE] DE did not improve "
                              f"({-_de_refine_result.fun:.6e} vs {_refine_val:.6e}, "
                              f"{_de_refine_elapsed:.1f} min, {_de_refine_result.nfev} evals)")
                except Exception as exc_der:
                    import traceback
                    print(f"{_ts()} [WARN] DE refinement failed: {exc_der}\n{traceback.format_exc()}")

                print(f"\n{_ts()} {'='*55}")
                print(f"{_ts()} STAGE 3 (REFINE): Nelder-Mead phase polish "
                      f"(maxiter={cfg.nm_refine_maxiter})")
                print(f"{_ts()} {'='*55}")
                _t_nm_refine_start = time.time()
                try:
                    nm_refine_result = nelder_mead_optimize(
                        _de_best_r,
                        _neg_obj_r,
                        maxiter=cfg.nm_refine_maxiter,
                        xatol=cfg.nm_xatol,
                        fatol=cfg.nm_fatol,
                    )
                    _nm_refine_elapsed = (time.time() - _t_nm_refine_start) / 60
                    if -nm_refine_result.fun > _refine_val:
                        _de_best_r = np.asarray(nm_refine_result.x, dtype=float)
                        _refine_val = -nm_refine_result.fun
                        print(f"{_ts()} [REFINE] NM improved overlap to {_refine_val:.6e}  "
                              f"({_nm_refine_elapsed:.1f} min, {nm_refine_result.nfev} evals, "
                              f"converged={nm_refine_result.success})")
                    else:
                        print(f"{_ts()} [REFINE] NM did not improve "
                              f"({-nm_refine_result.fun:.6e} vs {_refine_val:.6e}, "
                              f"{_nm_refine_elapsed:.1f} min, converged={nm_refine_result.success})")
                    for i, key in enumerate(param_names_to_infer):
                        result_array[key] = _de_best_r[i]
                    if _refine_with_phase:
                        result_array['Phi_phi0'] = float(_de_best_r[5])
                        result_array['Phi_r0']   = float(_de_best_r[6])
                        print(f"[REFINE] Best phases: Phi_phi0={_de_best_r[5]:.6f}  "
                              f"Phi_r0={_de_best_r[6]:.6f}  "
                              f"(signal: {ctx['Phi_phi0']:.6f}, {ctx['Phi_r0']:.6f})")
                    add_kwargs['chi2'] = result_array['chi2']
                    final_overlap_refined = calculate_detection_overlap(
                        result_array['m1'], result_array['m2'], result_array['a'],
                        result_array['p0'], result_array['e0'], ctx['Y0'],
                        ctx['dist'], ctx['qS'], ctx['phiS'], ctx['qK'], ctx['phiK'],
                        result_array['Phi_phi0'], ctx['Phi_theta0'], result_array['Phi_r0'],
                        add_kwargs, maximize_phase=False, **temp_dict)
                    print(f"[REFINE] Final overlap after NM: {final_overlap_refined:.6f}")
                    np.save(os.path.join(idx_dir, f"results_refined_{id+1}_time_{timestamp}.npy"), result_array)
                    np.save(os.path.join(idx_dir, f"starting_point_{id+1}.npy"), result_array)
                except Exception as exc_nm:
                    import traceback
                    print(f"[WARN] NM phase polish failed: {exc_nm}\n{traceback.format_exc()}")

            except Exception as exc:
                import traceback
                print(f"[ERROR] Differential Evolution optimization failed: "
                      f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")

            return result_array
                
                
    elif optimizer == 'paris':
        # --- PARIS Optimization Block ---

        try:
        # if True:
            # ---------------------------
            # Fisher prior computation
            # ---------------------------
            try:
                import cupy as cP
                USE_GPU = True
            except ImportError:
                USE_GPU = False

            try:

                Q, b, fisher_meta = compute_fisher_parallelotope(
                    ctx=ctx,
                    params_to_infer=param_names_to_infer,
                    fisher_params=signal_param_array,
                    use_gpu=USE_GPU,
                    prior_sigma_range=float(prior_sigma_range),
                    using_evec=using_evec,
                    additional_kwargs=fisher_add_kwargs,
                    _TARGET_SNR= _TARGET_SNR,
                    build_waveform_response= build_waveform_response,
                    cache_dir=cache_dir,
                    cache_index=grid_index,
                )
                print(f"{_ts()} Fisher parallelotope computed successfully.")
                fisher_ok = True

                # When parameter_selected == "intrinsic_phase", ndim is 7 (0PA) or
                # 8 (1PA) but the Fisher only covers the intrinsic params (5 or 6D).
                # Phase params (Phi_phi0, Phi_r0 at theta indices 5–6) have no
                # Fisher-informed prior — their correlations with intrinsic params are
                # negligible for EMRI signals.  Pad Q and b with a uniform ±pi prior
                # for the phase dimensions so PARIS operates in the correct space.
                if len(b) < ndim:
                    n_fisher = len(b)
                    Q_padded = np.eye(ndim)
                    b_padded = np.full(ndim, np.pi)      # default: uniform [0,2pi]
                    Q_padded[:n_fisher, :n_fisher] = Q   # intrinsic Fisher block
                    b_padded[:n_fisher] = b
                    # For 1PA intrinsic_phase (ndim=8): chi2 is last in both Fisher
                    # and theta, so the block copy above already handles it correctly
                    # as long as param_names_to_infer ends with chi2.
                    Q, b = Q_padded, b_padded
                    print(f"[FISHER] Padded to ndim={ndim}: Fisher dims {n_fisher}, "
                          f"phase dims {ndim - n_fisher} (uniform ±pi prior)")

            except Exception as e:
                raise RuntimeError(f"[FATAL] Fisher prior failed: {e}") from e

            # ---------------------------
            # Directory setup
            # ---------------------------
            idx_dir = os.path.join(base_dir, f"paris_{target_func}_id_{id}")
            os.makedirs(idx_dir, exist_ok=True)

            savepath = os.path.join(idx_dir, f"paris_results_{target_func}_{timestamp}_id_{id}")
            lhs_seed_rel = "lhs_seed"
            lhs_seed_dir = os.path.join(idx_dir, lhs_seed_rel)

            # ---------------------------
            # Run PARIS optimizer (or skip if --refine-only)
            # ---------------------------
            if refine_only:
                # Load best_theta from the warm-start starting point (previous
                # PARIS result) and bypass sampling entirely.
                if starting_point is not None:
                    keys = param_names_to_infer
                    best_theta = np.array([starting_point[k] for k in keys], dtype=float)
                    best_val   = float(objective(best_theta))
                    print(f"{_ts()} [REFINE-ONLY] Loaded from checkpoint: "
                          f"score={best_val:.6e}  theta={best_theta.tolist()}")
                    # Save a raw checkpoint so the rest of the code can proceed
                    _ckpt_path = os.path.join(idx_dir, f"checkpoint_paris_raw_{timestamp}.npy")
                    np.save(_ckpt_path, best_theta)
                    # Skip to polish — jump over the run_paris / extract block
                    fisher_meta = fisher_meta  # already computed above
                else:
                    print(f"{_ts()} [WARN] --refine-only but no starting point found; running PARIS")
                    refine_only = False

            if not refine_only:
                print(f"\n{_ts()} {'='*55}")
                print(f"{_ts()} STAGE 1: PARIS ({paris_conf['paris_seed_n']} seeds, "
                      f"{cfg.paris_niterations} iterations, ndim={ndim})")
                print(f"{_ts()} {'='*55}")
                _t_paris_start = time.time()
                sampler, prior_transform, ext_points = run_paris(
                ndim=ndim,
                prior_center=theta0,
                score_func=objective,
                spread_scale=float(paris_conf['spread_scale']),
                savepath=savepath,
                seed_cloud=int(paris_conf['seed_cloud']),
                seed_jitter=1e-10,
                paris_seed = int(paris_conf['paris_seed_n']),
                target_kind=target_func,
                lhs_save_dir=lhs_seed_dir,
                affine_Q=Q if fisher_ok else None,
                affine_b=b if fisher_ok else None,
                use_ellipse=bool(fisher_meta.get("using_evec", False)),
            )

            # ---------------------------
            # Extract best point (PARIS path only)
            # ---------------------------
            if not refine_only:
                def extract_best_point():
                    try:
                        pts = sampler.searched_points_list
                        logs = sampler.searched_log_densities_list

                        if not pts or not logs:
                            raise ValueError("Empty PARIS search results")

                        # best_unit = pts[0][int(np.argmax(logs[0]))]
                        best_unit = max((pts[i][np.argmax(logs[i])] for i in range(len(pts))),key=lambda u: paris_log_density(paris_prior_transform(u.reshape(1,-1)))[0])
                        return prior_transform(best_unit)

                    except Exception as e:
                        print(f"[WARN] Best extraction failed: {e}, using fallback")

                        fallback = getattr(sampler, "_fallback_best_point", None)
                        if fallback is None:
                            raise RuntimeError("No fallback best point available") from e

                        return np.asarray(fallback, dtype=float)

                best_theta = extract_best_point()

                if best_theta is None:
                    print("[WARN] Using starting point as fallback")
                    best_theta = theta0

                best_theta = np.asarray(best_theta, dtype=float)

                # Transform if still in unit cube
                if np.all((best_theta >= 0.0) & (best_theta <= 1.0)):
                    best_theta = prior_transform(best_theta)

                best_val = float(objective(best_theta))
                print(f"{_ts()} PARIS done in {(time.time()-_t_paris_start)/3600:.2f}h  "
                      f"best_score={best_val:.6e}  best_theta={best_theta.tolist()}")

                # Checkpoint: save raw PARIS best BEFORE polish so a crash during
                # the ~40-min Gaussian polish does not lose the PARIS result.
                _ckpt_path = os.path.join(idx_dir, f"checkpoint_paris_raw_{timestamp}.npy")
                np.save(_ckpt_path, best_theta)
                print(f"{_ts()} [CHECKPOINT] Raw PARIS best saved → {_ckpt_path}")

            # ---------------------------
            # Local polishing (Gaussian steps)
            # ---------------------------
            best_fit_points = signal_param_array
            if parameter_selected == "intrinsic":
                if run_type == '0pa_vs_2pa':
                    best_fit_points[0:5] = best_theta
                elif run_type == '1pa_vs_2pa':
                    best_fit_points[0:5] = best_theta[0:5]
                    best_fit_points[-1] = best_theta[5]

            else:
                if run_type == '0pa_vs_2pa':
                    best_fit_points[0:5] = best_theta[0:5]
                    best_fit_points[11] = best_theta[5]
                    best_fit_points[13] = best_theta[6]
                elif run_type == '1pa_vs_2pa':
                    best_fit_points[0:5] = best_theta[0:5]
                    best_fit_points[11] = best_theta[5]
                    best_fit_points[13] = best_theta[6]
                    best_fit_points[-1] = best_theta[7]
                else:
                    print("Unsupported Type")


            ndim_local = len(best_theta)

            Qp, bp, _ = compute_fisher_parallelotope(
                       ctx=ctx,
                    params_to_infer= param_names_to_infer,
                    fisher_params=best_fit_points,
                    use_gpu=USE_GPU,
                    prior_sigma_range=float(prior_sigma_range),
                    using_evec=using_evec,
                    additional_kwargs=fisher_add_kwargs,
                        _TARGET_SNR= _TARGET_SNR,
                    build_waveform_response= build_waveform_response
                )

            cov = covariance_from_fisher_parallelotope(
                Qp, bp, prior_sigma_range=float(prior_sigma_range)
            )

            rng = np.random.default_rng()

            for _ in range(500):
                step = rng.multivariate_normal(
                    mean=np.zeros(ndim_local),
                    cov=1e-5 * cov,
                )

                cand = _clip_physical_params_intrinsic(best_theta + step)
                val = float(objective(cand))

                if val > best_val:
                    best_theta, best_val = cand, val

            tracker.update(best_theta, best_val)

            print(f"[POLISH] Final best score: {best_val:.6e}")
            print(f"[POLISH] Final best point: {best_theta.tolist()}")

            # ---------------------------
            # Rename directory based on result
            # ---------------------------
            # vals_str = ",".join(f"{float(v):.12g}" for v in best_theta)
            # new_idx_dir = os.path.join(base_dir, f"{best_val:.12g}_{vals_str}")

            # if os.path.abspath(new_idx_dir) != os.path.abspath(idx_dir):
            #     os.rename(idx_dir, new_idx_dir)
            #     idx_dir = new_idx_dir
            #     savepath = os.path.join(idx_dir, os.path.basename(savepath))

            lhs_seed_dir = os.path.join(idx_dir, lhs_seed_rel)

            # ---------------------------
            # Save outputs
            # ---------------------------
            out = {
                "optimizer": "PARIS",
                "target_func": target_func,
                "theta0": theta0.tolist(),
                "snr_ref_2pa": float(ctx.get("snr", np.nan)),
                "savepath": savepath,
                "fisher_prior": True,
                "fisher_meta": fisher_meta,
                "best_point": best_theta.tolist(),
                "best_score": best_val,
                "lhs_seed_dir": lhs_seed_dir,}

            json_path = os.path.join(idx_dir, f"opt_PARIS_{target_func}_{timestamp}.json")
            with open(json_path, "w") as f:
                json.dump(out, f, indent=2)

            np.save(
                os.path.join(idx_dir, f"score_PARIS_{target_func}_{timestamp}.npy"),
                np.array([best_val], dtype=float),
            )

            match ndim:
                        case 5:
                            print(f"Optimized (m1, m2, a, p0, e0): {best_theta}")
                            result_array = starting_point.copy()
                            starting_point_keys = ['m1', 'm2', 'a', 'p0', 'e0']
                            for i, key in enumerate(starting_point_keys):
                                result_array[key] = best_theta[i]
                            print(f"Optimized parameters as array: {result_array}")
                            np.save(os.path.join(idx_dir, f"results_paris_{id+1}_time_{timestamp}.npy"), result_array)
                            np.save(os.path.join(idx_dir, f"starting_point_{id+1}.npy"), result_array)
                        case 6:
                            print(f"Optimized (m1, m2, a, p0, e0, chi2): {best_theta}")
                            result_array = starting_point.copy()
                            starting_point_keys = ['m1', 'm2', 'a', 'p0', 'e0', 'chi2']
                            for i, key in enumerate(starting_point_keys):
                                result_array[key] = best_theta[i]
                            print(f"Optimized parameters as array: {result_array}")
                            np.save(os.path.join(idx_dir, f"results_paris_{timestamp}.npy"), result_array)
                            np.save(os.path.join(idx_dir, f"starting_point_{id+1}.npy"), result_array)

                        case 7:
                            print(f"Optimized (m1, m2, a, p0, e0, Phi_phi0, Phi_r0): {best_theta}")
                            result_array = starting_point.copy()
                            starting_point_keys = ['m1', 'm2', 'a', 'p0', 'e0', 'Phi_phi0', 'Phi_r0']
                            for i, key in enumerate(starting_point_keys):
                                result_array[key] = best_theta[i]
                            print(f"Optimized parameters as array: {result_array}")
                            np.save(os.path.join(idx_dir, f"results_paris_{id+1}_time_{timestamp}.npy"), result_array)
                            np.save(os.path.join(idx_dir, f"starting_point_{id+1}.npy"), result_array)

                        case 8:
                            print(f"Optimized (m1, m2, a, p0, e0, Phi_phi0, Phi_r0, chi2): {best_theta}")
                            result_array = starting_point.copy()
                            starting_point_keys = ['m1', 'm2', 'a', 'p0', 'e0', 'Phi_phi0', 'Phi_r0','chi2']
                            for i, key in enumerate(starting_point_keys):
                                result_array[key] = best_theta[i]
                            print(f"Optimized parameters as array: {result_array}")
                            np.save(os.path.join(idx_dir, f"results_paris_{id+1}_time_{timestamp}.npy"), result_array)
                            np.save(os.path.join(idx_dir, f"starting_point_{id+1}.npy"), result_array)

                        case _:
                            print(f"Optimized parameters: {best_theta}")


            add_kwargs['chi2']=result_array['chi2']


            final_overlap = calculate_detection_overlap(
                    result_array['m1'], result_array['m2'], result_array['a'], result_array['p0'], result_array['e0'], ctx['Y0'],ctx['dist'],ctx['qS'],ctx['phiS'], ctx['qK'], ctx['phiK'], 
                    result_array['Phi_phi0'], ctx['Phi_theta0'], result_array['Phi_r0'],add_kwargs,
                    maximize_phase=False,
                    **temp_dict)
            print("Overlap of the best point:", final_overlap)
            out = {
                "optimizer": "PARIS",
                "target_func": target_func,
                "theta0": theta0.tolist(),
                "snr_ref_2pa": float(ctx.get("snr", np.nan)),
                "savepath": savepath,
                "fisher_prior": True,
                "fisher_meta": fisher_meta,
                "best_point": best_theta.tolist(),
                "best_score": best_val,
                "lhs_seed_dir": lhs_seed_dir,
                "initial_overlap": float(initial_overlap),
                 "final_overlap": float(final_overlap),
            }
            Config.save_results_with_config(
            cfg=cfg,
            results=out,
            save_dir=idx_dir,
            filename_prefix=f"opt_PARIS_{target_func}_id_{id}")

            # ---------------------------
            # Overlap sanity check
            # ---------------------------
            if final_overlap < cfg.overlap_warn_threshold:
                print(f"[WARN] Overlap after PARIS ({final_overlap:.4f}) is below "
                      f"threshold ({cfg.overlap_warn_threshold}). "
                      f"Refinement may not converge well.")
            else:
                print(f"[INFO] Overlap after PARIS: {final_overlap:.4f} — proceeding to refinement.")

            # ---------------------------
            # Post-PARIS refinement: DE then Nelder-Mead
            # ---------------------------
            if cfg.refine_after_paris:
                print(f"\n{_ts()} {'='*55}")
                print(f"{_ts()} STAGE 2 (REFINE): Differential Evolution "
                      f"({cfg.de_refine_maxiter} gen, popsize={cfg.de_refine_popsize})")
                print(f"{_ts()} {'='*55}")
                _t_de_start = time.time()
                try:
                    # Build Fisher bounds around PARIS best point
                    diag_sigma_r = np.asarray(fisher_meta['diag_sigma'])
                    if len(diag_sigma_r) < ndim:
                        diag_sigma_full_r = np.full(ndim, np.pi / float(prior_sigma_range))
                        diag_sigma_full_r[:len(diag_sigma_r)] = diag_sigma_r
                    else:
                        diag_sigma_full_r = diag_sigma_r[:ndim]
                    _rpr = cfg.refine_prior_sigma_range
                    refine_bounds = [(best_theta[i] - diag_sigma_full_r[i]*_rpr,
                                      best_theta[i] + diag_sigma_full_r[i]*_rpr)
                                     for i in range(ndim)]
                    _de_gen = [0]
                    best_theta_r = best_theta  # safe default; overwritten below
                    # In intrinsic mode, extend refinement to also optimise
                    # Phi_phi0 and Phi_r0 jointly with the intrinsic params.
                    _refine_with_phase = (parameter_selected == 'intrinsic')
                    if _refine_with_phase:
                        _phi0_init = float(result_array['Phi_phi0'])
                        _phir0_init = float(result_array['Phi_r0'])
                        best_theta_r = np.append(best_theta, [_phi0_init, _phir0_init])
                        refine_bounds_r = refine_bounds + [(0.0, 2*np.pi), (0.0, 2*np.pi)]
                        def neg_obj(theta):
                            try:
                                return -float(calculate_detection_overlap(
                                    theta[0], theta[1], theta[2], theta[3], theta[4],
                                    ctx['Y0'], ctx['dist'], ctx['qS'], ctx['phiS'],
                                    ctx['qK'], ctx['phiK'],
                                    theta[5], ctx['Phi_theta0'], theta[6],
                                    add_kwargs, maximize_phase=False, **temp_dict))
                            except Exception:
                                return np.inf
                    else:
                        best_theta_r = best_theta
                        refine_bounds_r = refine_bounds
                        def neg_obj(theta):
                            try:
                                return -float(objective(theta))
                            except Exception:
                                return np.inf
                    # When neg_obj returns -overlap (7D phase path), it is in [0,1]
                    # while best_val is the PARIS score (~SNR*overlap).  Track a
                    # separate comparison baseline in the same units as neg_obj.
                    _refine_val = (-neg_obj(best_theta_r)
                                   if _refine_with_phase else best_val)
                    def _de_callback(xk, convergence):
                        _de_gen[0] += 1
                        if _de_gen[0] % 10 == 0:
                            print(f"{_ts()} [DE] gen={_de_gen[0]:3d}  "
                                  f"score={-neg_obj(xk):.6e}  convergence={convergence:.4f}")
                        return False
                    de_result = differential_evolution_optimize(
                        theta0=best_theta_r,
                        objective=neg_obj,
                        fisher_bounds=refine_bounds_r,
                        maxiter=cfg.de_refine_maxiter,
                        seed=seed,
                        init='latinhypercube',
                        popsize=cfg.de_refine_popsize,
                        workers=cfg.de_workers,
                    )
                    _de_elapsed = (time.time() - _t_de_start) / 60
                    if -de_result.fun > _refine_val:
                        best_theta_r = np.asarray(de_result.x, dtype=float)
                        _refine_val = -de_result.fun
                        print(f"{_ts()} [REFINE] DE improved score to {_refine_val:.6e}  "
                              f"({_de_elapsed:.1f} min, {de_result.nfev} evals)")
                    else:
                        print(f"{_ts()} [REFINE] DE did not improve "
                              f"({-de_result.fun:.6e} vs {_refine_val:.6e}, "
                              f"{_de_elapsed:.1f} min, {de_result.nfev} evals)")
                except Exception as exc:
                    import traceback
                    print(f"{_ts()} [WARN] DE refinement failed: {exc}\n{traceback.format_exc()}")

                print(f"\n{_ts()} {'='*55}")
                print(f"{_ts()} STAGE 3 (REFINE): Nelder-Mead "
                      f"(maxiter={cfg.nm_refine_maxiter})")
                print(f"{_ts()} {'='*55}")
                _t_nm_start = time.time()
                try:
                    nm_result = nelder_mead_optimize(
                        best_theta_r,
                        neg_obj,
                        maxiter=cfg.nm_refine_maxiter,
                        xatol=cfg.nm_xatol,
                        fatol=cfg.nm_fatol,
                    )
                    _nm_elapsed = (time.time() - _t_nm_start) / 60
                    if -nm_result.fun > _refine_val:
                        best_theta_r = np.asarray(nm_result.x, dtype=float)
                        _refine_val = -nm_result.fun
                        print(f"{_ts()} [REFINE] NM improved score to {_refine_val:.6e}  "
                              f"({_nm_elapsed:.1f} min, {nm_result.nfev} evals, "
                              f"converged={nm_result.success})")
                    else:
                        print(f"{_ts()} [REFINE] NM did not improve "
                              f"({-nm_result.fun:.6e} vs {_refine_val:.6e}, "
                              f"{_nm_elapsed:.1f} min, converged={nm_result.success})")

                    # Unpack best point — intrinsic params first, then phases if extended
                    for i, key in enumerate(starting_point_keys):
                        result_array[key] = best_theta_r[i]
                    if _refine_with_phase:
                        result_array['Phi_phi0'] = float(best_theta_r[5])
                        result_array['Phi_r0']   = float(best_theta_r[6])
                        print(f"[REFINE] Best phases: Phi_phi0={best_theta_r[5]:.6f}  "
                              f"Phi_r0={best_theta_r[6]:.6f}  "
                              f"(signal: {ctx['Phi_phi0']:.6f}, {ctx['Phi_r0']:.6f})")
                    add_kwargs['chi2'] = result_array['chi2']
                    final_overlap_refined = calculate_detection_overlap(
                        result_array['m1'], result_array['m2'], result_array['a'],
                        result_array['p0'], result_array['e0'], ctx['Y0'],
                        ctx['dist'], ctx['qS'], ctx['phiS'], ctx['qK'], ctx['phiK'],
                        result_array['Phi_phi0'], ctx['Phi_theta0'], result_array['Phi_r0'],
                        add_kwargs, maximize_phase=False, **temp_dict)
                    print(f"[REFINE] Final overlap after NM: {final_overlap_refined:.6f}")

                    refine_out = {
                        'optimizer': 'paris+de+nm',
                        'best_point': best_theta.tolist(),
                        'best_score': best_val,
                        'final_overlap': float(final_overlap_refined),
                    }
                    Config.save_results_with_config(
                        cfg=cfg,
                        results=refine_out,
                        save_dir=idx_dir,
                        filename_prefix=f"opt_refined_{target_func}_id_{id}")
                    np.save(os.path.join(idx_dir, f"results_refined_{id+1}_time_{timestamp}.npy"), result_array)
                    np.save(os.path.join(idx_dir, f"starting_point_{id+1}.npy"), result_array)

                except Exception as exc:
                    import traceback
                    print(f"[WARN] NM refinement failed: {exc}\n{traceback.format_exc()}")

        # ---------------------------
        # Global failure handler
        # ---------------------------
        except Exception as exc:
            print(f"[WARN] PARIS optimization failed: {exc}")

        return result_array
            

if __name__ == "__main__":
    # CLI args override config values — used by PBS array jobs to set per-element
    # grid index without editing the config file.
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument('--start', type=int, default=None)
    _parser.add_argument('--end',   type=int, default=None)
    _parser.add_argument('--refine-only', dest='refine_only', action='store_true',
                         help='Skip PARIS; load previous checkpoint and run DE+NM only')
    _parser.add_argument('--run-type', dest='run_type', default=None,
                         choices=['0pa_vs_2pa', '1pa_vs_2pa'],
                         help='Override config run_type (also updates param_names_to_infer)')
    _cli, _ = _parser.parse_known_args()

    cfg = Config()
    if _cli.run_type is not None:
        cfg.run_type = _cli.run_type
        cfg.param_names_to_infer = ['m1', 'm2', 'a', 'p0', 'e0', 'chi2'] \
            if cfg.run_type == '1pa_vs_2pa' else ['m1', 'm2', 'a', 'p0', 'e0']
        _pa = '0pa' if cfg.run_type == '0pa_vs_2pa' else '1pa'
        cfg.result_file = cfg.result_files[cfg.TYPE].replace('.npy', f'_{_pa}.npy')
        cfg.basedir = f"/scratch/josh.mat/opt_grid/results/{cfg.TYPE}_{_pa}/"
    print("Start")

    file_folder = cfg.param_file
    parameter_array =  np.load(file_folder)

    result_folder = cfg.result_file
    if not os.path.exists(result_folder):
        result_array = np.zeros_like(parameter_array)
        np.save(result_folder, result_array)
        print(f"[INFO] Created result array: {result_folder}")
    else:
        result_array = np.load(result_folder)

    param_names_to_infer = cfg.param_names_to_infer
    parameter_selected = cfg.parameter_selected

    optimizer = cfg.optimizer
    target_func = cfg.target_func

    base_dir = cfg.basedir
    TYPE = cfg.TYPE

    run_type = cfg.run_type
    nchannels = cfg.nchannels

    include_noise = cfg.include_noise
    prior_sigma_range = cfg.prior_sigma_range
    using_evec = cfg.using_evec
    seed= cfg.seed
    paris_conf = dict()

    paris_conf['spread_scale'] = cfg.spread_scale
    paris_conf['seed_cloud'] = cfg.seed_cloud
    paris_conf['paris_seed_n'] = cfg.paris_seed_n
    paris_conf['paris_temperature'] = cfg.paris_temperature

    def _find_latest_starting_point(base_dir_i, target_func):
        """Return the highest-numbered starting_point_N.npy from any previous
        PARIS run, or None if no previous run exists.  This lets a re-run warm-
        start PARIS around the previous best rather than the raw signal params."""
        import glob
        pattern = os.path.join(base_dir_i, f"paris_{target_func}_id_*",
                               "starting_point_*.npy")
        matches = glob.glob(pattern)
        if not matches:
            return None
        def _sp_num(p):
            m = re.search(r'starting_point_(\d+)\.npy$', p)
            return int(m.group(1)) if m else -1
        return max(matches, key=_sp_num)

    startindex = _cli.start if _cli.start is not None else cfg.start_index
    endindex   = _cli.end   if _cli.end   is not None else cfg.end_index
    print(f"[INFO] Grid range: [{startindex}, {endindex})  TYPE={TYPE}  run_type={run_type}")
    for i in range(startindex,endindex):
        paramter_selected = parameter_array[i]
        params = ["m1","m2","a","p0","e0","xI0","dist","qS","phiS","qK","phiK",
                         "Phi_phi0","Phi_theta0","Phi_r0","dt","T","chi2"]
        param_dict = dict(zip(params, paramter_selected))
        base_dir_i = os.path.join(base_dir, f"{TYPE}_{i}")
        os.makedirs(base_dir_i, exist_ok=True)
        prev_sp = _find_latest_starting_point(base_dir_i, target_func)
        if prev_sp is not None:
            starting_point_file = prev_sp
            print(f"[INFO] Warm-starting from previous result: {prev_sp}")
        else:
            starting_point_file = os.path.join(base_dir_i, "starting_point_0.npy")
            np.save(starting_point_file, param_dict)
    
        # target_func, optimizer,
        # n_channels, startingpoints_file)
        result_dict = main(signal_param_array=paramter_selected[0:14],
                           dt=paramter_selected[14],T=paramter_selected[15],chi2 = paramter_selected[16],
                           run_type=run_type,
                           parameter_selected = parameter_selected,
                           param_names_to_infer= param_names_to_infer,
                           base_dir = base_dir_i,
                           target_func=target_func, optimizer = optimizer,
                           n_channels = nchannels,
                           startingpoints_file=starting_point_file,
                           include_noise =include_noise,
                           prior_sigma_range = prior_sigma_range,
                           using_evec = using_evec,
                           paris_conf=paris_conf,seed=seed,
                           cfg = cfg,
                           cache_dir=os.path.join(cfg.fisher_cache_dir, cfg.TYPE),
                           grid_index=i,
                           refine_only=_cli.refine_only)
        result = list(result_dict.values())
        result_array[i] = result
        np.save(result_folder,result_array)





