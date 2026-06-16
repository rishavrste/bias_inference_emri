import os
import json
import time
from typing import Tuple, Optional
import re

import numpy as np
from scipy.optimize import minimize, differential_evolution

#few and SEF imports
from few.waveform import GenerateEMRIWaveform
from few.waveform.waveform import SuperKludgeWaveform

from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity, A2TDISens, E2TDISens, T2TDISens
from stableemrifisher.utils import generate_PSD, inner_product

from config_paris import Config, ObjectiveTracker
from misc import (
    _clip_physical_params_intrinsic,
    load_startingpoint_param_array,
    compute_fft_with_windowing,
    add_noise_func,
    calculate_detection_snr,
    calculate_detection_overlap,
    calculate_time_max,
    chi2_match,
    inner_prod,
    compute_fisher_parallelotope,
    covariance_from_fisher_parallelotope,
    plot_time_series_from_fft,
    check_noise_model_consistency,
)
import parismc
try:
    import cupy as cp
    xp = cp
except ImportError:
    xp = np
    print("CuPy not found, using NumPy instead. For GPU acceleration, please install CuPy.")

# -----------------------------
# PARIS global context (picklable functions require module scope)
# -----------------------------
_PARIS_REF_CENTER = None          # type: Optional[np.ndarray]
_PARIS_SPREAD_SCALE = None        # type: Optional[float]
_PARIS_OBJECTIVE = None           # type: Optional[callable]
_PARIS_TARGET_KIND = None         # type: Optional[str]  # 'optimal_snr', 'optimal_snr_phase_max', 'phase_match', 'time_max'
_TARGET_SNR = None
# Fisher-parallelotope affine prior (primary for this script)
_PARIS_AFFINE_CENTER = None       # type: Optional[np.ndarray]
_PARIS_AFFINE_Q = None            # type: Optional[np.ndarray]
_PARIS_AFFINE_B = None            # type: Optional[np.ndarray]
_PARIS_DIM = None                 # type: Optional[int]
_PARIS_USE_ELLIPSE = True

# Maps ndim -> ordered theta keys, shared by every optimizer's result-array mapping
# and by main()'s theta0/prior-center construction below.
PARAM_KEYS_BY_NDIM = {
    5: ['m1', 'm2', 'a', 'p0', 'e0'],
    6: ['m1', 'm2', 'a', 'p0', 'e0', 'chi2'],
    7: ['m1', 'm2', 'a', 'p0', 'e0', 'Phi_phi0', 'Phi_r0'],
    8: ['m1', 'm2', 'a', 'p0', 'e0', 'Phi_phi0', 'Phi_r0', 'chi2'],
}


def _theta_to_result_array(starting_point: dict, theta: np.ndarray, ndim: int) -> dict:
    """Map an optimizer's flat theta vector back onto the named starting-point dict."""
    result_array = starting_point.copy()
    for key, value in zip(PARAM_KEYS_BY_NDIM.get(ndim, []), theta):
        result_array[key] = value
    return result_array


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
            val = float(_PARIS_OBJECTIVE(x))
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

    sum_kwargs = dict(pad_output=True, odd_len=True)
    
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
        waveform_true_fft,noise = add_noise_func(waveform_true_fft_without_noise,PSD_funcs_,delta_f, dt,n_channels= nchannels,seed=seed)
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
    metric_func = {
        'optimal_snr': calculate_detection_snr,
        'optimal_snr_phase_max': calculate_detection_snr,
        'time_max': calculate_time_max,
        'chi2_match': chi2_match,
    }.get(target_func)
    if metric_func is None:
        raise ValueError(f"Unknown target_func: {target_func}")

    def score(theta: np.ndarray) -> float:
        """Unpack theta (5/6/7/8-D, depending on use_1PA x with_phase) and evaluate metric_func."""
        if with_phase:
            if use_1PA:
                m1, m2, a, p0, e0, Phi_phi0, Phi_r0, chi2 = theta
                add_kwargs['chi2'] = chi2
            else:
                m1, m2, a, p0, e0, Phi_phi0, Phi_r0 = theta
        else:
            Phi_phi0, Phi_r0 = ctx['Phi_phi0'], ctx['Phi_r0']
            if use_1PA:
                m1, m2, a, p0, e0, chi2 = theta
                add_kwargs['chi2'] = chi2
            else:
                m1, m2, a, p0, e0 = theta

        add_kwargs['evolve_1PA'] = bool(use_1PA)
        add_kwargs['evolve_2PA'] = False

        val = metric_func(
            m1, m2, a, p0, e0, ctx['Y0'], ctx['dist'], ctx['qS'], ctx['phiS'], ctx['qK'], ctx['phiK'],
            Phi_phi0, ctx['Phi_theta0'], Phi_r0, add_kwargs,
            maximize_phase=bool(phase_max),
            **fixed)
        return float(val)

    return score
        

def nelder_mead_optimize(theta0: np.ndarray, objective, maxiter: int = 3000, maxfev: int = 15000, xatol: float = 1e-10, fatol: float = 1e-12):
    res = minimize(
        objective,
        theta0,
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'maxfev': maxfev, 'xatol': xatol, 'fatol': fatol,'adaptive': True},
    )
    return res

from scipy.optimize import differential_evolution
def differential_evolution_optimize(theta0: np.ndarray, objective, maxiter: int = 1000, tol: float = 1e-4, atol: float = 1e-5, x0: Optional[np.ndarray] = None,
                                    fisher_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,init='sobol',seed: Optional[int] = 42,
                                    popsize: int = 15):
    if fisher_bounds is not None:
        bounds = fisher_bounds
    else:
        bounds = [(x*(1-1e-3), x*(1+1e-3)) for x in theta0]  # Define bounds around initial guess
    if theta0.shape[0] == 6 or theta0.shape[0] == 8:
        print("Earlier Bound are :", bounds)
        print("\nApplying special bounds for chi2\n")
        bounds[-1] = (-1, 1)
        print("Later Bound are :", bounds)
    # Clip x0 to bounds so scipy does not reject it
    x0_clipped = np.array([np.clip(theta0[i], bounds[i][0], bounds[i][1]) for i in range(len(theta0))])
    res = differential_evolution(
        func=objective,
        bounds=bounds,
        maxiter=maxiter,
        tol=tol,
        atol=atol,
        x0=x0_clipped,
        seed=seed,
        init=init,
        popsize=popsize,
    )
    return res

def run_paris(ndim: int,
              prior_center: np.ndarray,
              score_func,
              spread_scale: float,
              savepath: str,
              cfg: Config,
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
    global _PARIS_USE_ELLIPSE
    _PARIS_REF_CENTER = np.asarray(prior_center, dtype=float).copy()
    _PARIS_SPREAD_SCALE = float(spread_scale)
    _PARIS_OBJECTIVE = score_func
    _PARIS_TARGET_KIND = target_kind
    _PARIS_EARLY_STOP_HIT = False
    _PARIS_USE_ELLIPSE = bool(use_ellipse)

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


def main(signal_param_array,
        dt,T,chi2,
        run_type,
        parameter_selected,base_dir,param_names_to_infer,
        target_func, optimizer,
        n_channels, startingpoints_file,
        include_noise,
        prior_sigma_range,using_evec,
        paris_conf,
        seed,cfg, use_gpu=True, warm_start_prior=False):
    

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

    # Initialise result_array to the starting point so that if the optimizer crashes
    # before writing its own result, we return the initial guess rather than a NameError.
    result_array = starting_point.copy() if starting_point is not None else {}

    emri_kwargs = {"T": T, "dt": dt,'chi2': chi2,'evolve_1PA': True,'evolve_primary': False,'evolve_2PA': True}
    add_kwargs = {'chi2': chi2,'evolve_1PA': True,'evolve_primary': False,'evolve_2PA': True}
        
    ctx = prepare_true_waveform(signal_param_array, emri_kwargs, add_kwargs,add_noise=include_noise, use_gpu=use_gpu,seed=seed,nchannels=n_channels)

    snr_2 = inner_prod(ctx['waveform_true_fft_without_noise'], ctx['waveform_true_fft_without_noise'], ctx['PSD_funcs'], ctx['delta_f'], xp=xp)
    snr = np.sqrt(snr_2.get()) if hasattr(snr_2, "get") else np.sqrt(snr_2)
    _TARGET_SNR = snr
        
    temp_dict = {'waveform_true_fft': ctx['waveform_true_fft'], 'waveform_true_fft_without_noise': ctx['waveform_true_fft_without_noise'],
                  'PSD': ctx['PSD_funcs'], 'dt': ctx['dt'], 'T': ctx['T'],
                     'N_fiducial': ctx['N_fiducial'], 'delta_f': ctx['delta_f'], 'use_gpu': use_gpu,
                      'waveform_response': ctx['waveform_response'],'xp': cp if use_gpu else np}
    
    for k in ['m1', 'm2', 'a', 'p0', 'e0', 'Y0', 'dist', 'qS', 'phiS', 'qK', 'phiK', 'Phi_phi0', 'Phi_theta0', 'Phi_r0', 'chi2','dt', 'T']:
        assert k in ctx, f"Missing {k} in 1PA context"

    # Initial theta from startingpoint array if available, else from signal row.
    # ndim and which params are inferred is fully determined by (run_type, parameter_selected);
    # PARAM_KEYS_BY_NDIM (module level) gives the ordered keys for each ndim.
    RUN_SPEC = {
        ('0pa_vs_2pa', 'intrinsic'):       dict(ndim=5, evolve_1PA=False),
        ('0pa_vs_2pa', 'intrinsic_phase'): dict(ndim=7, evolve_1PA=False),
        ('1pa_vs_2pa', 'intrinsic'):       dict(ndim=6, evolve_1PA=True),
        ('1pa_vs_2pa', 'intrinsic_phase'): dict(ndim=8, evolve_1PA=True),
    }
    spec = RUN_SPEC.get((run_type, parameter_selected))
    if spec is None:
        raise ValueError(f"Unsupported run_type {run_type} with parameter_selected {parameter_selected}")
    ndim = spec['ndim']
    keys = PARAM_KEYS_BY_NDIM[ndim]
    add_kwargs['evolve_1PA'] = spec['evolve_1PA']
    add_kwargs['evolve_2PA'] = False

    source = starting_point if starting_point is not None else ctx
    theta0 = np.array([source[k] for k in keys], dtype=float)
    if run_type == '1pa_vs_2pa':
        add_kwargs['chi2'] = theta0[-1]

    # The reference overlap is always evaluated against the loaded starting point;
    # phase angles are only meaningful (i.e. inferred) when parameter_selected == "intrinsic_phase".
    if parameter_selected == "intrinsic_phase":
        Phi_phi0_ref, Phi_r0_ref = starting_point['Phi_phi0'], starting_point['Phi_r0']
    else:
        Phi_phi0_ref, Phi_r0_ref = ctx['Phi_phi0'], ctx['Phi_r0']
    initial_overlap = calculate_detection_overlap(
        m1=starting_point['m1'], m2=starting_point['m2'], a=starting_point['a'],
        p0=starting_point['p0'], e0=starting_point['e0'],
        Y0=ctx['Y0'], dist=ctx['dist'], qS=ctx['qS'], phiS=ctx['phiS'], qK=ctx['qK'], phiK=ctx['phiK'],
        Phi_phi0=Phi_phi0_ref, Phi_theta0=ctx['Phi_theta0'], Phi_r0=Phi_r0_ref,
        add_kwargs=add_kwargs, maximize_phase=False, **temp_dict)
    print("Current Overlap:", initial_overlap)

    def _wrap_pi(x):
        return ((x + np.pi) % (2.0 * np.pi)) - np.pi

    # PARIS prior center: by default use the Fisher evaluation point (2PA truth from ctx).
    # If warm_start_prior=True (Phase_N+1 rerun of a poor case), center on the previous
    # phase's best-found point instead — it is already closer to the 1PA optimal.
    if warm_start_prior and starting_point is not None:
        if parameter_selected == "intrinsic_phase":
            paris_prior_center = np.array(
                [_wrap_pi(starting_point[k]) if (run_type == '1pa_vs_2pa' and k in ('Phi_phi0', 'Phi_r0'))
                 else starting_point[k] for k in keys], dtype=float)
        elif run_type == '1pa_vs_2pa':
            paris_prior_center = np.array([starting_point[k] for k in keys[:-1]] + [chi2], dtype=float)
        else:
            paris_prior_center = np.array([starting_point[k] for k in keys], dtype=float)
        print(f"[WARM START] PARIS prior centered on previous best: {paris_prior_center}")
    elif run_type == '1pa_vs_2pa':
        paris_prior_center = np.array([ctx[k] for k in keys[:-1]] + [chi2], dtype=float)
    else:
        paris_prior_center = np.array([ctx[k] for k in keys], dtype=float)

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

    # Fisher prior uses the same model as the analysis template (1PA for 1PA runs, 0PA for 0PA runs).
    # Use signal_param_array (2PA truth) as the evaluation point: it always has high SNR so the
    # Fisher derivatives are well-conditioned, and the 2PA truth parameters are valid for 1PA too.
    fisher_add_kwargs = dict(add_kwargs)
    fisher_start_params = signal_param_array.copy()

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
    
                print(f"Optimized ({', '.join(PARAM_KEYS_BY_NDIM.get(ndim, []))}): {result.x}")
                result_array = _theta_to_result_array(starting_point, result.x, ndim)
                print(f"Optimized parameters as array: {result_array}")
                np.save(os.path.join(idx_dir, f"results_nelder_mead_{id+1}_time_{timestamp}.npy"), result_array)
                np.save(os.path.join(idx_dir, f"starting_point_{id+1}.npy"), result_array)

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

                return result_array

    
    elif optimizer == 'differential_evolution':
        tol = 1e-8
        theta_ref = theta0.copy()
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
                _TARGET_SNR=_TARGET_SNR,
                build_waveform_response=build_waveform_response,
            )
            diag_sigma = fisher_meta['diag_sigma']

            # Pad phase dims (Phi_phi0, Phi_r0) with ±pi half-width
            diag_sigma_full = list(diag_sigma)
            if len(diag_sigma_full) < ndim:
                n_intr = 5
                phase_sigma = np.pi / float(prior_sigma_range)
                while len(diag_sigma_full) < ndim - (1 if run_type == '1pa_vs_2pa' else 0):
                    diag_sigma_full.insert(n_intr, phase_sigma)
                if run_type == '1pa_vs_2pa':
                    diag_sigma_full.append(diag_sigma[n_intr] if len(diag_sigma) > n_intr else phase_sigma)

            # Fisher-based bounds centered on theta_ref (2PA truth for fresh starts)
            bounds = [(theta_ref[i] - diag_sigma_full[i]*prior_sigma_range,
                       theta_ref[i] + diag_sigma_full[i]*prior_sigma_range)
                      for i in range(ndim)]

            # Optional per-param sigma boosts (e.g. wider e0/m2 search)
            for param_idx, cfg_attr in [(4, 'e0_prior_sigma_factor'), (1, 'm2_prior_sigma_factor')]:
                factor = getattr(cfg, cfg_attr, 1.0)
                if factor != 1.0:
                    lo, hi = bounds[param_idx]
                    half = (hi - lo) / 2.0 * factor
                    bounds[param_idx] = (theta_ref[param_idx] - half, theta_ref[param_idx] + half)
                    print(f"[FISHER] DE param[{param_idx}] bounds boosted by {factor:.2f}x")

            # Hard physical limits (prevents FEW CUDA crashes for out-of-grid proposals)
            bounds[1] = (max(bounds[1][0], 1.0),  bounds[1][1])           # m2 > 0
            bounds[2] = (max(bounds[2][0], -0.99), min(bounds[2][1], 0.99))  # |a| < 1
            bounds[3] = (max(bounds[3][0], 6.5),  bounds[3][1])           # p0 above separatrix floor
            bounds[4] = (max(bounds[4][0], 1e-8),  min(bounds[4][1], 0.85)) # e0 in FEW grid

            _de_names = (list(param_names_to_infer[:5]) + ['Phi_phi0', 'Phi_r0'] +
                         (list(param_names_to_infer[5:]) if run_type == '1pa_vs_2pa' else []))
            print("Fisher-based bounds for optimization:")
            for i, (lo, hi) in enumerate(bounds):
                lbl = _de_names[i] if i < len(_de_names) else f'param_{i}'
                print(f"  {lbl}: [{lo:.6e}, {hi:.6e}]")

            def bounded_objective(theta: np.ndarray) -> float:
                try:
                    return -float(objective(_clip_physical_params_intrinsic(theta)))
                except Exception:
                    return 1e10

            de_threshold = getattr(cfg, 'nm_pre_de_threshold', 0.98)
            nm_pre_iters = getattr(cfg, 'nm_pre_de_maxiter', 0)
            nm_ran = False
            de_x0 = theta0

            # --- NM pre-polish (warm-start cases only) ---
            if nm_pre_iters > 0:
                print(f"[NM] running {nm_pre_iters} iterations from warm-start point...")
                nm_pre = nelder_mead_optimize(
                    theta0=theta0,
                    objective=bounded_objective,
                    maxiter=nm_pre_iters,
                    maxfev=nm_pre_iters * 2,
                    xatol=cfg.nm_xatol,
                    fatol=cfg.nm_fatol,
                )
                nm_score = -float(nm_pre.fun)
                # objective returns SNR, not overlap; normalise for threshold comparison
                nm_overlap_est = nm_score / _TARGET_SNR if _TARGET_SNR else nm_score
                print(f"[NM] best score: {nm_score:.4f}  overlap≈{nm_overlap_est:.6f}")
                tracker.update(nm_pre.x, nm_score)
                de_x0 = nm_pre.x
                nm_ran = True

            # --- DE (skipped if NM already reached threshold) ---
            run_de = (not nm_ran) or (nm_overlap_est < de_threshold)
            if run_de:
                if nm_ran:
                    # Recentre DE bounds on NM best (tighter, prior_sigma_range σ)
                    nm_ref = _clip_physical_params_intrinsic(nm_pre.x.copy())
                    bounds = [(nm_ref[i] - diag_sigma_full[i] * prior_sigma_range,
                               nm_ref[i] + diag_sigma_full[i] * prior_sigma_range)
                              for i in range(ndim)]
                    bounds[1] = (max(bounds[1][0], 1.0),   bounds[1][1])
                    bounds[2] = (max(bounds[2][0], -0.99), min(bounds[2][1], 0.99))
                    bounds[3] = (max(bounds[3][0], 6.5),   bounds[3][1])
                    bounds[4] = (max(bounds[4][0], 1e-8),  min(bounds[4][1], 0.85))
                    de_maxiter = cfg.de_maxiter              # 600 for post-NM
                    print(f"[DE] {prior_sigma_range}σ bounds around NM best, {de_maxiter} iters...")
                else:
                    de_maxiter = getattr(cfg, 'de_maxiter_cold', cfg.de_maxiter)  # 1000 for cold
                    print(f"[DE] cold-start {prior_sigma_range}σ bounds, {de_maxiter} iters...")

                result = differential_evolution_optimize(
                    theta0=de_x0,
                    objective=bounded_objective,
                    fisher_bounds=bounds,
                    maxiter=de_maxiter,
                    tol=tol,
                    seed=seed,
                )
            else:
                print(f"[DE] skipped — NM overlap≈{nm_overlap_est:.4f} >= {de_threshold}")
                result = nm_pre

            best_score = -float(result.fun)
            tracker.update(result.x, best_score)

            idx_dir = os.path.join(base_dir, f"differential_evolution_{target_func}_run_id_{id}")
            os.makedirs(idx_dir, exist_ok=True)

            # Map optimised vector back to named result_array
            result_array = _theta_to_result_array(starting_point, result.x, ndim)
            np.save(
                os.path.join(idx_dir, f"results_differential_evolution_{id+1}_time_{timestamp}.npy"),
                result_array,
            )

            add_kwargs['chi2'] = result_array['chi2']
            final_overlap = calculate_detection_overlap(
                result_array['m1'], result_array['m2'], result_array['a'],
                result_array['p0'], result_array['e0'],
                ctx['Y0'], ctx['dist'], ctx['qS'], ctx['phiS'], ctx['qK'], ctx['phiK'],
                result_array['Phi_phi0'], ctx['Phi_theta0'], result_array['Phi_r0'],
                add_kwargs, maximize_phase=False, **temp_dict,
            )
            print(f"Overlap of the best point: {final_overlap:.10f}")

            out = {
                'optimizer': 'differential_evolution',
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
                cfg=cfg, results=out, save_dir=idx_dir,
                filename_prefix=f"opt_differential_evolution_{target_func}_id_{id}",
            )

        except Exception as exc:
            print(f"[ERROR] Differential Evolution optimization failed: {exc}")

        return result_array
                
                
    elif optimizer == 'paris':
        # --- PARIS Optimization Block ---

        if True:
            # ---------------------------
            # Fisher prior computation
            # ---------------------------
            try:
                import cupy as cP
                USE_GPU = True
            except ImportError:
                USE_GPU = False

            if True:

                Q, b, fisher_meta = compute_fisher_parallelotope(
                    ctx=ctx,
                    params_to_infer=param_names_to_infer,
                    fisher_params=fisher_start_params,
                    use_gpu=USE_GPU,
                    prior_sigma_range=float(prior_sigma_range),
                    using_evec=using_evec,
                    additional_kwargs=fisher_add_kwargs,
                    _TARGET_SNR= _TARGET_SNR,
                    build_waveform_response= build_waveform_response

                )
                print("Fisher parallelotope computed successfully.")
                fisher_ok = True

                # When parameter_selected == "intrinsic_phase", ndim is 7 (0PA) or 8 (1PA)
                # but the Fisher covers intrinsic params only (5 or 6D).
                # Phase dims (Phi_phi0, Phi_r0) at theta indices 5–6 get a uniform ±pi prior.
                # Chi2 (if 1PA) sits at theta index 7 and gets its own Fisher sigma.
                if len(b) < ndim:
                    n_fisher = len(b)   # e.g. 6 for ['m1','m2','a','p0','e0','chi2']
                    n_intr = 5          # m1, m2, a, p0, e0 always first
                    b_padded = np.zeros(ndim)
                    b_padded[:n_intr] = b[:n_intr]   # intrinsic Fisher half-widths
                    # phase dims sit between the intrinsic dims and chi2
                    n_phase = ndim - n_fisher         # number of phase dims to pad
                    for ph in range(n_phase):
                        b_padded[n_intr + ph] = np.pi  # uniform ±pi
                    if run_type == '1pa_vs_2pa' and n_fisher > n_intr:
                        b_padded[ndim - 1] = b[n_intr]  # chi2 Fisher sigma at last index
                    Q_padded = np.eye(ndim)
                    Q_padded[:n_intr, :n_intr] = Q[:n_intr, :n_intr]
                    if run_type == '1pa_vs_2pa' and n_fisher > n_intr:
                        Q_padded[ndim-1, ndim-1] = Q[n_intr, n_intr]
                    Q, b = Q_padded, b_padded
                    print(f"[FISHER] Padded to ndim={ndim}: intrinsic={n_intr}D, "
                          f"phases={n_phase}D (±pi), chi2={'yes' if run_type=='1pa_vs_2pa' else 'no'}")

                # Fisher is computed at snr_model; if it was rescaled, shrink b accordingly.
                _snr_scale_paris = np.sqrt(float(fisher_meta.get('scale_applied', 1.0)))
                if _snr_scale_paris > 1.0:
                    b = b / _snr_scale_paris
                    print(f"[FISHER] PARIS b SNR-scaled by 1/{_snr_scale_paris:.2f} "
                          f"(snr_model={fisher_meta['snr_model']:.2f} → target SNR)")

                # Per-parameter sigma override: e0 (index 4) uses a wider prior
                _e0_sigma_factor = getattr(cfg, 'e0_prior_sigma_factor', 1.0)
                if _e0_sigma_factor != 1.0:
                    b[4] = b[4] * _e0_sigma_factor
                    print(f"[FISHER] e0 prior half-width boosted by {_e0_sigma_factor:.1f}x → b[4]={b[4]:.6e}")

            # ---------------------------
            # Directory setup
            # ---------------------------
            idx_dir = os.path.join(base_dir, f"paris_{target_func}_id_{id}")
            os.makedirs(idx_dir, exist_ok=True)

            savepath = os.path.join(idx_dir, f"paris_results_{target_func}_{timestamp}_id_{id}")
            lhs_seed_rel = "lhs_seed"
            lhs_seed_dir = os.path.join(idx_dir, lhs_seed_rel)

            # ---------------------------
            # Run PARIS optimizer
            # ---------------------------
            
            sampler, prior_transform, ext_points = run_paris(
                ndim=ndim,
                prior_center=paris_prior_center,
                score_func=objective,
                spread_scale=float(paris_conf['spread_scale']),
                savepath=savepath,
                cfg=cfg,
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
            # Extract best point
            # ---------------------------
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

            rng = np.random.default_rng()
            ndim_local = len(best_theta)

            # Pad Qp, bp to ndim_local if Fisher was computed on fewer dims (phases excluded)
            if len(bp) < ndim_local:
                n_fisher = len(bp); n_intr = 5
                bp_padded = np.zeros(ndim_local)
                bp_padded[:n_intr] = bp[:n_intr]
                n_phase = ndim_local - n_fisher
                for ph in range(n_phase):
                    bp_padded[n_intr + ph] = np.pi
                if run_type == '1pa_vs_2pa' and n_fisher > n_intr:
                    bp_padded[ndim_local - 1] = bp[n_intr]
                Qp_padded = np.eye(ndim_local)
                Qp_padded[:n_intr, :n_intr] = Qp[:n_intr, :n_intr]
                if run_type == '1pa_vs_2pa' and n_fisher > n_intr:
                    Qp_padded[ndim_local - 1, ndim_local - 1] = Qp[n_intr, n_intr]
                Qp, bp = Qp_padded, bp_padded

            cov = covariance_from_fisher_parallelotope(
                Qp, bp, prior_sigma_range=float(prior_sigma_range)
            )

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

            # PARIS no-decrease guard: if PARIS ended up worse than the starting point, revert.
            _initial_score_estimate = float(initial_overlap) * float(_TARGET_SNR) * 50.0
            if best_val < _initial_score_estimate:
                print(f"[REVERT] PARIS score {best_val:.2f} < starting estimate {_initial_score_estimate:.2f} "
                      f"(initial_overlap={initial_overlap:.4f}); reverting to theta0")
                best_theta = theta0.copy()
                best_val = float(objective(best_theta))
                tracker.update(best_theta, best_val)

            # ---------------------------
            # Post-PARIS refinement: DE then Nelder-Mead (only if PARIS overlap < 0.99)
            # ---------------------------
            _paris_overlap = best_val / (_TARGET_SNR * 50.0)
            print(f"[REFINE CHECK] PARIS best overlap estimate: {_paris_overlap:.4f}")
            if cfg.refine_after_paris and _paris_overlap < 0.99:
                best_val_r = best_val
                best_theta_r = best_theta.copy()

                def neg_obj_r(theta):
                    try:
                        return -float(objective(theta))
                    except Exception:
                        return np.inf

                # Build refinement bounds: intrinsic dims from Fisher, phase dims ±1 rad, chi2 from Fisher
                diag_sigma_r = np.asarray(fisher_meta['diag_sigma'])
                _snr_scale_r = np.sqrt(float(fisher_meta.get('scale_applied', 1.0)))
                if _snr_scale_r > 1.0:
                    diag_sigma_r = diag_sigma_r / _snr_scale_r
                _rpr = cfg.refine_prior_sigma_range
                n_intr = 5  # m1, m2, a, p0, e0
                n_fisher = len(diag_sigma_r)  # 5 (0PA) or 6 (1PA)
                refine_bounds = [(best_theta_r[i] - diag_sigma_r[i]*_rpr,
                                  best_theta_r[i] + diag_sigma_r[i]*_rpr)
                                 for i in range(n_intr)]
                # phase dims (indices n_intr .. ndim-2 for 1PA, n_intr .. ndim-1 for 0PA)
                n_phase = ndim - n_fisher
                for ph_idx in range(n_intr, n_intr + n_phase):
                    refine_bounds.append((best_theta_r[ph_idx] - 1.0, best_theta_r[ph_idx] + 1.0))
                # chi2 (last dim for 1PA)
                if run_type == '1pa_vs_2pa' and n_fisher > n_intr:
                    refine_bounds.append((best_theta_r[ndim-1] - diag_sigma_r[n_intr]*_rpr,
                                          best_theta_r[ndim-1] + diag_sigma_r[n_intr]*_rpr))

                print(f"[REFINE] Starting {ndim}D refinement from PARIS best: score={best_val_r:.6e}")

                # Stage 2: DE
                try:
                    de_result = differential_evolution_optimize(
                        theta0=best_theta_r,
                        objective=neg_obj_r,
                        fisher_bounds=refine_bounds,
                        maxiter=cfg.de_refine_maxiter,
                        seed=seed,
                        init='latinhypercube',
                        popsize=cfg.de_refine_popsize,
                    )
                    if -de_result.fun > best_val_r:
                        best_theta_r = np.asarray(de_result.x, dtype=float)
                        best_val_r = -de_result.fun
                        print(f"[REFINE] DE improved score to {best_val_r:.6e}")
                    else:
                        print(f"[REFINE] DE did not improve ({-de_result.fun:.6e} vs {best_val_r:.6e})")
                except Exception as exc:
                    import traceback as _tb
                    print(f"[WARN] DE refinement failed: {exc}\n{_tb.format_exc()}")

                # Stage 3: NM
                try:
                    nm_result = nelder_mead_optimize(
                        best_theta_r,
                        neg_obj_r,
                        maxiter=cfg.nm_refine_maxiter,
                        maxfev=cfg.nm_refine_maxfev,
                        xatol=cfg.nm_xatol,
                        fatol=cfg.nm_refine_fatol,
                    )
                    if -nm_result.fun > best_val_r:
                        best_theta_r = np.asarray(nm_result.x, dtype=float)
                        best_val_r = -nm_result.fun
                        print(f"[REFINE] NM improved score to {best_val_r:.6e} "
                              f"(converged={nm_result.success})")
                    else:
                        print(f"[REFINE] NM did not improve ({-nm_result.fun:.6e} vs {best_val_r:.6e})")
                except Exception as exc:
                    import traceback as _tb
                    print(f"[WARN] NM refinement failed: {exc}\n{_tb.format_exc()}")

                if best_val_r > best_val:
                    best_theta = best_theta_r
                    best_val = best_val_r
                    tracker.update(best_theta, best_val)
                    print(f"[REFINE] Final best score after refinement: {best_val:.6e}")
                else:
                    print(f"[REFINE] Refinement did not improve over PARIS polish; keeping original.")

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

            print(f"Optimized ({', '.join(PARAM_KEYS_BY_NDIM.get(ndim, []))}): {best_theta}")
            result_array = _theta_to_result_array(starting_point, best_theta, ndim)
            print(f"Optimized parameters as array: {result_array}")
            np.save(os.path.join(idx_dir, f"results_paris_{id+1}_time_{timestamp}.npy"), result_array)
            np.save(os.path.join(idx_dir, f"starting_point_{id+1}.npy"), result_array)

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

        return result_array
            

if __name__ == "__main__":
    cfg = Config()
    print("Start")

    file_folder = cfg.param_file
    parameter_array =  np.load(file_folder)

    result_folder = cfg.result_file
    if os.path.exists(result_folder):
        result_array = np.load(result_folder, allow_pickle=True)
    else:
        result_array = parameter_array.copy()
        np.save(result_folder, result_array)

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

    startindex = cfg.start_index
    endindex = cfg.end_index
    prev_basedir = getattr(cfg, 'prev_basedir', None)
    good_threshold = getattr(cfg, 'good_overlap_threshold', 0.95)
    warm_start_threshold = getattr(cfg, 'warm_start_threshold', 0.0)
    print(f"Running cases START_INDEX={startindex} to END_INDEX={endindex}")

    for i in range(startindex,endindex):
        paramter_selected = parameter_array[i]
        params = ["m1","m2","a","p0","e0","xI0","dist","qS","phiS","qK","phiK",
                         "Phi_phi0","Phi_theta0","Phi_r0","dt","T","chi2"]
        param_dict = dict(zip(params, paramter_selected))
        base_dir_i = os.path.join(base_dir, f"{TYPE}_{i}")
        starting_point_file = os.path.join(base_dir_i, "starting_point_0.npy")
        print("Starting point file:", starting_point_file)
        os.makedirs(base_dir_i, exist_ok=True)
        # Default starting point is the 2PA truth; may be overridden below for warm starts
        np.save(starting_point_file, param_dict)

        # --- Check previous run: skip if already good, warm-start if poor ---
        skipped = False
        warm_start_prior = False
        if prev_basedir is not None:
            import glob as _glob
            # Support a list of prev_basedirs — search all and take best overlap
            _prev_basedirs = prev_basedir if isinstance(prev_basedir, list) else [prev_basedir]
            prev_jsons = []
            for _pbd in _prev_basedirs:
                _pcd = os.path.join(_pbd, f"{TYPE}_{i}")
                if os.path.isdir(_pcd):
                    prev_jsons += _glob.glob(os.path.join(_pcd, "**", "opt_PARIS_*_id_*.json"), recursive=True)
                    prev_jsons += _glob.glob(os.path.join(_pcd, "**", "opt_nelder_mead_*_id_*.json"), recursive=True)
                    prev_jsons += _glob.glob(os.path.join(_pcd, "**", "opt_differential_evolution_*_id_*.json"), recursive=True)
            if prev_jsons:
                # Pick JSON with highest final_overlap (not most recently modified)
                prev_overlap = 0.0
                best_prev_json = None
                best_prev_json_dir = None
                for _pj in prev_jsons:
                    try:
                        with open(_pj) as _f:
                            _d = json.load(_f)
                        _r = _d.get('results', _d)
                        _ov = float(_r.get('final_overlap', 0.0))
                        if _ov > prev_overlap:
                            prev_overlap = _ov
                            best_prev_json = _pj
                            best_prev_json_dir = os.path.dirname(_pj)
                    except Exception:
                        pass
                try:
                    print(f"[SKIP CHECK] Case {i}: best previous final_overlap={prev_overlap:.4f} "
                          f"(threshold={good_threshold}) from {best_prev_json}")
                    if prev_overlap >= good_threshold:
                        print(f"[SKIP] Case {i}: previous result is good (overlap={prev_overlap:.4f} "
                              f">= {good_threshold}). Reusing result, skipping re-run.")
                        prev_npy = _glob.glob(os.path.join(best_prev_json_dir, "results_paris_*.npy"))
                        prev_npy += _glob.glob(os.path.join(best_prev_json_dir, "results_nelder_mead_*.npy"))
                        prev_npy += _glob.glob(os.path.join(best_prev_json_dir, "results_differential_evolution_*.npy"))
                        if prev_npy:
                            prev_result = np.load(max(prev_npy, key=os.path.getmtime), allow_pickle=True).item()
                            result_array[i] = list(prev_result.values())
                            np.save(result_folder, result_array)
                        skipped = True
                    elif prev_overlap >= warm_start_threshold:
                        # Moderate case: warm-start PARIS from best previous result
                        prev_npy = _glob.glob(os.path.join(best_prev_json_dir, "results_paris_*.npy"))
                        prev_npy += _glob.glob(os.path.join(best_prev_json_dir, "results_nelder_mead_*.npy"))
                        prev_npy += _glob.glob(os.path.join(best_prev_json_dir, "results_differential_evolution_*.npy"))
                        if prev_npy:
                            prev_best_npy = max(prev_npy, key=os.path.getmtime)
                            prev_best = np.load(prev_best_npy, allow_pickle=True).item()
                            np.save(starting_point_file, prev_best)
                            warm_start_prior = True
                            print(f"[WARM START] Case {i}: prior centered on previous best "
                                  f"(overlap={prev_overlap:.4f}) from {prev_best_npy}")
                        else:
                            print(f"[WARM START] Case {i}: no results_npy found in {best_prev_json_dir}; using 2PA truth")
                    else:
                        # Poor case (below warm_start_threshold): start fresh from 2PA truth
                        print(f"[FRESH START] Case {i}: previous overlap {prev_overlap:.4f} < "
                              f"warm_start_threshold {warm_start_threshold:.2f}; using 2PA truth")
                except Exception as _e:
                    print(f"[SKIP CHECK] Could not process previous result for case {i}: {_e}")

        if skipped:
            continue

        # Use a wider prior for cold starts, tighter for warm starts.
        prior_sigma_range_cold = getattr(cfg, 'prior_sigma_range_cold', prior_sigma_range)
        case_sigma_range = prior_sigma_range if warm_start_prior else prior_sigma_range_cold

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
                           prior_sigma_range = case_sigma_range,
                           using_evec = using_evec,
                           paris_conf=paris_conf,seed=seed,
                           cfg = cfg,
                           warm_start_prior=warm_start_prior)
        result = list(result_dict.values())
        print(result)
        result_array[i] = result
        np.save(result_folder,result_array)





