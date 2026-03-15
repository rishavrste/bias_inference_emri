import os
import json
import time
import argparse
from typing import Tuple, Optional

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
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens
from stableemrifisher.utils import generate_PSD, inner_product
from stableemrifisher.fisher import StableEMRIFisher


from src.config_paris import Config, ObjectiveTracker
from misc import _is_pos_def, check_and_clip_prior, _clip_physical_params_intrinsic,compute_fft_with_windowing, calculate_optimal_snr_0pa_vs_1pa, compute_fisher_parallelotope,covariance_from_fisher_parallelotope

# -----------------------------
# PARIS global context (picklable functions require module scope)
# -----------------------------..l
_PARIS_REF_CENTER = None          # type: Optional[np.ndarray]
_PARIS_SPREAD_SCALE = None        # type: Optional[float]
_PARIS_OBJECTIVE = None           # type: Optional[callable]
_PARIS_TARGET_KIND = None         # type: Optional[str]  # 'optimal_snr', 'optimal_snr_phase_max', 'phase_match'
_PARIS_EARLY_STOP_HIT = False

# Fisher-parallelotope affine prior (primary for this script)
_PARIS_AFFINE_CENTER = None       # type: Optional[np.ndarray]
_PARIS_AFFINE_Q = None            # type: Optional[np.ndarray]
_PARIS_AFFINE_B = None            # type: Optional[np.ndarray]
_PARIS_DIM = None                 # type: Optional[int]
_PARIS_USE_ELLIPSE = True

# Target SNR for Fisher scaling
_TARGET_SNR = Config()._TARGET_SNR

params_to_infer = Config().param_names_to_infer

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
    """Top-level log-density (actually score) wrapper with early-stop.

    - Receives physical parameters (after prior_transform) per parismc contract.
    - Returns a scalar/array of scores (larger is better). After early-stop trigger,
      subsequent evaluations return -inf to end sampling quickly.
    """
    global _PARIS_EARLY_STOP_HIT

    params = np.asarray(params)

    def eval_one(x):
        global _PARIS_EARLY_STOP_HIT
        if _PARIS_EARLY_STOP_HIT:
            return float('-inf')
        try:    
            val = float(_PARIS_OBJECTIVE(x))
        except Exception:
            return float('-inf')
        # Early-stop policy per user spec
        if _PARIS_TARGET_KIND in ('optimal_snr', 'optimal_snr_phase_max'):
            if val >= 19.0:
                _PARIS_EARLY_STOP_HIT = True
                try:
                    print(f"[EARLY-STOP] SNR {val:.6f} >= 19; future calls => -inf")
                except Exception:
                    pass
        elif _PARIS_TARGET_KIND == 'phase_match':
            # score = -phase_diff; trigger when phase_diff < 1.5 => score > -1.5
            if val > -1.5:
                _PARIS_EARLY_STOP_HIT = True
                try:
                    print(f"[EARLY-STOP] phase-diff {-val:.6f} < 1.5; future calls => -inf")
                except Exception:
                    pass
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
    waveform_model = GenerateEMRIWaveform(SuperKludgeWaveform, sum_kwargs=sum_kwargs, return_list=False)

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
      #  use_gpu=use_gpu,
        is_ecliptic_latitude=False,
        remove_garbage="zero",
        orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
        order=order,
        tdi=tdi_gen,
        tdi_chan="AET",
    )
    print("[INFO] Finished loading modules and building ResponseWrapper")
    return response

def prepare_true_waveform(signal_row: np.ndarray, add_kwargs: dict, use_gpu: bool = False):
    """
    Build fiducial 1PA waveform, PSD, and FFT from a signal parameter row.

    signal_row columns:
      [m1, m2, a, p0, e0, Y0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, dt, T, chi2]
    """
    (
        m1, m2, a, p0, e0, Y0,
        dist, qS, phiS, qK, phiK,
        Phi_phi0, Phi_theta0, Phi_r0,
        dt, T
    ) = signal_row

    waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=use_gpu)

    chi2 = add_kwargs.get('chi2')
    deviation_included = add_kwargs.get('deviation_included', True)
    evolve_1PA = add_kwargs.get('evolve_1PA',)
    evolve_primary = add_kwargs.get('evolve_primary', False)
    evolve_2PA = add_kwargs.get('evolve_2PA',False)
    dev_1 = add_kwargs.get('dev_a')
    dev_2 = add_kwargs.get('dev_b')

    wave_params = [
        m1, m2, a, p0, e0, Y0,
        dist, qS, phiS, qK, phiK,
        Phi_phi0, Phi_theta0, Phi_r0,chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included, dev_1, dev_2
    ]
    
    emri_kwargs = {"T": T, "dt": dt, '1PA': evolve_1PA,'chi2': chi2, 'evolve_primary': evolve_primary,
                    '2PA': evolve_2PA,'deviation_included': deviation_included,'dev_a': dev_1, 'dev_b': dev_2}

    waveform_true = waveform_response(*wave_params, **emri_kwargs)

    channels = [A1TDISens, E1TDISens, T1TDISens]
    noise_kwargs = [{"sens_fn": ch} for ch in channels]
    PSD_funcs = generate_PSD(
        waveform=waveform_true,
        dt=dt,
        noise_PSD=get_sensitivity,
        channels=channels,
        noise_kwargs=noise_kwargs,
        use_gpu=use_gpu,
    )

    # Verify SNR level (grid builder normalized dist to target PA2 SNR already)
    snr = float(np.sqrt(inner_product(waveform_true, waveform_true, PSD_funcs, dt, use_gpu=use_gpu)))
    print(f"[TRUE] SNR: {snr:.6f}")

    waveform_true_fft = compute_fft_with_windowing(waveform_true, dt, use_gpu=use_gpu)
    N_fiducial = len(waveform_true[0])
    print("[INFO] Finished preparing true waveform (GPU)")

    return {
        'm1': m1, 'm2': m2, 'a': a, 'p0': p0, 'e0': e0, 'Y0': Y0,
        'dist': dist, 'qS': qS, 'phiS': phiS, 'qK': qK, 'phiK': phiK,
        'Phi_phi0': Phi_phi0, 'Phi_theta0': Phi_theta0, 'Phi_r0': Phi_r0,
        'dt': dt, 'T': T, 'chi2': chi2,
        'dev_1': dev_1, 'dev_2': dev_2,
        'waveform_response': waveform_response,
        'PSD_funcs': PSD_funcs,
        'waveform_true_fft': waveform_true_fft,
        'N_fiducial': N_fiducial,
        'snr': snr,
    }

# this need a lot of work to be implemented
def objective_factory(target_func: str,
                      ctx: dict,
                      phase_max: bool = False,
                      use_gpu_for_snr: bool = True,
                      infer_deviation_included: bool = False,
                      only_intrinsic_params: bool = False,
                      add_args: dict = None) -> callable:
    """
    Build a score(theta) where larger is better for all targets.

    - 'optimal_snr' and 'optimal_snr_phase_max': score = optimal SNR (maximize)
    - 'phase_match': score = -phase_diff_metric (maximize score => minimize phase diff)
    """
    # Only needed for SNR-based objective
    if target_func in ('optimal_snr', 'optimal_snr_phase_max'):
        fixed = {
            'waveform_response': ctx['waveform_response'],
            'PSD': ctx['PSD_funcs'],
            'dt': ctx['dt'],
            'T': ctx['T'],
            'N_fiducial': ctx['N_fiducial'],
            'waveform_true_fft': ctx['waveform_true_fft'],
            'xp': np,
            'use_gpu': bool(use_gpu_for_snr),
        }

    def score_optimal_snr(theta: np.ndarray) -> float:
   
   
        if only_intrinsic_params == True:
            m1, m2, a, p0, e0 = theta[:5]
            val = calculate_optimal_snr_0pa_vs_1pa(
                m1, m2, a, p0, e0, ctx['Y0'],ctx['dist'],ctx['qS'],ctx['phiS'], ctx['qK'], ctx['phiK'], 
                ctx['Phi_phi0'], ctx['Phi_theta0'], ctx['Phi_r0'],add_args,
                maximize_phase=bool(phase_max),
                **fixed,
            )
        else:
            if infer_deviation_included:
                m1, m2, a, p0, e0,qS,phiS,Phi_phi0,Phi_r0,dev_1,dev_2 = theta[:6]
                add_args['dev_1'] = dev_1
                add_args['dev_2'] = dev_2
                val = calculate_optimal_snr_0pa_vs_1pa(
                    m1, m2, a, p0, e0, ctx['Y0'],ctx['dist'],qS,phiS, ctx['qK'], ctx['phiK'], 
                    Phi_phi0, ctx['Phi_theta0'], Phi_r0,add_args,
                    maximize_phase=bool(phase_max),
                    **fixed)
                print(val, 'param', repr(theta))
        # Score is the SNR itself (maximize)
        return float(val)

    # Phase-match uses trajectory frequency metric from phase-match.py logic
    # Here we re-implement a light-weight metric using FEW's fundamental frequencies.
  

    # Precompute 1PA Omega_phi(t) interpolation and frequency weights
    # Build 1PA trajectory from the signal row
    SK_traj_1PA = EMRIInspiral(func=KerrEccEqFlux)  # For 0PA vs 1PA metric we still compare to the same t-grid
    # Note: We only need 1PA ref already embedded in ctx? We rebuild Omega_phi_1PA curve here robustly
    # However ctx doesn’t include (t,p,e,x) of 1PA; reconstruct for phase metric
    from few.trajectory.inspiral import EMRIInspiral as EMRIInspiralFull
    from few.trajectory.ode.flux import SuperKludgeFlux

    SK_traj_true = EMRIInspiralFull(func=SuperKludgeFlux)

    # do we need to implement it inside the function

    traj_ref = SK_traj_true.get_inspiral(
        ctx['m1'], ctx['m2'], ctx['a'], ctx['p0'], ctx['e0'], ctx['Y0'],
        ctx['chi2'], True, False, False,False,False,0,0,Phi_phi0 = ctx['Phi_phi0'], Phi_theta0 = ctx['Phi_theta0'], Phi_r0 = ctx['Phi_r0'],
        T=ctx['T'], dt=ctx['dt'], err=1e-11, DENSE_STEPPING=False,
        buffer_length=1000, integrate_backwards=False,
        max_step_size=None,
    )
    t_ref, p_ref, e_ref, x_ref = traj_ref[0], traj_ref[1], traj_ref[2], traj_ref[3]
    Omega_phi_ref, _, _ = get_fundamental_frequencies(ctx['a'], p_ref, e_ref, x_ref)
    # Common time grid and 1PA omega interpolation
    t_common = np.linspace(t_ref.min(), t_ref.max(), 1000)
    Omega_phi_1PA_interp = CubicSpline(t_ref, Omega_phi_ref)(t_common)

    # Frequency weighting w(t) ~ sum 1/S_n(f_gw)
    m_mode = 2
    Msec = (ctx['m1'] + ctx['m2']) * MTSUN_SI
    Omega2_SI = Omega_phi_1PA_interp / Msec
    f_gw = m_mode * Omega2_SI / (2.0 * np.pi)
    w = np.zeros_like(f_gw)
    for ch in (A1TDISens, E1TDISens, T1TDISens):
        Sn = get_sensitivity(f_gw, sens_fn=ch)
        Sn = np.maximum(Sn, 1e-60)
        w += 1.0 / Sn

    def phase_metric_for_theta(theta: np.ndarray) -> float:
        # Supports both 0PA against 1PA reference
        if only_intrinsic_params == True:
            m1, m2, a, p0, e0 = theta[:5]
            add_args['evolve_1PA'] = False
            use_1pa = False
        else:
            if infer_deviation_included:
                m1, m2, a, p0, e0,qS,phiS,Phi_phi0,Phi_r0,dev_1,dev_2 = theta[:6]
                add_args['dev_1'] = dev_1
                add_args['dev_2'] = dev_2
                add_args['deviation_included'] = True
                add_args['evolve_1PA'] = False
                use_1pa = False
            else:
                m1, m2, a, p0, e0,qS,phiS,Phi_phi0,Phi_r0 = theta[:6]
                add_args['dev_1'] = dev_1
                add_args['dev_2'] = dev_2
                add_args['deviation_included'] = True
                add_args['evolve_1PA'] = False
                use_1pa = False

        # Build trajectory for the chosen template order (0PA or 1PA) without 2PA
        SK_traj_tmpl = EMRIInspiral(func=KerrEccEqFlux)

        traj_0 = SK_traj_tmpl.get_inspiral(
            m1, m2, a, p0, e0, ctx['Y0'], add_args['chi2'], use_1pa, False,  False, False,False,False,add_args['dev_1'],add_args['dev_1'],
            Phi_phi0 = Phi_phi0, Phi_theta0 = ctx['Phi_theta0'], Phi_r0 = Phi_r0,
        T=ctx['T'], dt=ctx['dt'], err=1e-11, DENSE_STEPPING=False,
        buffer_length=1000, integrate_backwards=False,
        max_step_size=None,)

        t0, p0_arr, e0_arr, x0 = traj_0[0], traj_0[1], traj_0[2], traj_0[3]
        Omega_phi_0, _, _ = get_fundamental_frequencies(a, p0_arr, e0_arr, x0)
        # Interpolate to common grid
        Omega_phi_0_interp = CubicSpline(t0, Omega_phi_0)(t_common)
        # Weighted phase difference metric
        dOmega_geo = Omega_phi_0_interp - Omega_phi_1PA_interp
        dOmega_SI = dOmega_geo / Msec
        dphi = np.concatenate([[0.0], m_mode * cumulative_trapezoid(dOmega_SI, t_common)])
        num = np.trapz(w * dphi**2, t_common)
        den = np.trapz(w, t_common)
        res = np.sqrt(num / den)
        # Print theta as a copyable NumPy array with high precision
        theta_repr = "np.array(" + np.array2string(
            theta,
            formatter={'float_kind': lambda x: f"{x:.12g}"},
            separator=', '
        ) + ")"
        print(f"{res:.6g}", 'param', theta_repr)
        return res

    def score_phase_match(theta: np.ndarray) -> float:
        # Lower phase difference should be better => use negative for a larger-is-better score
        return -float(phase_metric_for_theta(theta))

    if target_func in ('optimal_snr', 'optimal_snr_phase_max'):
        return score_optimal_snr
    elif target_func == 'phase_match':
        return score_phase_match
    else:
        raise ValueError(f"Unknown target_func: {target_func}")
    

def nelder_mead_optimize(theta0: np.ndarray, objective, maxiter: int = 300, xatol: float = 1e-10, fatol: float = 1e-12): #300
    res = minimize(
        objective,
        theta0,
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'maxfev': 300, 'xatol': xatol, 'fatol': fatol},
    )
    return res

def run_paris(ndim: int,
              prior_center: np.ndarray,
              score_func,
              spread_scale: float,
              savepath: str,
              seed_cloud: int = 1000,
              seed_jitter: float = 1e-10,
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
    import parismc

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
    n_seed = 10
    sigma = 1e-3
    init_cov_list = [sigma**2 * np.eye(ndim) for _ in range(n_seed)]
    config = parismc.SamplerConfig(
        proc_merge_prob=0.9,
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

    max_idx = int(np.argmax(external_lhs_log_densities))
    max_point = external_lhs_points[max_idx]
    fallback_point = paris_prior_transform(max_point)
    fallback_score = float(external_lhs_log_densities[max_idx])
    sampler._fallback_best_point = np.asarray(fallback_point, dtype=float)
    sampler._fallback_best_score = fallback_score

    try:
        sampler.run_sampling(
            num_iterations=2000,
            savepath=savepath,
            print_iter=100,
            external_lhs_points=external_lhs_points,
            external_lhs_log_densities=external_lhs_log_densities,
        )
    except Exception as exc:
        print(f"[WARN] PARIS sampling failed: {exc}")
    return sampler, paris_prior_transform, external_lhs_points

def main():

    signal_param_array = Config.signal
    optimizer = Config.optimizer
    target_func = Config.target_func
    dt = Config.dt
    T = Config.T
    chi2= Config.chi2
    dev_1 = Config.dev_1
    dev_2 = Config.dev_2

    timestamp = time.strftime('%Y%m%d-%H%M%S')

    # Create a base output directory to store all artifacts for this run
    base_dir = f"opt_{optimizer}_{target_func}_{timestamp}"
    os.makedirs(base_dir, exist_ok=True)

    print('Preparing true waveform or phase')
    # Collect optimized rows and JSON summaries for optional batch save (num, 17)
   
    if target_func in ('optimal_snr', 'optimal_snr_phase_max'):
        ctx = prepare_true_waveform(signal_param_array, use_gpu=True)
    else:
            (
                m1, m2, a, p0, e0, Y0,
                dist, qS, phiS, qK, phiK,
                Phi_phi0, Phi_theta0, Phi_r0, 
            ) = signal_param_array,
            ctx = {
                'm1': m1, 'm2': m2, 'a': a, 'p0': p0, 'e0': e0, 'Y0': Y0,
                'dist': dist, 'qS': qS, 'phiS': phiS, 'qK': qK, 'phiK': phiK,
                'Phi_phi0': Phi_phi0, 'Phi_theta0': Phi_theta0, 'Phi_r0': Phi_r0,
                'dt': dt, 'T': T, 'chi2': chi2, 'dev_1': dev_1, 'dev_2': dev_2
            }

    # pack some keys explicitly needed later in objective
    for k in ['m1', 'm2', 'a', 'p0', 'e0', 'Y0', 'chi2', 'dist', 'qS', 'phiS', 'qK', 'phiK', 'Phi_phi0', 'Phi_theta0', 'Phi_r0', 'dt', 'T','dev_1','dev_2']:
        assert k in ctx, f"Missing {k} in 1PA context"

    # Initial theta from startingpoint array if available, else from signal row
    if Config.run_type == '0pa_vs_1pa' and Config.parameter_selected == "intrinsic":
            theta_names = ['m1', 'm2', 'a', 'p0', 'e0']
            theta0 = np.array([m1,m2,a,p0,e0], dtype=float)
            ndim = 5
    elif Config.run_type == '0pa_vs_1pa_dev' and Config.parameter_selected == "intrinsic":
            assert "Not implemented yet: deviation_included=True with intrinsic-only inference"

    elif Config.run_type == '0pa_vs_1pa' and Config.parameter_selected == "extrinsic":
        theta_names = ['m1', 'm2', 'a', 'p0', 'e0','qS', 'phiS', 'Phi_phi0', 'Phi_r0']
        theta0 = np.array([m1, m2, a, p0, e0,qS, phiS, Phi_phi0, Phi_r0], dtype=float)
        ndim = 9
    elif Config.run_type == '0pa_vs_1pa_dev' and Config.parameter_selected == "extrinsic":
        theta_names = ['m1', 'm2', 'a', 'p0', 'e0','qS', 'phiS', 'Phi_phi0', 'Phi_r0','dev_1','dev_2']
        theta0 = np.array([m1, m2, a, p0, e0,qS, phiS, Phi_phi0, Phi_r0, dev_1, dev_2], dtype=float)
        ndim = 11
    else:         raise ValueError(f"Unsupported run_type {Config.run_type} with parameter_selected {Config.parameter_selected}")


    # Objective setup with tracker for fallback support
    phase_max_flag = (target_func == 'optimal_snr_phase_max')
    raw_objective = objective_factory(
        target_func=target_func,
        ctx=ctx,
        phase_max=phase_max_flag,
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
                denom = np.abs(theta_ref) #+ 1e-30
                rel = np.abs(np.asarray(theta) - theta_ref) / denom
                if np.any(rel > tol):
                    return 1e7
                # Nelder–Mead minimizes; convert larger-is-better score to loss
                score_val = objective(theta)
                return -float(score_val)
            result = nelder_mead_optimize(
                theta0,
                bounded_objective,
                xatol=Config.nm_xatol,
                fatol=Config.nm_fatol,
            )
            best_score = -float(result.fun)
            tracker.update(result.x, best_score)

            # Per-index output directory named with best score and optimized point
            
            _opt_vals = result.x
            
            _vals_str = ','.join(f"{v:.12g}" for v in _opt_vals)
            idx_dir = os.path.join(base_dir, f"{best_score:.12g}_{_vals_str}")
            os.makedirs(idx_dir, exist_ok=True)
            out = {
                'optimizer': 'nelder-mead',
                'target_func': Config.target_func,
                'theta0': theta0.tolist(),
                'x': result.x.tolist(),
                'fun': float(result.fun),
                'best_score': best_score,
                'success': bool(result.success),
                'snr_ref_2pa': float(ctx.get('snr', np.nan)),
            }
            out_name = os.path.join(idx_dir, f"opt_nelder-mead_{Config.target_func}_{timestamp}.json")
            with open(out_name, 'w') as f:
                json.dump(out, f, indent=2)
            print(f"Saved result: {out_name}")
            print(f"[RESULT] Best loss (=-score): {out['fun']:.6e}")
            print(f"[RESULT] Best score: {out['best_score']:.6e}")
            print(f"[RESULT] Best point: {out['x']}")
            
    else:  # PARIS
        try:
            try:
                Q, b, fisher_meta = compute_fisher_parallelotope(
                    ctx=ctx,
                    theta0=theta0,
                    use_gpu=(Config.target_func in ('optimal_snr', 'optimal_snr_phase_max')),
                    prior_sigma_range=float(Config.prior_sigma_range),
                    using_evec=Config.using_evec,)
                fisher_ok = True
            except Exception as e:
                    raise RuntimeError(f"[FATAL] Fisher prior failed: {e}")

            idx_dir = os.path.join(base_dir, f"paris_{Config.target_func}")
            os.makedirs(idx_dir, exist_ok=True)
            savepath = os.path.join(idx_dir, f"paris_results_{Config.target_func}_{timestamp}")
            lhs_seed_rel = 'lhs_seed'
            lhs_seed_dir = os.path.join(idx_dir, lhs_seed_rel)
            sampler, prior_transform, ext_points = run_paris(
                    ndim=ndim,
                    prior_center=theta0,
                    score_func=objective,
                    spread_scale=float(Config.spread_scale),
                    savepath=savepath,
                    seed_cloud=int(Config.seed_cloud),
                    seed_jitter=float(Config.seed_jitter),
                    target_kind=Config.target_func,
                    lhs_save_dir=lhs_seed_dir,
                    affine_Q=Q if fisher_ok else None,
                    affine_b=b if fisher_ok else None,
                    use_ellipse=bool(fisher_meta.get('using_evec', False)),
                )

            out = {
                    'optimizer': 'PARIS',
                    'target_func': Config.target_func,
                    'theta0': theta0.tolist(),
                    'snr_ref_2pa': float(ctx.get('snr', np.nan)),
                    'savepath': savepath,
                    'fisher_prior': True,
                    'fisher_meta': fisher_meta,}

            best_theta = None
            try:
                pts = sampler.searched_points_list
                logs = sampler.searched_log_densities_list
                if not pts or not logs:
                    raise ValueError("Empty PARIS search results")
                best_unit = pts[0][int(np.argmax(logs[0]))]
                best_theta = prior_transform(best_unit)
            except Exception as e:
                print(f"[WARN] PARIS best extraction failed: {e}; using fallback point")
                fallback = getattr(sampler, '_fallback_best_point', None)
                if fallback is None:
                    raise RuntimeError("PARIS failed and no fallback best point available") from e
                best_theta = np.asarray(fallback, dtype=float)

            if best_theta is None:
                print("[WARN] No PARIS best point available; using starting point as placeholder")
                best_theta = theta0

            if np.all((best_theta >= 0.0) & (best_theta <= 1.0)):
                best_theta = prior_transform(np.asarray(best_theta))
            best_theta = np.asarray(best_theta, dtype=float)
            best_val = float(objective(best_theta))
            ctx_polish = dict(ctx)
           
            Qp, bp, _ = compute_fisher_parallelotope(
                ctx=ctx_polish,
                theta0=np.asarray(best_theta),
                use_gpu=(Config.target_func in ('optimal_snr', 'optimal_snr_phase_max')),
                prior_sigma_range=float(Config.prior_sigma_range),
                using_evec=Config.using_evec,
            )
            cov = covariance_from_fisher_parallelotope(Qp, bp, prior_sigma_range=float(Config.prior_sigma_range))
            rng = np.random.default_rng()
            ndim_local = len(best_theta)
            for _it in range(500):
                step = rng.multivariate_normal(mean=np.zeros(ndim_local), cov=1e-5 * cov)
                cand = np.asarray(best_theta) + step
                cand = _clip_physical_params_intrinsic(cand)
                val = float(objective(np.asarray(cand)))
                if val > best_val:
                    best_val = val
                    best_theta = cand
            tracker.update(best_theta, best_val)
            print(f"[POLISH] Final best score: {best_val:.6e}")
            print(f"[POLISH] Final best point: {np.asarray(best_theta).tolist()}")
           
            _opt_vals = np.asarray(best_theta)
            _vals_str = ','.join(f"{float(v):.12g}" for v in _opt_vals)
            new_idx_dir = os.path.join(base_dir, f"idx{idx}_{best_val:.12g}_{_vals_str}")
            try:
                if os.path.abspath(new_idx_dir) != os.path.abspath(idx_dir):
                    os.rename(idx_dir, new_idx_dir)
                    idx_dir = new_idx_dir
                    savepath = os.path.join(idx_dir, os.path.basename(savepath))
            except Exception:
                pass
            lhs_seed_dir = os.path.join(idx_dir, lhs_seed_rel)
            out_enriched = dict(out)
            out_enriched['savepath'] = savepath
            out_enriched.update({
                'best_point': np.asarray(best_theta).tolist(),
                'best_score': float(best_val),
                'lhs_seed_dir': lhs_seed_dir,
            })
            out_name = os.path.join(idx_dir, f"opt_PARIS_{Config.target_func}_{timestamp}.json")
            with open(out_name, 'w') as f:
                json.dump(out_enriched, f, indent=2)
            opt_row = np.array(sig_row, dtype=float)
            if pa_template == '0PA':
                opt_row[:5] = np.asarray(best_theta)[:5]
            else:
                opt_row[:5] = np.asarray(best_theta)[:5]
                if len(best_theta) > 5:
                    opt_row[16] = np.asarray(best_theta)[5]
            opt_array = opt_row.reshape(1, -1)
            out_npy = os.path.join(idx_dir, f"opt_PARIS_{Config.target_func}_{timestamp}.npy")
            np.save(out_npy, opt_array)
            print(f"Saved optimized row to {out_npy} with shape {opt_array.shape}")
            score_npy = os.path.join(idx_dir, f"score_PARIS_{Config.target_func}_{timestamp}.npy")
            np.save(score_npy, np.array([best_val], dtype=float))
            print(f"Saved per-index score to {score_npy}")
 
        except Exception as exc:
            print(f"[WARN] PARIS optimization failed: {exc}")

if __name__ == '__main__':
    main()
