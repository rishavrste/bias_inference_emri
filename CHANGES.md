# Restoration guide

After a `git pull` (or hard reset) that overwrites the repo, apply the changes below.
All paths are relative to the repo root unless stated otherwise.
`precompute_fisher.py` is a new file not present in the upstream repo.

---

## 1. src/config_paris.py — replace the entire `__init__` body

Find this (the original opening of `__init__`):
```python
    def __init__(self, **kwargs):
    
        # Target SNR for Fisher scaling
        self.param_names_to_infer = ['m1', 'm2', 'a', 'p0', 'e0']  #Default parameters to infer; can be overridden by --params CLI arg
        self.params_name = ["m1","m2","a","p0","e0","xI0","dist","qS","phiS","qK","phiK",
                         "Phi_phi0","Phi_theta0","Phi_r0"]
        self.param_file = "/scratch/e1583490/SuperKludege_Optimizations/opt_grid/signal_parameter_array_IMRI.npy"
        self.result_file = "/scratch/e1583490/SuperKludege_Optimizations/opt_grid/result_parameter_array_IMRI.npy"
        self.TYPE = "IMRI"   #IMRI or IMRI_phase
        self.start_index = 0
        self.end_index =  1


        self.nchannels = 2  #Number of TDI channels to use (default 3 for A, E, T)

        self.spread_scale = 0.4 #Multiplicative spread for PARIS prior band (e.g., 0.1 => ±10%)
        self.grid_index = 0.0  #Default to 0; can be overridden by $GRID_INDEX env var or --grid-index CLI arg
        self.nm_xatol = 1e-6  #tol for Nelder-Mead; set high to disable
        self.using_evec = False  #Use Fisher eigenvectors to define ellipse prior; default builds diagonal box
        self.seed_cloud = 200  #Number of initial unit-cube seeds for PARIS around center
        self.paris_seed_n = 100
        # self.paris_seed_n = 10
        self.paris_niterations = 2000  #Number of PARIS iterations; default 1000

        self.nm_fatol = 1e-6  #Absolute function tolerance for Nelder-Mead; default 0.01
        self.de_maxiter = 1000  #Max iterations for differential evolution; default 1000
        self.nm_maxiter = 10000  #Max iterations for differential evolution; default 1000
        self.target_func = 'optimal_snr'  #'optimal_snr', 'optimal_snr_phase_max', 'time_max', 'phase_match','chi2_match'
        self.optimizer = 'paris'  # nelder-mead or paris or differential_evolution

        self.parameter_selected = "intrinsic" #or "intrinsic_phase","intrinsic"
        self.run_type = "0pa_vs_2pa" # "0pa_vs_2pa", "1pa_vs_2pa"
        self.include_noise = False # Whether to include noise in the likelihood evaluations (default False for testing)

        self.prior_sigma_range = 28.0  #Default range for uniform prior in PARIS (±20% of center)

        self.basedir = "/scratch/e1583490/SuperKludege_Optimizations/IMRI/"

        self.output_text_file = "paris_optimization_results.txt"  #File to save optimization results in text format
        self.seed= 42   

        self.use_gpu = True  #Whether to use GPU acceleration (default False for testing)
```

Replace with:
```python
    def __init__(self, **kwargs):

        # --- Identity / run type (must be defined before any derived values) ---
        self.TYPE = "IMRI"          # IMRI | EMRI | IMRI_TAIL
        self.run_type = "0pa_vs_2pa"  # "0pa_vs_2pa" | "1pa_vs_2pa"
        self.parameter_selected = "intrinsic"  # "intrinsic" | "intrinsic_phase"

        # --- Parameters to infer / differentiate in Fisher ---
        # chi2 (secondary spin) included only for 1PA runs where it is a free parameter.
        if self.run_type == '1pa_vs_2pa':
            self.param_names_to_infer = ['m1', 'm2', 'a', 'p0', 'e0', 'chi2']
        else:
            self.param_names_to_infer = ['m1', 'm2', 'a', 'p0', 'e0']
        self.params_name = ["m1","m2","a","p0","e0","xI0","dist","qS","phiS","qK","phiK",
                         "Phi_phi0","Phi_theta0","Phi_r0"]

        # --- Data paths ---
        self.param_files = {
            'IMRI':      '/scratch/josh.mat/opt_grid/signal_parameter_array_IMRI.npy',
            'EMRI':      '/scratch/josh.mat/opt_grid/signal_parameter_array_EMRI.npy',
            'IMRI_TAIL': '/scratch/josh.mat/opt_grid/signal_parameter_array_IMRI_TAIL.npy',
        }
        self.result_files = {
            'IMRI':      '/scratch/josh.mat/opt_grid/result_parameter_array_IMRI.npy',
            'EMRI':      '/scratch/josh.mat/opt_grid/result_parameter_array_EMRI.npy',
            'IMRI_TAIL': '/scratch/josh.mat/opt_grid/result_parameter_array_IMRI_TAIL.npy',
        }
        self.param_file  = self.param_files[self.TYPE]
        self.result_file = self.result_files[self.TYPE]
        self.basedir = f"/scratch/josh.mat/opt_grid/results/{self.TYPE}/"
        self.fisher_cache_dir = "/scratch/josh.mat/opt_grid/fisher_cache/"

        # --- Grid range ---
        self.start_index = 0
        self.end_index =  1

        # --- Observation / TDI ---
        self.nchannels = 2  # 2: A,E only  3: A,E,T

        # --- Optimizer / sampler settings ---
        self.spread_scale = 0.4
        self.grid_index = 0.0
        self.nm_xatol = 1e-6
        self.using_evec = False
        self.seed_cloud = 200
        self.paris_seed_n = 100
        self.paris_niterations = 2000
        self.nm_fatol = 1e-6
        self.de_maxiter = 1000
        self.nm_maxiter = 10000
        self.target_func = 'optimal_snr'  # 'optimal_snr' | 'optimal_snr_phase_max' | 'time_max' | 'chi2_match'
        self.optimizer = 'paris'  # 'nelder-mead' | 'paris' | 'differential_evolution'
        self.include_noise = False
        self.prior_sigma_range = 28.0

        # --- Misc ---
        self.output_text_file = "paris_optimization_results.txt"
        self.seed = 42
        self.use_gpu = True
```

---

## 2. src/misc.py — `compute_fisher_parallelotope`

### 2a. Signature: add two new keyword arguments

Find:
```python
def compute_fisher_parallelotope(ctx: dict,
                                 fisher_params: list,
                                 params_to_infer: list,
                                 additional_kwargs: dict,
                                 build_waveform_response = None,
                                 use_gpu: bool = True,
                                 _TARGET_SNR: float = None,
                                 prior_sigma_range: float = 20,
                                 using_evec: bool = False) -> Tuple[np.ndarray, np.ndarray, dict]:
```

Replace with:
```python
def compute_fisher_parallelotope(ctx: dict,
                                 fisher_params: list,
                                 params_to_infer: list,
                                 additional_kwargs: dict,
                                 build_waveform_response = None,
                                 use_gpu: bool = True,
                                 _TARGET_SNR: float = None,
                                 prior_sigma_range: float = 20,
                                 using_evec: bool = False,
                                 cache_dir: Optional[str] = None,
                                 cache_index: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
```

### 2b. Body: replace the entire SEF block with cache-aware version

Find (the block starting just after the nchannels/channels setup and ending just before the `try:` that calls `fishinv`):
```python
    sef = StableEMRIFisher(waveform_class=SuperKludgeWaveform,
```
…through to…
```python
    snr_model = float(np.sqrt(inner_product(waveform_tmpl, waveform_tmpl, PSD_funcs, float(ctx['dt']), use_gpu=bool(use_gpu))))
    print(f"[MODEL] SNR in fisher calculation: {snr_model:.6f}")
```
…then the next line is the scale / try block.

Replace that entire region (from after the `raise ValueError` for bad nchannels up to the `print(f"[MODEL] SNR …")` line inclusive) with:
```python
    # --- Fisher cache ---
    # Cache stores the raw Fisher matrix F and template snr_model before any SNR scaling.
    # The scale factor (_TARGET_SNR / snr_model)^2 is applied at runtime after loading.
    F = None
    snr_model = None
    cache_path = None
    if cache_dir is not None and cache_index is not None:
        if additional_kwargs.get('evolve_2PA'):
            pa_tag = '2pa'
        elif additional_kwargs.get('evolve_1PA'):
            pa_tag = '1pa'
        else:
            pa_tag = '0pa'
        params_tag = '_'.join(params_to_infer)
        cache_path = os.path.join(cache_dir, f"fisher_{cache_index:04d}_{pa_tag}_{params_tag}.npy")
        if os.path.exists(cache_path):
            cached = np.load(cache_path, allow_pickle=True).item()
            F = np.asarray(cached['F'], dtype=float)
            snr_model = float(cached['snr_model'])
            print(f"[FISHER] Loaded from cache: {cache_path}")

    if F is None:
        sef = StableEMRIFisher(waveform_class=SuperKludgeWaveform,
                           waveform_class_kwargs = dict(sum_kwargs=dict(pad_output=False, odd_len=True)),
                           waveform_generator = GenerateEMRIWaveform,
                           waveform_generator_kwargs= dict(return_list=False),
                           ResponseWrapper=ResponseWrapper,
                           ResponseWrapper_kwargs = dict(Tobs=ctx['T'],
                                                        t0=10000.0,
                                                        dt=ctx['dt'],
                                                        index_lambda=8,
                                                        index_beta=7,
                                                        flip_hx=True,
                                                        is_ecliptic_latitude=False,
                                                        remove_garbage="zero",
                                                        orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
                                                        force_backend = "cuda12x" if use_gpu else "cpu",
                                                        order=20,
                                                        tdi="2nd generation",
                                                        tdi_chan=tdi_chan),
                           stats_for_nerds = True, use_gpu = use_gpu,
                           deriv_type='stable',
                           noise_model=get_sensitivity,
                           noise_kwargs=noise_kwargs,
                           channels=channels,
                           T = ctx['T'], dt = ctx['dt'],
                           stability_plot = False,
                           der_order = 6, Ndelta = 12,
                           plunge_check=True, return_derivatives=False
                           )
        emri_kwargs = {"T":ctx['T'], "dt":ctx['dt']}

        pars_list_com = list(fisher_params) + [ctx['chi2'],additional_kwargs['evolve_1PA'],additional_kwargs['evolve_primary'],
         additional_kwargs['evolve_2PA']]

        SNR = sef.SNRcalc_SEF(*pars_list_com,**emri_kwargs,use_gpu=use_gpu)
        print("SNR: ", SNR)
        param_dict = {
        'm1': fisher_params[0],
        'm2': fisher_params[1],
        'a': fisher_params[2],
        'p0': fisher_params[3],
        'e0': fisher_params[4],
        'xI0': fisher_params[5],
        'dist': fisher_params[6],
        'qS': fisher_params[7],
        'phiS': fisher_params[8],
        'qK': fisher_params[9],
        'phiK': fisher_params[10],
        'Phi_phi0': fisher_params[11],
        'Phi_theta0': fisher_params[12],
        'Phi_r0': fisher_params[13]}

        Fisher = sef(wave_params = param_dict, param_names=param_names, add_param_args=additional_kwargs,
                live_dangerously = False, stability_plot = False, der_order = 6, Ndelta = 12,
                )

        F = np.asarray(Fisher, dtype=float)

        waveform_tmpl = xp.array(sef.waveform)
        print("shape of waveform_tmpl: ", waveform_tmpl.shape)
        PSD_funcs = generate_PSD(waveform=waveform_tmpl, dt=float(ctx['dt']), noise_PSD=get_sensitivity,
                                 channels=channels, noise_kwargs=noise_kwargs, use_gpu=bool(use_gpu))
        snr_model = float(np.sqrt(inner_product(waveform_tmpl, waveform_tmpl, PSD_funcs, float(ctx['dt']), use_gpu=bool(use_gpu))))

        if cache_path is not None:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_path, {'F': F, 'snr_model': snr_model})
            print(f"[FISHER] Saved to cache: {cache_path}")

    try:
        print(f"[FISHER] {repr(F)}")
        print(f"[FISHER_IS_PD] {_is_pos_def(F)}")
        param_dict_m1 = fisher_params[0]
        F_inv = fishinv(param_dict_m1, F, index_of_M=0)
```

Note: in the original code the `try:` block contains `param_dict['m1']` — change that to `param_dict_m1 = fisher_params[0]` and use `param_dict_m1` in the `fishinv` call, because `param_dict` is now scoped inside the `if F is None:` block.

Also remove the now-unreachable line:
```python
    emri_flags = {"T": ctx['T'], "dt": ctx['dt'], '1PA': False, 'evolve_primary': False, '2PA': False}
```

And replace:
```python
    print(f"[MODEL] SNR in fisher calculation: {snr_model:.6f}")
```
(which no longer exists after the restructure — the SNR print moved inside the `if F is None:` block above).

---

## 3. src/inference.py

### 3a. `main()` signature: add two keyword arguments

Find:
```python
        seed,cfg, use_gpu=True):
```
Replace with:
```python
        seed,cfg, use_gpu=True,
        cache_dir=None, grid_index=None):
```

### 3b. After `add_kwargs = {...}` add `fisher_add_kwargs`

Find:
```python
    add_kwargs = {'chi2': chi2,'evolve_1PA': True,'evolve_primary': False,'evolve_2PA': True}
        
    ctx = prepare_true_waveform(
```
Replace with:
```python
    add_kwargs = {'chi2': chi2,'evolve_1PA': True,'evolve_primary': False,'evolve_2PA': True}
    # Fisher is always computed at 2PA (the true signal), independent of the
    # template PA order used in the optimisation objective.
    fisher_add_kwargs = {'chi2': chi2,'evolve_1PA': True,'evolve_primary': False,'evolve_2PA': True}
        
    ctx = prepare_true_waveform(
```

### 3c. Differential-evolution Fisher call: swap kwargs and add cache args

Find (inside `elif optimizer == 'differential_evolution':`):
```python
                Q, b, fisher_meta = compute_fisher_parallelotope(
                    ctx=ctx,
                    params_to_infer= param_names_to_infer,
                    fisher_params=signal_param_array,
                    use_gpu=USE_GPU,
                    prior_sigma_range=float(prior_sigma_range),
                    using_evec=using_evec,
                    additional_kwargs=add_kwargs,
                    _TARGET_SNR= _TARGET_SNR,
                    build_waveform_response= build_waveform_response

                )
```
Replace with:
```python
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
```

### 3d. PARIS first Fisher call: same swap

Find (inside `elif optimizer == 'paris':`, the first Fisher call):
```python
                Q, b, fisher_meta = compute_fisher_parallelotope(
                    ctx=ctx,
                    params_to_infer=param_names_to_infer,
                    fisher_params=signal_param_array,
                    use_gpu=USE_GPU,
                    prior_sigma_range=float(prior_sigma_range),
                    using_evec=using_evec,
                    additional_kwargs=add_kwargs,
                    _TARGET_SNR= _TARGET_SNR,
                    build_waveform_response= build_waveform_response

                )
```
Replace with:
```python
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
```

### 3e. PARIS polishing call: swap kwargs, move `ndim_local` above

Find:
```python
            Qp, bp, _ = compute_fisher_parallelotope(
                       ctx=ctx,
                    params_to_infer= param_names_to_infer,
                    fisher_params=best_fit_points,
                    use_gpu=USE_GPU,
                    prior_sigma_range=float(prior_sigma_range),
                    using_evec=using_evec,
                    additional_kwargs=add_kwargs,
                        _TARGET_SNR= _TARGET_SNR,
                    build_waveform_response= build_waveform_response
                )

            cov = covariance_from_fisher_parallelotope(
                Qp, bp, prior_sigma_range=float(prior_sigma_range)
            )

            rng = np.random.default_rng()
            ndim_local = len(best_theta)
```
Replace with:
```python
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
```

### 3f. `__main__` loop: add `cache_dir` and `grid_index` to `main()` call

Find:
```python
                           paris_conf=paris_conf,seed=seed,
                           cfg = cfg)
```
Replace with:
```python
                           paris_conf=paris_conf,seed=seed,
                           cfg = cfg,
                           cache_dir=os.path.join(cfg.fisher_cache_dir, cfg.TYPE),
                           grid_index=i)
```

Note: `cfg.TYPE` is `"IMRI"`, `"EMRI"`, or `"IMRI_TAIL"`. This scopes the cache lookup
to a per-type subdirectory so that IMRI index 0 and EMRI index 0 never share a filename.

---

## 4. src/precompute_fisher.py — new file (create from scratch)

This file does not exist in the upstream repo. Create it at `src/precompute_fisher.py` with the following content:

```python
"""
Pre-compute 2PA Fisher matrices for all injected parameter sets and save them
to fisher_cache_dir.  Subsequent inference runs load from cache instead of
recomputing via StableEMRIFisher.

The Fisher is always evaluated at 2PA (evolve_1PA=True, evolve_2PA=True),
matching the injected signal, regardless of which template PA order is used
during optimisation.

Cache file naming:
    fisher_{index:04d}_2pa_{params_tag}.npy
where params_tag = '_'.join(params_to_infer).

Usage:
    python precompute_fisher.py                   # IMRI (Config default)
    python precompute_fisher.py --type EMRI
    python precompute_fisher.py --type IMRI_TAIL
    python precompute_fisher.py --type all        # all three configurations
    python precompute_fisher.py --type IMRI --start 0 --end 5
    python precompute_fisher.py --type all --1pa  # also compute 1PA (adds chi2)
"""

import argparse
import os

import numpy as np

try:
    import cupy as cp
    USE_GPU = True
except ImportError:
    USE_GPU = False
    print("[INFO] CuPy not found; using CPU.")

from config_paris import Config
from misc import compute_fisher_parallelotope
from inference import build_waveform_response


def build_minimal_ctx(T, dt, chi2, nchannels, use_gpu):
    """Minimal context for compute_fisher_parallelotope.

    StableEMRIFisher generates its own waveform internally; only the
    ResponseWrapper and a few scalars are needed from ctx.
    waveform_true_fft is only accessed for its channel count (.shape[0]).
    """
    return {
        'waveform_response': build_waveform_response(T=T, dt=dt, use_gpu=use_gpu),
        'waveform_true_fft': np.zeros((nchannels, 1)),
        'T': T,
        'dt': dt,
        'chi2': chi2,
    }


def compute_one(param_row, index, type_name, cfg, use_gpu, params_to_infer=None):
    T    = param_row[15]
    dt   = param_row[14]
    chi2 = param_row[16]

    if params_to_infer is None:
        params_to_infer = cfg.param_names_to_infer

    ctx = build_minimal_ctx(T, dt, chi2, cfg.nchannels, use_gpu)

    # Always 2PA: Fisher reflects the true injected signal.
    additional_kwargs = {
        'chi2': chi2,
        'evolve_1PA': True,
        'evolve_primary': False,
        'evolve_2PA': True,
    }

    # Each system type gets its own subdirectory so that index 0 of IMRI and
    # index 0 of EMRI do not collide on the same filename.
    cache_dir = os.path.join(cfg.fisher_cache_dir, type_name)

    # _TARGET_SNR is only used for the Q/b output which we discard here;
    # F and snr_model cached by the function are independent of it.
    compute_fisher_parallelotope(
        ctx=ctx,
        fisher_params=param_row[0:14],
        params_to_infer=params_to_infer,
        additional_kwargs=additional_kwargs,
        use_gpu=use_gpu,
        _TARGET_SNR=1.0,
        prior_sigma_range=cfg.prior_sigma_range,
        using_evec=cfg.using_evec,
        cache_dir=cache_dir,
        cache_index=index,
    )


def run_for_type(type_name, cfg, start, end, use_gpu, also_1pa):
    param_file = cfg.param_files[type_name]
    param_array = np.load(param_file)
    n_total = param_array.shape[0]

    end_eff = min(end, n_total) if end is not None else n_total

    # params for each PA variant
    params_0pa = ['m1', 'm2', 'a', 'p0', 'e0']
    params_1pa = ['m1', 'm2', 'a', 'p0', 'e0', 'chi2']

    print(f"\n{'='*60}")
    print(f"[{type_name}] {param_file}  ({n_total} rows)")
    print(f"[{type_name}] Processing indices [{start}, {end_eff})")
    print(f"[{type_name}] 0PA params: {params_0pa}")
    if also_1pa:
        print(f"[{type_name}] 1PA params: {params_1pa}")

    os.makedirs(os.path.join(cfg.fisher_cache_dir, type_name), exist_ok=True)

    for i in range(start, end_eff):
        row = param_array[i]
        print(f"\n  [{i+1}/{end_eff}] index {i}  "
              f"m1={row[0]:.3e}  m2={row[1]:.3e}  a={row[2]:.3f}  "
              f"p0={row[3]:.4f}  e0={row[4]:.3f}  T={row[15]:.3f}yr")
        try:
            compute_one(row, i, type_name, cfg, use_gpu, params_to_infer=params_0pa)
        except Exception as exc:
            print(f"  [ERROR] 0PA index {i}: {exc}")
        if also_1pa:
            try:
                compute_one(row, i, type_name, cfg, use_gpu, params_to_infer=params_1pa)
            except Exception as exc:
                print(f"  [ERROR] 1PA index {i}: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute 2PA Fisher matrices for the parameter grid."
    )
    parser.add_argument(
        '--type', default=None,
        choices=['IMRI', 'EMRI', 'IMRI_TAIL', 'all'],
        help='Grid type to process (default: Config.TYPE)',
    )
    parser.add_argument('--start', type=int, default=0,
                        help='First grid index (default: 0)')
    parser.add_argument('--end',   type=int, default=None,
                        help='Last grid index exclusive (default: all rows)')
    parser.add_argument('--1pa', dest='also_1pa', action='store_true',
                        help='Also compute 1PA Fisher (adds chi2 to params_to_infer)')
    args = parser.parse_args()

    cfg = Config()

    types_to_run = (
        ['IMRI', 'EMRI', 'IMRI_TAIL'] if args.type == 'all'
        else [args.type if args.type is not None else cfg.TYPE]
    )

    print(f"[INFO] Cache dir : {cfg.fisher_cache_dir}")
    print(f"[INFO] GPU       : {USE_GPU}")
    print(f"[INFO] nchannels : {cfg.nchannels}")
    print(f"[INFO] Types     : {types_to_run}")
    print(f"[INFO] also_1pa  : {args.also_1pa}")

    for type_name in types_to_run:
        run_for_type(type_name, cfg, args.start, args.end, USE_GPU, args.also_1pa)

    print(f"\n[DONE] Fisher cache written to {cfg.fisher_cache_dir}")


if __name__ == "__main__":
    main()
```

---

## 5. Scratch data files (not in repo — create once)

```bash
# Parameter grids (copy from colleague or regenerate with grid_array_generation.py)
# Already present at /scratch/josh.mat/opt_grid/

# Zeroed result arrays (one per system type):
python3 -c "
import numpy as np
for name in ['IMRI', 'EMRI', 'IMRI_TAIL']:
    src = f'/scratch/josh.mat/opt_grid/signal_parameter_array_{name}.npy'
    dst = f'/scratch/josh.mat/opt_grid/result_parameter_array_{name}.npy'
    shape = np.load(src).shape
    np.save(dst, np.zeros(shape, dtype=float))
    print(f'Created {dst}  shape={shape}')
"
```

Fisher cache files are regenerated by running:
```bash
cd src
python precompute_fisher.py --type all        # 0PA Fishers
python precompute_fisher.py --type all --1pa  # also 1PA Fishers (with chi2)
```

Cache layout on disk (type-segregated to avoid index collisions between EMRI/IMRI/IMRI_TAIL):
```
fisher_cache/
  IMRI/       fisher_0000_2pa_m1_m2_a_p0_e0.npy
              fisher_0000_2pa_m1_m2_a_p0_e0_chi2.npy
              ...
  EMRI/       fisher_0000_2pa_m1_m2_a_p0_e0.npy
              ...
  IMRI_TAIL/  fisher_0000_2pa_m1_m2_a_p0_e0.npy
              ...
```

---

## 6. ~/few_env/pyvenv.cfg — one-line fix (non-repo, apply once)

The venv was created with `include-system-site-packages = false`, so Python did not
inherit the conda base packages (numpy, FEW, etc.) that are installed there.

Find:
```
include-system-site-packages = false
```
Replace with:
```
include-system-site-packages = true
```

---

## 7. src/precompute_fisher.pbs — new file (create from scratch)

```bash
#!/bin/bash
#PBS -P CFP03-CF-051
#PBS -j oe
#PBS -k oed
#PBS -N precompute_fisher
#PBS -q auto
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -l walltime=36:00:00

cd $PBS_O_WORKDIR

LOG_DIR=/scratch/josh.mat/opt_grid/logs_precompute
mkdir -p "$LOG_DIR"

image="/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif"
module load singularity

singularity exec --nv -e "$image" bash << 'EOF' > "$LOG_DIR/precompute_fisher.out.$PBS_JOBID" 2> "$LOG_DIR/precompute_fisher.err.$PBS_JOBID"
# The few_env activate script hardcodes the /nfs/home path which is not visible
# inside the container (home is mounted at /home/svu/josh.mat here).
# Use miniconda Python directly — it has all packages installed and runs fine.
PYTHON=/home/svu/josh.mat/miniconda3/bin/python3
cd /home/svu/josh.mat/git_repos/bias_inference_emri/src

# Computes 2PA Fishers for all three system types (IMRI, EMRI, IMRI_TAIL).
# --1pa also computes the 6-param variant (adds chi2) alongside the 5-param 0PA version.
# Both are stored with distinct filenames and the inference code loads whichever
# matches the configured run_type (0pa_vs_2pa -> 5-param file, 1pa_vs_2pa -> 6-param file).
$PYTHON precompute_fisher.py --type all --1pa
EOF
```

**Why miniconda Python directly instead of `source few_env/bin/activate`:**
The venv's activate script hardcodes `VIRTUAL_ENV=/nfs/home/svu/josh.mat/few_env`
(the login-node NFS path). Inside the singularity container the home is mounted at
`/home/svu/josh.mat` (no `/nfs` prefix), so the activate script adds a non-existent
path to `PATH` and `which python3` falls through to the container's `/usr/bin/python3`
(Python 3.10, no packages). The miniconda Python at `/home/svu/josh.mat/miniconda3/bin/python3`
runs directly and has all required packages.

---

## 8. src/inference.pbs — new file (create from scratch)

PBS script for running an inference job. Configure `src/config_paris.py` before
submitting (TYPE, run_type, start_index, end_index, optimizer).
Logs land in `/scratch/josh.mat/opt_grid/logs_inference/`.

```bash
#!/bin/bash
#PBS -P CFP03-CF-051
#PBS -j oe
#PBS -k oed
#PBS -N inference_emri
#PBS -q auto
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -l walltime=8:00:00

cd $PBS_O_WORKDIR

LOG_DIR=/scratch/josh.mat/opt_grid/logs_inference
mkdir -p "$LOG_DIR"

image="/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif"
module load singularity

singularity exec --nv -e "$image" bash << 'EOF' > "$LOG_DIR/inference.out.$PBS_JOBID" 2> "$LOG_DIR/inference.err.$PBS_JOBID"
PYTHON=/home/svu/josh.mat/miniconda3/bin/python3
cd /home/svu/josh.mat/git_repos/bias_inference_emri/src
$PYTHON inference.py
EOF
```

Walltime guide (single grid point, GPU, PARIS optimizer):
- IMRI_TAIL (T=0.25yr): ~1hr
- IMRI     (T=1yr):    ~4hr
- EMRI     (T=2.5yr):  ~8hr

---

## Key config fields to set before each run

| Field | Options | Effect |
|---|---|---|
| `TYPE` | `IMRI` \| `EMRI` \| `IMRI_TAIL` | Selects parameter/result files and basedir |
| `run_type` | `0pa_vs_2pa` \| `1pa_vs_2pa` | Template PA order; also sets `param_names_to_infer` |
| `start_index` / `end_index` | integers | Grid points to process |
| `optimizer` | `paris` \| `nelder-mead` \| `differential_evolution` | Optimiser |
| `target_func` | `optimal_snr` \| `chi2_match` \| `time_max` | Objective |

---

## Round 2 tuning (2026-06-05) — improved settings for stuck points

Changes applied to escape local basins in the 0PA EMRI grid.

### src/config_paris.py

1. `paris_seed_n`: 80 → **100**
2. `paris_temperature`: new field, **100.0** — score divided by this in `paris_log_density`; flattens the landscape so PARIS explores more broadly
3. `de_refine_maxiter`: 50 → **100** generations
4. `de_refine_popsize`: 2 → **5** (pop=25 with ndim=5; ~2500 evals per DE run, ~7 min)
5. `nm_refine_maxiter` / `nm_refine_maxfev`: 300 → **1000**

### src/inference.py

1. Added `_PARIS_TEMPERATURE = 1.0` module global (default 1.0 = no change)
2. In `paris_log_density.eval_one`: `return val` → `return val / _PARIS_TEMPERATURE`
3. In `run_paris`: added `_PARIS_TEMPERATURE = float(paris_conf.get('paris_temperature', 1.0))` to the global-setter block
4. In `paris_conf` dict construction: added `paris_conf['paris_temperature'] = cfg.paris_temperature`

### src/rerun_stuck.sh (new file)

Submits individual PBS jobs for each 0PA EMRI point with overlap < 0.97:
```bash
bash src/rerun_stuck.sh
```
