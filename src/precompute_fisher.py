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

    os.makedirs(cfg.fisher_cache_dir, exist_ok=True)

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
