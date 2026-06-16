import os
from typing import Optional
import numpy as np
import json
import time


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_list(name: str, default: list) -> list:
    """Comma-separated env var override for a list of paths; falls back to `default`."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return [p.strip() for p in raw.split(",") if p.strip()]


class Config:
    """Run configuration for the 1PA-vs-2PA / 0PA-vs-2PA overlap-optimization pipeline.

    All data paths default to this project's own campaign layout but can be overridden
    via environment variables (see each field below) so the same code can be pointed at
    a different signal/result grid or phase-output tree without editing this file.
    """

    def __init__(self, **kwargs):

        self.run_type = "1pa_vs_2pa"  # "0pa_vs_2pa", "1pa_vs_2pa"
        self.parameter_selected = "intrinsic_phase"  # "intrinsic" | "intrinsic_phase"

        # Fisher is computed for intrinsic params only; phase dims (Phi_phi0, Phi_r0) receive a
        # uniform ±pi prior via padding in inference.py.  Do NOT list phases here.
        if self.run_type == '1pa_vs_2pa':
            self.param_names_to_infer = ['m1', 'm2', 'a', 'p0', 'e0', 'chi2']
        else:
            self.param_names_to_infer = ['m1', 'm2', 'a', 'p0', 'e0']

        self.params_name = ["m1","m2","a","p0","e0","xI0","dist","qS","phiS","qK","phiK",
                         "Phi_phi0","Phi_theta0","Phi_r0"]

        # Signal grid (truth params) and the array where best-fit results get written.
        # Override with $OPT_PARAM_FILE / $OPT_RESULT_FILE to point at a different grid.
        self.param_file = _env_str(
            "OPT_PARAM_FILE",
            "/scratch/e1583490/SuperKludege_Optimizations/opt_grid/signal_parameter_array_IMRI_TAIL.npy")
        self.result_file = _env_str(
            "OPT_RESULT_FILE",
            "/scratch/e1583490/SuperKludege_Optimizations/opt_grid/result_parameter_array_IMRI_TAIL_with_phase_1PA.npy")
        self.TYPE = _env_str("OPT_CASE_PREFIX", "IMRI_TAIL_1PA")
        self.start_index = int(os.environ.get('START_INDEX', 0))
        self.end_index = int(os.environ.get('END_INDEX', 13))

        self.nchannels = 2  #Number of TDI channels to use (default 3 for A, E, T)

        self.spread_scale = 0.4 #Multiplicative spread for PARIS prior band (e.g., 0.1 => ±10%)
        self.grid_index = 0.0  #Default to 0; can be overridden by $GRID_INDEX env var or --grid-index CLI arg
        self.nm_xatol = 1e-12  #tol for Nelder-Mead
        self.using_evec = False  #Use Fisher eigenvectors to define ellipse prior; default builds diagonal box
        self.seed_cloud = 200  #Number of initial unit-cube seeds for PARIS around center
        self.paris_seed_n = 50
        self.paris_niterations = 3000  #Number of PARIS iterations

        self.nm_fatol = 1e-12  #Absolute function tolerance for Nelder-Mead
        self.de_maxiter = 1500           # DE iterations
        self.de_maxiter_cold = 1500
        self.nm_pre_de_maxiter = 0       # no NM pre-polish — DE only
        self.nm_pre_de_threshold = 0.98  # threshold to skip DE after NM
        self.nm_maxiter = 10000  #Max iterations for Nelder-Mead
        self.target_func = 'optimal_snr'  #'optimal_snr', 'optimal_snr_phase_max', 'time_max', 'chi2_match'
        self.optimizer = 'differential_evolution'  # nelder-mead or paris or differential_evolution

        self.include_noise = False  # Whether to include noise in the likelihood evaluations

        # Fisher sigma * prior_sigma_range = PARIS/DE half-width for intrinsic dims.
        # Phase dims always use ±pi (via padding), so this only affects intrinsic params.
        self.prior_sigma_range = 5.0             # 5σ warm start from best fit
        self.prior_sigma_range_cold = 5.0

        self.e0_prior_sigma_factor = 1.0
        self.m2_prior_sigma_factor = 1.0

        # Post-PARIS refinement: DE (global) then NM (local polish) from the PARIS best point.
        self.refine_after_paris = True
        self.de_refine_maxiter = 600    # generations (small pop => ~2*ndim*150 evals)
        self.de_refine_popsize = 2      # small population -- already have good PARIS start
        self.nm_refine_maxiter = 5000
        self.nm_refine_maxfev = 10000
        self.nm_refine_fatol = 1e-10    # avoids burning budget at float64 noise floor
        self.refine_prior_sigma_range = 5.0  # tighter bounds for DE/NM refinement

        # Where this run's per-case output directories get created, and (optionally) a list
        # of earlier-run output trees to check for a warm-startable / already-good previous
        # result. Override with $OPT_BASEDIR and $OPT_PREV_BASEDIRS (comma-separated).
        self.basedir = _env_str(
            "OPT_BASEDIR",
            "/scratch/e1583490/SuperKludege_Optimizations/IMRI_TAIL_1PA_with_Phase_15/")
        self.prev_basedir = _env_list("OPT_PREV_BASEDIRS", [
            "/scratch/e1583490/SuperKludege_Optimizations/IMRI_TAIL_1PA_with_Phase_14/",
            "/scratch/e1583490/SuperKludege_Optimizations/IMRI_TAIL_1PA_with_Phase_13/",
            "/scratch/e1583490/SuperKludege_Optimizations/IMRI_TAIL_1PA_with_Phase_12/",
            "/scratch/e1583490/SuperKludege_Optimizations/IMRI_TAIL_1PA_with_Phase_11/",
            "/scratch/e1583490/SuperKludege_Optimizations/IMRI_TAIL_1PA_with_Phase_8/",
            "/scratch/e1583490/SuperKludege_Optimizations/IMRI_TAIL_1PA_with_Phase_6/",
            "/scratch/e1583490/SuperKludege_Optimizations/IMRI_TAIL_1PA_with_Phase_3/",
        ])
        self.good_overlap_threshold = 0.99
        # warm start from best fit for all cases (threshold=0 → always warm)
        self.warm_start_threshold = 0.0

        self.output_text_file = "paris_optimization_results.txt"  #File to save optimization results in text format
        self.seed = 42

        self.use_gpu = True  #Whether to use GPU acceleration

    def check_initialization(self):
        """Validate option combinations that are mutually exclusive."""
        if "qS" in self.param_names_to_infer or "phiS" in self.param_names_to_infer:
            if self.target_func == 'phase_match':
                raise ValueError(
                    "target_func cannot be 'phase_match' when 'qS' or 'phiS' are being inferred."
                )

    def to_dict(self):
        """Convert config to a serializable dictionary."""
        return {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in self.__dict__.items()
            if not k.startswith("_")  # optional: skip private vars
        }

    def save_results_with_config(cfg, results: dict, save_dir: str, filename_prefix: str):
        """
        Save results + config to:
        1. JSON (structured)
        2. Text file (human readable)
        """

        os.makedirs(save_dir, exist_ok=True)

        timestamp = time.strftime('%Y%m%d-%H%M%S')

        # -------- JSON (structured) --------
        full_output = {
            "timestamp": timestamp,
            "config": cfg.to_dict(),
            "results": results,
        }

        json_path = os.path.join(save_dir, f"{filename_prefix}_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(full_output, f, indent=2)

        # -------- TEXT (human readable) --------
        text_path = os.path.join(save_dir, cfg.output_text_file)

        with open(text_path, "a") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"RUN TIMESTAMP: {timestamp}\n")

            # ---- CONFIG ----
            f.write("\n--- CONFIG ---\n")
            for k, v in cfg.to_dict().items():
                f.write(f"{k}: {v}\n")

            # ---- RESULTS ----
            f.write("\n--- RESULTS ---\n")
            for k, v in results.items():
                f.write(f"{k}: {v}\n")

            f.write("=" * 80 + "\n")

        print(f"[SAVE] JSON: {json_path}")
        print(f"[SAVE] TEXT: {text_path}")


class ObjectiveTracker:
    """Track the most recent objective evaluation for fallback saves."""

    def __init__(self, theta: np.ndarray, score: Optional[float] = None):
        self.theta = np.asarray(theta, dtype=float).copy()
        self.score = None if score is None else float(score)

    def update(self, theta: np.ndarray, score: float) -> None:
        self.theta = np.asarray(theta, dtype=float).copy()
        self.score = float(score)

    def set_theta(self, theta: np.ndarray) -> None:
        self.theta = np.asarray(theta, dtype=float).copy()
