import os
from typing import Optional
import numpy as np
import warnings
import traceback
import json
import time


class Config:

    def __init__(self, **kwargs):

        # --- Identity / run type (must be defined before any derived values) ---
        self.TYPE = "EMRI"          # IMRI | EMRI | IMRI_TAIL
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

        # --- Data paths (repo-relative; works for any user who clones) ---
        _src = os.path.dirname(os.path.abspath(__file__))
        _repo = os.path.dirname(_src)
        _data = os.path.join(_repo, 'data')

        self.param_files = {
            'IMRI':      os.path.join(_data, 'signal_parameter_array_IMRI.npy'),
            'EMRI':      os.path.join(_data, 'signal_parameter_array_EMRI.npy'),
            'IMRI_TAIL': os.path.join(_data, 'signal_parameter_array_IMRI_TAIL.npy'),
        }
        # Result arrays are run outputs — not in repo. inference.py creates them
        # automatically if absent. Override self.result_file to a scratch path if
        # you want outputs to land somewhere else.
        self.result_files = {
            'IMRI':      os.path.join(_data, 'result_parameter_array_IMRI.npy'),
            'EMRI':      os.path.join(_data, 'result_parameter_array_EMRI.npy'),
            'IMRI_TAIL': os.path.join(_data, 'result_parameter_array_IMRI_TAIL.npy'),
        }
        _pa = '0pa' if self.run_type == '0pa_vs_2pa' else '1pa'
        self.param_file  = self.param_files[self.TYPE]
        self.result_file = self.result_files[self.TYPE].replace('.npy', f'_{_pa}.npy')
        self.basedir = f"/scratch/josh.mat/opt_grid/results/{self.TYPE}_{_pa}/"
        self.fisher_cache_dir = os.path.join(_data, 'fisher_cache')

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
        self.paris_niterations = 1000
        self.paris_temperature = 1.0    # divide score by this to flatten landscape; >1 is more exploratory
        self.nm_fatol = 1e-6
        # Stage-1 DE: ~0.144s/eval serial; popsize=15 → 75 members/gen
        # 600 gen × 75 × 0.144s ≈ 108 min; total run ~2h with setup+refine
        self.de_maxiter = 600
        self.de_popsize = 15
        self.de_workers = 1           # 1 = single-threaded (FEW CUDA not thread-safe)
        self.nm_maxiter = 10000
        self.target_func = 'optimal_snr_phase_max'  # 'optimal_snr' | 'optimal_snr_phase_max' | 'time_max' | 'chi2_match'
        self.optimizer = 'paris'  # 'nelder-mead' | 'paris' | 'differential_evolution'
        self.include_noise = False
        self.prior_sigma_range = 30.0
        self.min_prior_widths = {
            'm1':  50000.0,   # 5% of typical m1=1e6
            'm2':    300.0,   # covers worst-case 0PA m2 bias (~300 at high retrograde spin)
            'a':       0.10,  # covers observed a bias up to ~0.065 across the grid
            'p0':      1.0,   # ~3% of typical p0~30
            'e0':      0.05,  # critical: fixes Fisher over-tightness at high eccentricity
            'chi2':    0.1,
        }

        # --- Post-DE refinement ---
        # Narrow DE then Nelder-Mead with phases from best DE point.
        self.refine_after_paris = True
        # DE refine: ~0.1s/eval (maximize_phase=False); 100 gen × popsize × ndim evals
        self.de_refine_maxiter = 150   # generations
        self.de_refine_popsize = 8
        # NM refinement
        self.nm_refine_maxiter = 5000
        self.nm_refine_maxfev  = 10000
        self.overlap_warn_threshold = 0.9  # warn if final overlap < this
        self.refine_prior_sigma_range = 30.0  # tighter bounds for DE/NM (vs prior_sigma_range for PARIS)

        # --- Misc ---
        self.output_text_file = "paris_optimization_results.txt"
        self.seed = 42
        self.use_gpu = True
    
    def check_initialization(self):
    # Check if extrinsic sky parameters are included
        if "qS" in self.param_names_to_infer or "phiS" in self.param_names_to_infer:
            if self.target_func == 'phase_match':
                raise ValueError(
                    "target_func cannot be 'phase_match' when 'qS' or 'phiS' are being inferred."
                )
        

    def get_default_config(**kwargs):
        """
        Get default configuration with optional overrides.
        Parameters
        ----------
        **kwargs : dict
            Configuration parameters to override
        Returns
        -------
        Config
            Configuration object

        Examples
        --------
        >>> cfg = get_default_config()
        >>> cfg = get_default_config(use_gpu=True, n_walkers=100)
        """
        return Config(**kwargs)

    def print_summary(self):
        """Print a detailed summary of current configuration."""

        print("=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)

        # ------------------ PARAMETERS ------------------
        print("\n--- All Parameters ---")
        if len(self.params_name) != len(self.params):
            print("WARNING: params_name and params length mismatch!")

        for i, (name, value) in enumerate(zip(self.params_name, self.params)):
            tag = " (inferred)" if name in self.param_names_to_infer else ""
            print(f"[{i:02d}] {name:12s} : {value:.6e}{tag}")

        # ------------------ INFERENCE ------------------
        print("\n--- Inference Parameters ---")
        print(f"Parameters to infer ({len(self.param_names_to_infer)}):")
        for p in self.param_names_to_infer:
            print(f"  - {p}")

        # ------------------ COMPUTATION ------------------
        print("\n--- Computation Settings ---")
        print(f"dt                : {self.dt}")
        print(f"T                 : {self.T}")
        print(f"TARGET_SNR        : {self._TARGET_SNR}")
        print(f"include_noise     : {self.include_noise}")

        # ------------------ OPTIMIZER ------------------
        print("\n--- Optimizer Settings ---")
        print(f"optimizer         : {self.optimizer}")
        print(f"target_func       : {self.target_func}")
        print(f"nm_xatol          : {self.nm_xatol}")
        print(f"nm_fatol          : {self.nm_fatol}")

        # ------------------ PARIS ------------------
        print("\n--- PARIS Settings ---")
        print(f"spread_scale      : {self.spread_scale}")
        print(f"prior_sigma_range : {self.prior_sigma_range}")
        print(f"using_evec        : {self.using_evec}")
        print(f"seed_cloud        : {self.seed_cloud}")

        # ------------------ RUN SETUP ------------------
        print("\n--- Run Setup ---")
        print(f"grid_index        : {self.grid_index}")
        print(f"startingpoints    : {self.startingpoints}")
        print(f"parameter_selected: {self.parameter_selected}")
        print(f"run_type          : {self.run_type}")
        print(f"basedir           : {self.basedir}")

        # ------------------ DIAGNOSTICS ------------------
        print("\n--- Diagnostics ---")
        print(f"chi2              : {self.chi2}")
        print(f"dev_1             : {self.dev_1}")
        print(f"dev_2             : {self.dev_2}")

        print("\n" + "=" * 60)

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

    if __name__ == "__main__":
        cfg = get_default_config()
        cfg.print_summary()


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