import os
from typing import Optional
import numpy as np
import warnings
import traceback
import json
import time

#run all imri phase with nealder-mead
class Config:

    def __init__(self, **kwargs):
    
        # Target SNR for Fisher scaling
        self.param_names_to_infer = ['m1', 'm2', 'a', 'p0', 'e0',"Phi_phi0","Phi_r0",'chi2'] 
        self.params_name = ["m1","m2","a","p0","e0","xI0","dist","qS","phiS","qK","phiK",
                         "Phi_phi0","Phi_theta0","Phi_r0"]
        self.param_file = "/scratch/e1583490/SuperKludege_Optimizations/opt_grid/signal_parameter_array_IMRI.npy"
        self.result_file = "/scratch/e1583490/SuperKludege_Optimizations/opt_grid/result_parameter_array_IMRI_with_phase_1PA.npy"
        # self.param_file = "/scratch/e1583490/SuperKludege_Optimizations/opt_grid/signal_parameter_array_IMRI_TAIL.npy"
        # self.result_file = "/scratch/e1583490/SuperKludege_Optimizations/opt_grid/result_parameter_array_IMRI_TAIL_with_phase_1PA.npy"
        self.TYPE = "IMRI_phase"   #IMRI or IMRI_phase
        self.start_index = 0 # have to redo 1 afterwards for IMRI case, but can do 2-25 first; set to 0 to run all
        self.end_index =  25


        self.nchannels = 2  #Number of TDI channels to use (default 3 for A, E, T)

        self.spread_scale = 0.4 #Multiplicative spread for PARIS prior band (e.g., 0.1 => ±10%)
        self.grid_index = 0.0  #Default to 0; can be overridden by $GRID_INDEX env var or --grid-index CLI arg
        self.nm_xatol = 1e-6  #tol for Nelder-Mead; set high to disable
        self.using_evec = False  #Use Fisher eigenvectors to define ellipse prior; default builds diagonal box
        self.seed_cloud = 200  #Number of initial unit-cube seeds for PARIS around center
        self.paris_seed_n = 15
        # self.paris_seed_n = 10
        self.paris_niterations = 2000  #Number of PARIS iterations; default 1000

        self.nm_fatol = 1e-6  #Absolute function tolerance for Nelder-Mead; default 0.01
        self.de_maxiter = 250  #Max iterations for differential evolution; default 1000
        self.nm_maxiter = 10000  #Max iterations for differential evolution; default 1000
        self.target_func = 'optimal_snr'  #'optimal_snr', 'optimal_snr_phase_max', 'time_max', 'phase_match','chi2_match'
        self.optimizer = 'nelder-mead'  # nelder-mead or paris or differential_evolution

        self.parameter_selected = "intrinsic_phase" #or "intrinsic_phase","intrinsic"
        self.run_type = "1pa_vs_2pa" # "0pa_vs_2pa", "1pa_vs_2pa"
        self.include_noise = False # Whether to include noise in the likelihood evaluations (default False for testing)

        self.prior_sigma_range = 75.0  #Default range for uniform prior in PARIS (±20% of center)

        # self.basedir = "/scratch/e1583490/SuperKludege_Optimizations/IMRI_TAIL_1PA_with_Phase_2/"
        self.basedir = "/scratch/e1583490/SuperKludege_Optimizations/IMRI_with_phase_1PA/"

        self.output_text_file = "paris_optimization_results.txt"  #File to save optimization results in text format
        self.seed= 42   

        self.use_gpu = True  #Whether to use GPU acceleration (default False for testing)
    
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