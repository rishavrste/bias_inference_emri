import os
from typing import Optional
import numpy as np
import warnings
import traceback
import json
import time


class Config:

    def __init__(self, **kwargs):
    
        # Target SNR for Fisher scaling
      #  self._TARGET_SNR = 20.0
        self.param_names_to_infer = ['m1', 'm2', 'a', 'p0', 'e0', 'qS', 'phiS', 'Phi_phi0', 'Phi_r0','dev_1','dev_2']  #Default parameters to infer; can be overridden by --params CLI arg


        # self.params = np.array([1.00000000e+06,  1.00000000e+04,  9.00000000e-01,2.85813146e+01,5.00000000e-01,1.00000000e+00,3.31765439e+01,  1.04719755e+00,  7.85398163e-₀1,
        # 6.28318531e-₀1,  5.23598776e-₀1,  1.₀₀₀₀₀₀₀₀e-₀¹,2.₀₀₀₀₀₀₀₀e-₀¹,  3.₀₀₀₀₀₀₀₀e-₀¹])
        self.params = np.array([1e6,1e1,0.9, 7.5,  5.00000000e-01,  1.00000000e+00,
          5.0,  np.pi/4,  1.0, 1,  np.pi/3,  9.00000000e-01,
         5.00000000e-01,  4.00000000e-01])

        # self.params = np.array([1e6,5e3,0.70, 25.0,  0.25,  1.00000000e+00,
        #   3,  np.pi/4,  1.0, 1,  np.pi/3,  9.00000000e-01,
        #   5.00000000e-01,  4.00000000e-01])
        
        # [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0,\
        #      chi2,evolve_1PA,evolve_primary,evolve_2PA,deviation_included,dev_0_p,dev_0_e,dev_1_p,dev_1_e,dev_2_p,dev_2_e]
        
        self.params_name = ["m1","m2","a","p0","e0","xI0","dist","qS","phiS","qK","phiK",
                         "Phi_phi0","Phi_theta0","Phi_r0"]
        
        self.dt = 10  #Time step for waveform generation; default 0.1s
        self.T= 1.0

        self.chi2 = 0.  #it was 0.95 
        
        self.dev_1 = 0.0
        self.dev_2 = 0.0

        self.spread_scale = 0.1 #Multiplicative spread for PARIS prior band (e.g., 0.1 => ±10%)
        self.grid_index = 0.0  #Default to 0; can be overridden by $GRID_INDEX env var or --grid-index CLI arg
        self.nm_xatol = 1e-2  #tol for Nelder-Mead; set high to disable
        self.using_evec = False  #Use Fisher eigenvectors to define ellipse prior; default builds diagonal box
        self.seed_cloud = 200  #Number of initial unit-cube seeds for PARIS around center
        self.paris_seed_n = 99
        self.nm_fatol = 1e-3  #Absolute function tolerance for Nelder-Mead; default 0.01
        self.de_maxiter = 1200  #Max iterations for differential evolution; default 1000
        self.nm_maxiter = 15000  #Max iterations for differential evolution; default 1000
        self.target_func = 'optimal_snr'  #'optimal_snr', 'optimal_snr_phase_max', 'time_max', 'phase_match','chi2_match'
        self.optimizer = 'differential_evolution'  # nelder-mead or paris or differential_evolution
        # self.signal_param_array = np.array([])  #To be loaded from file or defined in code
        self.startingpoints = "/scratch/e1583490/emri_with_noise_dev_a_1_batch_sigma_75/differential_evolution_optimal_snr_run_id_5/starting_point_6.npy"  #Default path for starting points; can be overridden by --startingpoints CLI arg
        # self.analytic_model = '1PA'  #Analytic model to use for waveform generation; default SHOULD BE NONE
        self.analytic_model = None  #Analytic model to use for waveform generation; default SHOULD BE NONE

        self.parameter_selected = "extrinsic" #or "extrinsic"
        self.run_type = "0pa_vs_1pa_dev" # or "0pa_vs_1pa_dev"
        self.include_noise = True #Whether to include noise in the likelihood evaluations (default False for testing)

        self.prior_sigma_range = 4000  #Default range for uniform prior in PARIS (±20% of center)

        self.basedir = "/scratch/e1583490/emri_with_noise_dev_a_1_batch_sigma_75/"  #Base directory for saving results; can be overridden by --basedir CLI arg
        self.output_text_file = "paris_optimization_results_6.txt"  #File to save optimization results in text format
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