import os
from typing import Optional
import numpy as np
import warnings
import traceback



class Config:

    def __init__(self, **kwargs):
    
        # Target SNR for Fisher scaling
        self._TARGET_SNR = 20.0
        self.param_names_to_infer = ['m1', 'm2', 'a', 'p0', 'e0']

        self.sigma_range = 400.0
        self.params = np.array([1e6,1e4,0.9, 2.85813146e+01,  5.00000000e-01,  1.00000000e+00,
          3.31765439e+01,  1.04719755e+00,  7.85398163e-01, 6.28318531e-01,  5.23598776e-01,  1.00000000e-01,
         2.00000000e-01,  3.00000000e-01])
        
        self.params_name = ["m1","m2","a","p0","e0","xI0","dist","qS","phiS","qK","phiK",
                         "Phi_phi0","Phi_theta0","Phi_r0"]
        
        self.dt = 10  #Time step for waveform generation; default 0.1s
        self.T= 0.25
        self.chi2 = 0.95
        self.dev_1 = 0.0
        self.dev_2 = 0.0

        self.spread_scale = 0.1 #Multiplicative spread for PARIS prior band (e.g., 0.1 => ±10%)
        self.grid_index = 0.0  #Default to 0; can be overridden by $GRID_INDEX env var or --grid-index CLI arg
        self.nm_xatol = 1e9  #tol for Nelder-Mead; set high to disable
        self.using_evec = True  #Use Fisher eigenvectors to define ellipse prior; default builds diagonal box
        self.seed_cloud = 5000  #Number of initial unit-cube seeds for PARIS around center
        self.nm_fatol = 1e-2  #Absolute function tolerance for Nelder-Mead; default 0.01
        self.target_func = 'optimal_snr_phase_max'  #Default target function for optimization
        self.optimizer = 'PARIS'  #Default optimizer
        # self.signal_param_array = np.array([])  #To be loaded from file or defined in code
        self.startingpoints = 'signal_parameter_array_IMRI.npy'  #Default path for starting points; can be overridden by --startingpoints CLI arg

        self.parameter_selected = "intrinsic" #or "extrinsic"
        self.run_type = "0pa_vs_1pa" # or "0pa_vs_1pa_dev"
        self.include_noise = False #Whether to include noise in the likelihood evaluations (default False for testing)

        self.basedir = "/scratch/e1583490/try"
        self.prior_sigma_range = 50.0  #Default range for uniform prior in PARIS (±20% of center)
        

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
        """Print a summary of current configuration."""

        print("CONFIGURATION DETAILS")

        print("\n--- Parameters ---")
        for i, (name, value) in enumerate(zip(self.params_name, self.params)):
            print(f"  [{i:02d}] {name:12s} : {value:.6e}")

        print("\n--- Inference Parameters ---")
        for p in self.param_names_to_infer:
            print(f"  - {p}")

        print("\n--- Computation ---")
        print(f"  dt                : {self.dt}")
        print(f"  T                 : {self.T}")
        print(f"  TARGET_SNR        : {self._TARGET_SNR}")

        print("\n--- Optimizer ---")
        print(f"  optimizer         : {self.optimizer}")
        print(f"  target_func       : {self.target_func}")
        print(f"  nm_xatol          : {self.nm_xatol}")
        print(f"  nm_fatol          : {self.nm_fatol}")

        print("\n--- PARIS Settings ---")
        print(f"  spread_scale      : {self.spread_scale}")
        print(f"  sigma_range       : {self.sigma_range}")
        print(f"  using_evec        : {self.using_evec}")
        print(f"  seed_cloud        : {self.seed_cloud}")

        print("\n--- Run Setup ---")
        print(f"  grid_index        : {self.grid_index}")
        print(f"  startingpoints    : {self.startingpoints}")
        print(f"  parameter_selected: {self.parameter_selected}")
        print(f"  run_type          : {self.run_type}")

        print("\n--- Diagnostics ---")
        print(f"  chi2              : {self.chi2}")
        print(f"  dev_1             : {self.dev_1}")
        print(f"  dev_2             : {self.dev_2}")

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