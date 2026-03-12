import os
from typing import Optional
import numpy as np
import warnings
import traceback


class Config:

    def __init__(self, **kwargs):
    
        # Target SNR for Fisher scaling
        self._TARGET_SNR = 20.0
        self.param_names_to_infer = ['m1', 'm2', 'a', 'p0', 'e0', 'qS', 'phiS',  'Phi_phi0',  'Phi_r0']
        self.sigma_range = 20.0
        self.params = np.array([])
        self.spread_scale = 0.1 #Multiplicative spread for PARIS prior band (e.g., 0.1 => ±10%)
        self.grid_index = 0.0  #Default to 0; can be overridden by $GRID_INDEX env var or --grid-index CLI arg
        self.nm_xatol = 1e9  #tol for Nelder-Mead; set high to disable
        self.using_evec = True  #Use Fisher eigenvectors to define ellipse prior; default builds diagonal box
        self.seed_cloud = 5000  #Number of initial unit-cube seeds for PARIS around center
        self.nm_fatol = 1e-2  #Absolute function tolerance for
        self.target_func = 'optimal_snr_phase_max'  #Default target function for optimization
        self.optimizer = 'PARIS'  #Default optimizer
        self.signal_param_array = np.array([])  #To be loaded from file or defined in code
        self.startingpoints = 'signal_parameter_array_IMRI.npy'  #Default path for starting points; can be overridden by --startingpoints CLI arg
        self.dt = 10  #Time step for waveform generation; default 0.1s
        self.T= 1
        self.chi2 = 0.0
        self.dev_1 = 0.0
        self.dev_2 = 0.0
        self.parameter_selected = "intrinsic" #or "extrinsic"
        self.run_type = "0pa_vs_1pa" # or "0pa_vs_1pa_dev"





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