"""
Waveform analysis utilities and configuration
Contains:
- Global config for scenario and template PA
- Helpers to save/load signal parameter arrays
- Functions for SNR, overlap, and optimal SNR computation
"""

import numpy as np
import cupy as cp
import os
#SEF imports
from stableemrifisher.utils import generate_PSD, padding, inner_product

########################################
# Global configuration (user-editable) #
########################################

# Scenario of the signal to generate: "EMRI" or "IMRI"
scenario = "IMRI"

# Lower-PA template to search with: "1PA" or "0PA"
pa_template = "0PA"


############################
# Filepath helper routines #
############################

def get_repo_root():
    """Return repository root assumed as the directory of this utils.py."""
    return os.path.dirname(os.path.abspath(__file__))


def get_signal_param_array_filename():
    """Filename for the signal parameter array based on scenario."""
    return f"signal_parameter_array_{scenario}.npy"


def get_signal_param_array_path():
    """Absolute path for the signal parameter array file."""
    return os.path.join(get_repo_root(), get_signal_param_array_filename())


def load_signal_param_array():
    """Load the signal parameter array saved for the current scenario."""
    path = get_signal_param_array_path()
    if not os.path.exists(path):
        # Also allow loading from current working directory if present
        alt = get_signal_param_array_filename()
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(
                f"Signal parameter array not found: {path} (or {alt}). "
                f"Please run grid_array_generation.py to generate it."
            )
    return np.load(path)


def get_startingpoint_param_array_filename():
    """Default filename for starting-point parameter array based on scenario."""
    return f"startingpoint_param_array_{scenario}.npy"


def get_startingpoint_param_array_path():
    """Absolute path for the starting-point parameter array file."""
    return os.path.join(get_repo_root(), get_startingpoint_param_array_filename())


from typing import Optional


def load_startingpoint_param_array(filename: Optional[str] = None, allow_missing: bool = True):
    """
    Load the starting-point parameter array.

    - If `filename` is provided, load from that path if it exists.
    - Otherwise try repo-root path, then CWD fallback.
    - If not found and `allow_missing` is True, return None; else raise.
    """
    # Explicit path override
    if filename is not None:
        if os.path.exists(filename):
            return np.load(filename)
        if allow_missing:
            return None
        raise FileNotFoundError(f"Starting-point parameter array not found at explicit path: {filename}")

    # Default resolution based on scenario
    path = get_startingpoint_param_array_path()
    if not os.path.exists(path):
        alt = get_startingpoint_param_array_filename()
        if os.path.exists(alt):
            path = alt
        else:
            if allow_missing:
                return None
            raise FileNotFoundError(f"Starting-point parameter array not found: {path} (or {alt})")
    return np.load(path)


# Import backend modules (should be set by main script). These need to be
# imported from the main notebook/script: xp, fft_module, use_gpu, wave, dt,
# T, N_fiducial, fiducial_data



def inner_product_combine(a, b, PSD, dt, window=None, fmin=None, fmax=None, use_gpu=False):
    """
    Compute the frequency domain inner product of two time-domain arrays.
    This function computes the frequency domain inner product of two time-domain arrays using the GPU for acceleration.
    It operates under the assumption that the signals are evenly spaced and applies a Tukey window to each signal.
    This function is optimized for GPUs.
    
    Args:
        a (np.ndarray): The first time-domain signal. It should have dimensions (N_channels, N), where N is the length of the signal.
        b (np.ndarray): The second time-domain signal. It should have dimensions (N_channels, N), where N is the length of the signal.
        PSD (np.ndarray): The power spectral density (PSD) of the signals. It should be a 1D array of length N_channels.
        dt (float): The sampling interval, i.e., the spacing between time samples.
        window (np.ndarray, optional): A window array to envelope the waveform time series. Default is None (no window).
        fmin (float, optional): Minimum frequency for inner_product sum. Default is None.
        fmax (float, optional): Maximum frequency for inner_product sum. Default is None.
        use_gpu (bool, optional): Whether to use GPU. Default is False.
    
    Returns:
        float: The frequency-domain inner product of the two signals.
    """
    # Compute FFT for both signals
    a_fft = compute_fft_with_windowing(a, dt, window, use_gpu)
    b_fft = compute_fft_with_windowing(b, dt, window, use_gpu)
    
    # Compute inner product from FFT results
    return inner_product_from_fft(a_fft, b_fft, PSD, dt, fmin, fmax, use_gpu)


def compute_fft_with_windowing(signal, dt, window=None, use_gpu=False):
    """
    Compute FFT of time-domain signal with optional windowing.
    
    Args:
        signal (np.ndarray): Time-domain signal with dimensions (N_channels, N)
        dt (float): Sampling interval
        window (np.ndarray, optional): Window array to envelope the waveform. Default is None
        use_gpu (bool, optional): Whether to use GPU. Default is False
    
    Returns:
        tuple: (fft_result, original_length) where fft_result is FFT excluding DC component
               and original_length is the length of the time-domain signal
    """
    if use_gpu:
        xp = cp
    else:
        xp = np
    
    signal = xp.atleast_2d(signal)
    original_length = signal.shape[1]  # Store original signal length
    
    # Apply window if provided
    if window is not None:
        window = xp.atleast_2d(xp.asarray(window))
        signal_windowed = signal * window
    else:
        signal_windowed = signal
    
    # Compute FFT and exclude DC component (first element)
    if xp.iscomplexobj(signal_windowed):
        fft_plus = dt * xp.fft.rfft(signal_windowed.real, axis=-1)[:, 1:]
        fft_cross = dt * xp.fft.rfft(signal_windowed.imag, axis=-1)[:, 1:]
        return (fft_plus, fft_cross), original_length
    else:
        fft_result = dt * xp.fft.rfft(signal_windowed, axis=-1)[:, 1:]
        return fft_result#, original_length


def inner_product_from_fft(a_fft_data, b_fft_data, PSD, dt, original_length, fmin=None, fmax=None, use_gpu=False, debug=False, maximize_phase=False):
    """
    Compute frequency domain inner product from pre-computed FFT arrays.
    
    Args:
        a_fft_data: FFT of first signal (or tuple of (fft_plus, fft_cross) for complex signals)
        b_fft_data: FFT of second signal (or tuple of (fft_plus, fft_cross) for complex signals)
        PSD (np.ndarray): Power spectral density, 1D array of length N_channels
        dt (float): Sampling interval
        original_length (int): Length of the original time-domain signal
        fmin (float, optional): Minimum frequency for inner product sum. Default is None
        fmax (float, optional): Maximum frequency for inner product sum. Default is None
        use_gpu (bool, optional): Whether to use GPU. Default is False
        debug (bool, optional): Whether to print debug information. Default is False
    
    Returns:
        float: Frequency-domain inner product
    """
    if use_gpu:
        xp = cp
    else:
        xp = np
    
    # Handle complex signals (tuple input)
    if isinstance(a_fft_data, tuple):
        a_fft_plus, a_fft_cross = a_fft_data
        b_fft_plus, b_fft_cross = b_fft_data
        is_complex = True
        if debug:
            print("DEBUG [inner_product_from_fft]: Complex signal detected")
    else:
        a_fft = a_fft_data
        b_fft = b_fft_data
        is_complex = False
        if debug:
            print("DEBUG [inner_product_from_fft]: Real signal detected")
    
    if debug:
        print(f"DEBUG [inner_product_from_fft]: Original signal length: {original_length}")
        print(f"DEBUG [inner_product_from_fft]: dt: {dt}")
    
    # Create frequency mask using original signal length
    freq = xp.fft.rfftfreq(original_length) / dt
    if use_gpu:
        freq = freq.get()  # Convert to numpy for comparison
    
    if fmin is not None or fmax is not None:
        if fmin is not None:
            mask_min = (freq[1:] > fmin)  # Exclude DC component
        if fmax is not None:
            mask_max = (freq[1:] < fmax)
        
        if fmin is not None and fmax is None:
            freq_mask = mask_min
        elif fmin is None and fmax is not None:
            freq_mask = mask_max
        else:
            freq_mask = xp.logical_and(mask_min, mask_max)
    else:
        freq_mask = xp.full(len(freq) - 1, True, dtype=bool)  # -1 to exclude DC
    
    # Convert back to GPU array if needed
    if use_gpu and not isinstance(freq_mask, type(a_fft_plus if is_complex else a_fft)):
        freq_mask = xp.asarray(freq_mask)
    
    if debug:
        print(f"DEBUG [inner_product_from_fft]: Frequency mask sum: {freq_mask.sum()}")
        print(f"DEBUG [inner_product_from_fft]: Total freq points: {len(freq_mask)}")
    
    PSD = xp.atleast_2d(xp.asarray(PSD))
    df = (original_length * dt) ** -1  # Use original signal length for df calculation
    
    if debug:
        print(f"DEBUG [inner_product_from_fft]: df: {df}")
        print(f"DEBUG [inner_product_from_fft]: PSD shape: {np.array(PSD).shape}")
    
    # Apply frequency mask and compute inner product
    if is_complex:
        a_fft_plus_masked = a_fft_plus[:, freq_mask]
        a_fft_cross_masked = a_fft_cross[:, freq_mask]
        b_fft_plus_masked = b_fft_plus[:, freq_mask]
        b_fft_cross_masked = b_fft_cross[:, freq_mask]

        '''
        inner_prod = 4 * df * ((a_fft_plus_masked.conj() * b_fft_plus_masked + 
                               a_fft_cross_masked * b_fft_cross_masked.conj()).real / 
                              PSD[:, freq_mask]).sum()
        if debug:
            print(f"DEBUG [inner_product_from_fft]: Complex inner product before GPU conversion: {inner_prod}")
        '''
        complex_inner_prod = 4 * df * ((a_fft_plus_masked.conj() * b_fft_plus_masked + 
                                       a_fft_cross_masked * b_fft_cross_masked.conj()) / 
                                      PSD[:, freq_mask]).sum()
        
        if maximize_phase:
            # Phase maximization: take magnitude of complex inner product
            inner_prod = xp.abs(complex_inner_prod)
            if debug:
                print(f"DEBUG [inner_product_from_fft]: Complex inner product magnitude: {inner_prod}")
                print(f"DEBUG [inner_product_from_fft]: Original complex value: {complex_inner_prod}")
        else:
            # Traditional approach: take real part
            inner_prod = complex_inner_prod.real
            if debug:
                print(f"DEBUG [inner_product_from_fft]: Complex inner product real part: {inner_prod}")        
    else:
        a_fft_masked = a_fft[:, freq_mask]
        b_fft_masked = b_fft[:, freq_mask]

        '''
        inner_prod = 4 * df * ((a_fft_masked.conj() * b_fft_masked).real / 
                              PSD[:, freq_mask]).sum()
        if debug:
            print(f"DEBUG [inner_product_from_fft]: Real inner product before GPU conversion: {inner_prod}")
        '''
        # Compute complex inner product
        complex_inner_prod = 4 * df * ((a_fft_masked.conj() * b_fft_masked) / 
                                      PSD[:, freq_mask]).sum()
        
        if maximize_phase:
            # Phase maximization: take magnitude of complex inner product
            inner_prod = xp.abs(complex_inner_prod)
            if debug:
                print(f"DEBUG [inner_product_from_fft]: Real signal inner product magnitude: {inner_prod}")
                print(f"DEBUG [inner_product_from_fft]: Original complex value: {complex_inner_prod}")
        else:
            # Traditional approach: take real part
            inner_prod = complex_inner_prod.real
            if debug:
                print(f"DEBUG [inner_product_from_fft]: Real signal inner product real part: {inner_prod}")
            
    
    if use_gpu:
        inner_prod = inner_prod.get()
    
    if debug:
        print(f"DEBUG [inner_product_from_fft]: Final inner product: {inner_prod}")
    
    return inner_prod


def calculate_overlap_1pa_vs_2pa(m1_1pa, m2_1pa, a_1pa, p0_1pa, e0_1pa, Y0_1pa, 
                               dist_1pa, qS_1pa, phiS_1pa, qK_1pa, phiK_1pa,
                               Phi_phi0_1pa, Phi_theta0_1pa, Phi_r0_1pa, 
                               chi2_1pa, phase_shift_plus=0.0, phase_shift_cross=0.0,
                               waveform_response=None, PSD=None, dt=None, T=None, N_fiducial=None, 
                               waveform_2pa_fft=None, xp=None, use_gpu=False):
    
    # Generate 1PA waveform
    evolve_1PA_infunc = True
    evolve_primary_infunc = False
    evolve_2PA_infunc = False  # This makes it 1PA    

    wave_params = [m1_1pa, m2_1pa, a_1pa, p0_1pa, e0_1pa, Y0_1pa, 
                           dist_1pa, qS_1pa, phiS_1pa, qK_1pa, phiK_1pa,
                           Phi_phi0_1pa, Phi_theta0_1pa, Phi_r0_1pa, chi2_1pa, evolve_1PA_infunc, evolve_primary_infunc, evolve_2PA_infunc]
    emri_kwargs = {"T":T, "dt": dt, '1PA': evolve_1PA_infunc, 'evolve_primary': evolve_primary_infunc, '2PA': evolve_2PA_infunc}
    waveform_1pa = waveform_response(*wave_params, **emri_kwargs)    
    
    # Convert 1PA to frequency domain   
    N_1pa = len(waveform_1pa[0])
    

    # Apply padding if 1PA waveform is shorter than fiducial 2PA
    if N_1pa < N_fiducial:
        pad_length = N_fiducial - N_1pa
        waveform_1pa[0] = xp.pad(xp.asarray(waveform_1pa[0]), (0, pad_length), mode='constant', constant_values=0)
        waveform_1pa[1] = xp.pad(xp.asarray(waveform_1pa[1]), (0, pad_length), mode='constant', constant_values=0)   
        waveform_1pa[2] = xp.pad(xp.asarray(waveform_1pa[2]), (0, pad_length), mode='constant', constant_values=0)
    elif N_1pa > N_fiducial:
        waveform_1pa[0] = waveform_1pa[0][:N_fiducial]
        waveform_1pa[1] = waveform_1pa[1][:N_fiducial]
        waveform_1pa[2] = waveform_1pa[2][:N_fiducial]

    waveform_1pa_fft = compute_fft_with_windowing(waveform_1pa, dt, use_gpu=use_gpu)
    
    # <h1|h2> - cross correlation between 1PA template and 2PA data
    h1_dot_h2 = inner_product_from_fft(waveform_2pa_fft, waveform_1pa_fft, PSD, dt, N_fiducial, use_gpu=use_gpu)
    h1_dot_h1 = inner_product_from_fft(waveform_1pa_fft, waveform_1pa_fft, PSD, dt, N_fiducial, use_gpu=use_gpu)
    h2_dot_h2 = inner_product_from_fft(waveform_2pa_fft, waveform_2pa_fft, PSD, dt, N_fiducial, use_gpu=use_gpu)
    
    # Overlap calculation
    overlap = np.abs(h1_dot_h2) / np.sqrt(h1_dot_h1 * h2_dot_h2)
    
    return overlap




# Convenience function to create 0PA waveform
def calculate_overlap_0pa_vs_2pa(m1_1pa, m2_1pa, a_1pa, p0_1pa, e0_1pa, Y0_1pa, 
                               dist_1pa, qS_1pa, phiS_1pa, qK_1pa, phiK_1pa,
                               Phi_phi0_1pa, Phi_theta0_1pa, Phi_r0_1pa, 
                               chi2_1pa, phase_shift_plus=0.0, phase_shift_cross=0.0,
                               waveform_response=None, PSD=None, dt=None, T=None, N_fiducial=None, 
                               waveform_2pa_fft=None, xp=None, use_gpu=False, maximize_phase=False):
    
    # Generate 1PA waveform
    evolve_1PA_infunc = False
    evolve_primary_infunc = False
    evolve_2PA_infunc = False  # This makes it 1PA    

    wave_params = [m1_1pa, m2_1pa, a_1pa, p0_1pa, e0_1pa, Y0_1pa, 
                           dist_1pa, qS_1pa, phiS_1pa, qK_1pa, phiK_1pa,
                           Phi_phi0_1pa, Phi_theta0_1pa, Phi_r0_1pa, chi2_1pa, evolve_1PA_infunc, evolve_primary_infunc, evolve_2PA_infunc]
    emri_kwargs = {"T":T, "dt": dt, '1PA': evolve_1PA_infunc, 'evolve_primary': evolve_primary_infunc, '2PA': evolve_2PA_infunc}
    waveform_1pa = waveform_response(*wave_params, **emri_kwargs)    
    
    # Convert 1PA to frequency domain   
    N_1pa = len(waveform_1pa[0])
    

    # Apply padding if 1PA waveform is shorter than fiducial 2PA
    if N_1pa < N_fiducial:
        pad_length = N_fiducial - N_1pa
        waveform_1pa[0] = xp.pad(xp.asarray(waveform_1pa[0]), (0, pad_length), mode='constant', constant_values=0)
        waveform_1pa[1] = xp.pad(xp.asarray(waveform_1pa[1]), (0, pad_length), mode='constant', constant_values=0)   
        waveform_1pa[2] = xp.pad(xp.asarray(waveform_1pa[2]), (0, pad_length), mode='constant', constant_values=0)
    elif N_1pa > N_fiducial:
        waveform_1pa[0] = waveform_1pa[0][:N_fiducial]
        waveform_1pa[1] = waveform_1pa[1][:N_fiducial]
        waveform_1pa[2] = waveform_1pa[2][:N_fiducial]

    waveform_1pa_fft = compute_fft_with_windowing(waveform_1pa, dt, use_gpu=use_gpu)
    
    # <h1|h2> - cross correlation between 1PA template and 2PA data
    h1_dot_h2 = inner_product_from_fft(waveform_2pa_fft, waveform_1pa_fft, PSD, dt, N_fiducial, use_gpu=use_gpu, maximize_phase=maximize_phase)
    h1_dot_h1 = inner_product_from_fft(waveform_1pa_fft, waveform_1pa_fft, PSD, dt, N_fiducial, use_gpu=use_gpu, maximize_phase=maximize_phase)
    h2_dot_h2 = inner_product_from_fft(waveform_2pa_fft, waveform_2pa_fft, PSD, dt, N_fiducial, use_gpu=use_gpu, maximize_phase=maximize_phase)
    
    # Overlap calculation
    overlap = np.abs(h1_dot_h2) / np.sqrt(h1_dot_h1 * h2_dot_h2)
    
    return overlap

def calculate_optimal_snr_1pa_vs_2pa(m1_1pa, m2_1pa, a_1pa, p0_1pa, e0_1pa, Y0_1pa, 
                               dist_1pa, qS_1pa, phiS_1pa, qK_1pa, phiK_1pa,
                               Phi_phi0_1pa, Phi_theta0_1pa, Phi_r0_1pa, 
                               chi2_1pa, phase_shift_plus=0.0, phase_shift_cross=0.0,
                               waveform_response=None, PSD=None, dt=None, T=None, N_fiducial=None, 
                               waveform_2pa_fft=None, xp=None, use_gpu=False):
    
    # Generate 1PA waveform
    evolve_1PA_infunc = True
    evolve_primary_infunc = False
    evolve_2PA_infunc = False  # This makes it 1PA    

    wave_params = [m1_1pa, m2_1pa, a_1pa, p0_1pa, e0_1pa, Y0_1pa, 
                           dist_1pa, qS_1pa, phiS_1pa, qK_1pa, phiK_1pa,
                           Phi_phi0_1pa, Phi_theta0_1pa, Phi_r0_1pa, chi2_1pa, evolve_1PA_infunc, evolve_primary_infunc, evolve_2PA_infunc]
    emri_kwargs = {"T":T, "dt": dt, '1PA': evolve_1PA_infunc, 'evolve_primary': evolve_primary_infunc, '2PA': evolve_2PA_infunc}
    waveform_1pa = waveform_response(*wave_params, **emri_kwargs)    
    
    # Convert 1PA to frequency domain   
    N_1pa = len(waveform_1pa[0])
    

    # Apply padding if 1PA waveform is shorter than fiducial 2PA
    if N_1pa < N_fiducial:
        pad_length = N_fiducial - N_1pa
        waveform_1pa[0] = xp.pad(xp.asarray(waveform_1pa[0]), (0, pad_length), mode='constant', constant_values=0)
        waveform_1pa[1] = xp.pad(xp.asarray(waveform_1pa[1]), (0, pad_length), mode='constant', constant_values=0)   
        waveform_1pa[2] = xp.pad(xp.asarray(waveform_1pa[2]), (0, pad_length), mode='constant', constant_values=0)
    elif N_1pa > N_fiducial:
        waveform_1pa[0] = waveform_1pa[0][:N_fiducial]
        waveform_1pa[1] = waveform_1pa[1][:N_fiducial]
        waveform_1pa[2] = waveform_1pa[2][:N_fiducial]

    waveform_1pa_fft = compute_fft_with_windowing(waveform_1pa, dt, use_gpu=use_gpu)
    
    # <h1|h2> - cross correlation between 1PA template and 2PA data
    h1_dot_h2 = inner_product_from_fft(waveform_2pa_fft, waveform_1pa_fft, PSD, dt, N_fiducial, use_gpu=use_gpu)
    h1_dot_h1 = inner_product_from_fft(waveform_1pa_fft, waveform_1pa_fft, PSD, dt, N_fiducial, use_gpu=use_gpu)
    #h2_dot_h2 = inner_product_from_fft(waveform_2pa_fft, waveform_2pa_fft, PSD, dt, N_fiducial, use_gpu=use_gpu)
    
    # Overlap calculation
    overlap = np.abs(h1_dot_h2) / np.sqrt(h1_dot_h1)
    
    return overlap    

def calculate_optimal_snr_0pa_vs_2pa(m1_1pa, m2_1pa, a_1pa, p0_1pa, e0_1pa, Y0_1pa, 
                               dist_1pa, qS_1pa, phiS_1pa, qK_1pa, phiK_1pa,
                               Phi_phi0_1pa, Phi_theta0_1pa, Phi_r0_1pa, 
                               chi2_1pa, phase_shift_plus=0.0, phase_shift_cross=0.0,
                               waveform_response=None, PSD=None, dt=None, T=None, N_fiducial=None, 
                               waveform_2pa_fft=None, xp=None, use_gpu=False, maximize_phase=False):
    
    # Generate 1PA waveform
    evolve_1PA_infunc = False
    evolve_primary_infunc = False
    evolve_2PA_infunc = False 

    wave_params = [m1_1pa, m2_1pa, a_1pa, p0_1pa, e0_1pa, Y0_1pa, 
                           dist_1pa, qS_1pa, phiS_1pa, qK_1pa, phiK_1pa,
                           Phi_phi0_1pa, Phi_theta0_1pa, Phi_r0_1pa, chi2_1pa, evolve_1PA_infunc, evolve_primary_infunc, evolve_2PA_infunc]
    emri_kwargs = {"T":T, "dt": dt, '1PA': evolve_1PA_infunc, 'evolve_primary': evolve_primary_infunc, '2PA': evolve_2PA_infunc}
    waveform_1pa = waveform_response(*wave_params, **emri_kwargs)    
    
    # Convert 1PA to frequency domain   
    N_1pa = len(waveform_1pa[0])
    

    # Apply padding if 1PA waveform is shorter than fiducial 2PA
    if N_1pa < N_fiducial:
        pad_length = N_fiducial - N_1pa
        waveform_1pa[0] = xp.pad(xp.asarray(waveform_1pa[0]), (0, pad_length), mode='constant', constant_values=0)
        waveform_1pa[1] = xp.pad(xp.asarray(waveform_1pa[1]), (0, pad_length), mode='constant', constant_values=0)   
        waveform_1pa[2] = xp.pad(xp.asarray(waveform_1pa[2]), (0, pad_length), mode='constant', constant_values=0)
    elif N_1pa > N_fiducial:
        waveform_1pa[0] = waveform_1pa[0][:N_fiducial]
        waveform_1pa[1] = waveform_1pa[1][:N_fiducial]
        waveform_1pa[2] = waveform_1pa[2][:N_fiducial]

    waveform_1pa_fft = compute_fft_with_windowing(waveform_1pa, dt, use_gpu=use_gpu)
    
    # <h1|h2> - cross correlation between 1PA template and 2PA data
    h1_dot_h2 = inner_product_from_fft(waveform_2pa_fft, waveform_1pa_fft, PSD, dt, N_fiducial, use_gpu=use_gpu, maximize_phase=maximize_phase)
    h1_dot_h1 = inner_product_from_fft(waveform_1pa_fft, waveform_1pa_fft, PSD, dt, N_fiducial, use_gpu=use_gpu, maximize_phase=maximize_phase)
    #h2_dot_h2 = inner_product_from_fft(waveform_2pa_fft, waveform_2pa_fft, PSD, dt, N_fiducial, use_gpu=use_gpu)
    
    # Overlap calculation
    overlap = np.abs(h1_dot_h2) / np.sqrt(h1_dot_h1)
    
    return overlap
