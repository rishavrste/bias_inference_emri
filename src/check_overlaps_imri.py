"""
Compute 1PA-vs-2PA overlaps for IMRI cases.

For each case:
  Signal  (h_true): 2PA waveform  (evolve_1PA=True, evolve_2PA=True)
  Template (h_1pa): 1PA waveform  (evolve_1PA=True, evolve_2PA=False)
  Overlap = <h_1pa|h_true> / sqrt(<h_1pa|h_1pa> * <h_true|h_true>)  (maximize_phase=False)

Usage:
    python check_overlaps_imri.py [--no-gpu] [--nchannels 2]
    python check_overlaps_imri.py --signal-file /path/sig.npy --result-file /path/res.npy
"""

import argparse, os, sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--nchannels", type=int, default=2)
parser.add_argument("--signal-file",
    default="/scratch/e1583490/SuperKludege_Optimizations/opt_grid/signal_parameter_array_IMRI.npy")
parser.add_argument("--result-file",
    default="/scratch/e1583490/SuperKludege_Optimizations/opt_grid/result_parameter_array_IMRI_with_phase_1PA.npy")
args = parser.parse_args()

NCHANNELS = args.nchannels

# ── imports ───────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/svu/e1583490/packages_for_bias/SuperKludgeIndie/src")

import cupy as cp
from few.waveform import GenerateEMRIWaveform
from lisatools.sensitivity import get_sensitivity, A2TDISens, E2TDISens, T2TDISens
from lisatools.detector import EqualArmlengthOrbits
from fastlisaresponse import ResponseWrapper
from stableemrifisher.utils import generate_PSD
from few.waveform.waveform import SuperKludgeWaveform
from misc import compute_fft_with_windowing, inner_prod


PARAM_NAMES = ["m1","m2","a","p0","e0","Y0","dist","qS","phiS","qK","phiK",
               "Phi_phi0","Phi_theta0","Phi_r0","dt","T","chi2"]

# ── waveform builder ──────────────────────────────────────────────────────────
def build_response(T, dt):
    sum_kwargs = dict(pad_output=True, odd_len=True)
    model = GenerateEMRIWaveform(
        SuperKludgeWaveform, sum_kwargs=sum_kwargs,
        return_list=False, use_gpu=True)
    return ResponseWrapper(
        waveform_gen=model,
        Tobs=T, t0=10000.0, dt=dt,
        index_lambda=8, index_beta=7,
        flip_hx=True, is_ecliptic_latitude=False,
        remove_garbage="zero",
        orbits=EqualArmlengthOrbits(use_gpu=True),
        force_backend="cuda12x",
        order=20, tdi="2nd generation", tdi_chan="AET")


def gen_waveform_time(response, params14, chi2, evolve_1PA, evolve_2PA, T, dt):
    """Return time-domain waveform array on GPU, shape (NCHANNELS, N)."""
    wave_params = list(params14) + [chi2, evolve_1PA, False, evolve_2PA]
    emri_kw = {"T": T, "dt": dt, "chi2": chi2,
                "evolve_1PA": evolve_1PA, "evolve_primary": False,
                "evolve_2PA": evolve_2PA}
    h = cp.array(response(*wave_params, **emri_kw))[:NCHANNELS, :]
    return h


def compute_overlap(h_sig_f, h_res_f, PSD, delta_f):
    """Overlap matching calculate_detection_overlap(..., maximize_phase=False).

    overlap = <h_sig|h_res> / sqrt(<h_sig|h_sig> * <h_res|h_res>)
    """
    num = inner_prod(h_sig_f, h_res_f, PSD, delta_f, xp=cp)
    d1  = inner_prod(h_sig_f, h_sig_f, PSD, delta_f, xp=cp)
    d2  = inner_prod(h_res_f, h_res_f, PSD, delta_f, xp=cp)
    num = float(num.get()) if hasattr(num, 'get') else float(num)
    d1  = float(d1.get())  if hasattr(d1,  'get') else float(d1)
    d2  = float(d2.get())  if hasattr(d2,  'get') else float(d2)
    return num / np.sqrt(d1 * d2)


# ── load arrays ───────────────────────────────────────────────────────────────
signal_array = np.load(args.signal_file, allow_pickle=True)
result_array = np.load(args.result_file, allow_pickle=True)
N_cases = signal_array.shape[0]

print(f"Signal : {args.signal_file}")
print(f"Results: {args.result_file}")
print(f"Cases  : {N_cases}  |  nchannels={NCHANNELS}  |  GPU=True (always)\n")

channels_psd = [A2TDISens, E2TDISens, T2TDISens][:NCHANNELS]
noise_kwargs  = [{"sens_fn": ch} for ch in channels_psd]

# All IMRI cases share the same T and dt — build the response wrapper once
T0  = float(signal_array[0, PARAM_NAMES.index('T')])
dt0 = float(signal_array[0, PARAM_NAMES.index('dt')])
print(f"Building response wrapper (T={T0}, dt={dt0}) ...")
response = build_response(T0, dt0)
print("Done.\n")

# ── per-case loop ─────────────────────────────────────────────────────────────
print(f"{'i':>4}  {'m1_sig':>12} {'m2_sig':>8} {'a_sig':>6} "
      f"{'m1_res':>12} {'m2_res':>8} {'chi2_res':>8}  {'overlap':>20}")
print("-" * 90)

overlaps = []
for i in range(N_cases):
    sig = dict(zip(PARAM_NAMES, signal_array[i]))
    res = dict(zip(PARAM_NAMES, result_array[i]))

    T, dt, chi2_sig = float(sig['T']), float(sig['dt']), float(sig['chi2'])
    chi2_res = float(res['chi2'])
    params14_sig = signal_array[i, :14]

    # result uses its own intrinsic params; sky/dist/Y0 kept from signal
    params14_res = params14_sig.copy()
    for k in ['m1', 'm2', 'a', 'p0', 'e0', 'Phi_phi0', 'Phi_r0']:
        params14_res[PARAM_NAMES.index(k)] = res[k]

    # ── 2PA signal (truth) ────────────────────────────────────────────────
    h_sig_t = gen_waveform_time(response, params14_sig, chi2_sig,
                                evolve_1PA=True, evolve_2PA=True, T=T, dt=dt)
    N_fid   = h_sig_t.shape[1]
    delta_f = float(1.0 / (N_fid * dt))

    PSD = cp.array(generate_PSD(
        waveform=h_sig_t, dt=dt,
        noise_PSD=get_sensitivity,
        channels=channels_psd,
        noise_kwargs=noise_kwargs,
        use_gpu=True))

    h_sig_f = compute_fft_with_windowing(h_sig_t, dt, N_fid, use_gpu=True, n_channels=NCHANNELS)

    # ── 1PA result template ───────────────────────────────────────────────
    h_res_t = gen_waveform_time(response, params14_res, chi2_res,
                                evolve_1PA=True, evolve_2PA=False, T=T, dt=dt)
    # ensure same length as signal (pad with zeros or trim)
    if h_res_t.shape[1] < N_fid:
        pad = cp.zeros((NCHANNELS, N_fid - h_res_t.shape[1]))
        h_res_t = cp.concatenate([h_res_t, pad], axis=1)
    else:
        h_res_t = h_res_t[:, :N_fid]

    h_res_f = compute_fft_with_windowing(h_res_t, dt, N_fid, use_gpu=True, n_channels=NCHANNELS)

    # ── overlap ───────────────────────────────────────────────────────────
    try:
        ov = compute_overlap(h_sig_f, h_res_f, PSD, delta_f)
    except Exception as e:
        print(f"  [ERROR case {i}]: {e}")
        ov = float('nan')

    overlaps.append(ov)
    print(f"{i:>4}  {sig['m1']:>12.1f} {sig['m2']:>8.1f} {sig['a']:>6.3f} "
          f"{res['m1']:>12.1f} {res['m2']:>8.1f} {chi2_res:>8.5f}  {ov:.15f}")

# ── summary ───────────────────────────────────────────────────────────────────
arr = np.array([o for o in overlaps if not np.isnan(o)])
print("-" * 90)
print(f"\nSummary  |  mean={arr.mean():.15f}  "
      f"min={arr.min():.15f} (case {int(np.nanargmin(overlaps))})  "
      f"max={arr.max():.15f}")
good = sum(o >= 0.99 for o in overlaps if not np.isnan(o))
print(f"Cases >= 0.99 overlap: {good}/{N_cases}")
print(f"\nAll overlaps (full precision):")
for i, o in enumerate(overlaps):
    flag = " <-- poor" if (not np.isnan(o) and o < 0.99) else ""
    print(f"  case {i:>2}: {o:.15f}{flag}")
