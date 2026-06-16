"""
Recompute 1PA-vs-2PA overlaps on the GPU for every case in a signal/result grid.

For each case:
  Signal   (h_true): 2PA waveform (evolve_1PA=True, evolve_2PA=True)
  Template (h_fit):  1PA best fit (evolve_1PA=True, evolve_2PA=False)
  Overlap = <h_true|h_fit> / sqrt(<h_true|h_true> * <h_fit|h_fit>)  (maximize_phase=False)

This is an independent verification of the overlaps recorded by inference.py's
optimizer runs / by sync_result_array.py, used to confirm a result file is
trustworthy before relying on it.

Usage:
    python tests/check_overlaps.py
    python tests/check_overlaps.py --signal-file /path/sig.npy --result-file /path/res.npy
    python tests/check_overlaps.py --nchannels 3
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import cupy as cp
from few.waveform import GenerateEMRIWaveform
from few.waveform.waveform import SuperKludgeWaveform
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity, A2TDISens, E2TDISens, T2TDISens
from stableemrifisher.utils import generate_PSD
from misc import compute_fft_with_windowing, inner_prod

COLUMN_ORDER = ["m1", "m2", "a", "p0", "e0", "xI0", "dist", "qS", "phiS", "qK", "phiK",
                "Phi_phi0", "Phi_theta0", "Phi_r0", "dt", "T", "chi2"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--signal-file", default=os.environ.get(
        "OPT_PARAM_FILE",
        "/scratch/e1583490/SuperKludege_Optimizations/opt_grid/signal_parameter_array_IMRI_TAIL.npy"))
    p.add_argument("--result-file", default=os.environ.get(
        "OPT_RESULT_FILE",
        "/scratch/e1583490/SuperKludege_Optimizations/opt_grid/result_parameter_array_IMRI_TAIL_with_phase_1PA.npy"))
    p.add_argument("--nchannels", type=int, default=2, choices=(2, 3))
    p.add_argument("--threshold", type=float, default=0.99)
    return p.parse_args()


def row_to_params(row):
    return dict(zip(COLUMN_ORDER, row))


def build_response(T, dt):
    waveform_model = GenerateEMRIWaveform(
        SuperKludgeWaveform,
        sum_kwargs=dict(pad_output=True, odd_len=True),
        return_list=False, use_gpu=True)
    return ResponseWrapper(
        waveform_gen=waveform_model, Tobs=T, t0=10000.0, dt=dt,
        index_lambda=8, index_beta=7, flip_hx=True,
        is_ecliptic_latitude=False, remove_garbage="zero",
        orbits=EqualArmlengthOrbits(use_gpu=True),
        force_backend="cuda12x", order=20,
        tdi="2nd generation", tdi_chan="AET")


def compute_overlap(response, sig_row, fit_params, psd_funcs, delta_f, n_fiducial, nchannels):
    dt = float(sig_row[14])
    T = float(sig_row[15])
    chi2_sig = float(sig_row[16])
    m1s, m2s, as_, p0s, e0s, Y0, dist, qS, phiS, qK, phiK, Phi_phi0s, Phi_theta0s, Phi_r0s = sig_row[:14]

    h_true = cp.array(response(
        m1s, m2s, as_, p0s, e0s, Y0, dist, qS, phiS, qK, phiK,
        Phi_phi0s, Phi_theta0s, Phi_r0s,
        chi2_sig, True, False, True,
        T=T, dt=dt, chi2=chi2_sig,
        evolve_1PA=True, evolve_primary=False, evolve_2PA=True,
    ))[0:nchannels, :]

    p = fit_params
    chi2_fit = float(p.get('chi2', chi2_sig))
    h_fit = cp.array(response(
        float(p['m1']), float(p['m2']), float(p['a']),
        float(p['p0']), float(p['e0']), float(p['xI0']),
        float(p['dist']), float(p['qS']), float(p['phiS']),
        float(p['qK']), float(p['phiK']),
        float(p['Phi_phi0']), float(p['Phi_theta0']), float(p['Phi_r0']),
        chi2_fit, True, False, False,
        T=T, dt=dt, chi2=chi2_fit,
        evolve_1PA=True, evolve_primary=False, evolve_2PA=False,
    ))[0:nchannels, :]

    h_true_f = compute_fft_with_windowing(h_true, dt, n_fiducial, use_gpu=True, n_channels=nchannels)
    h_fit_f = compute_fft_with_windowing(h_fit, dt, n_fiducial, use_gpu=True, n_channels=nchannels)

    num = inner_prod(h_true_f, h_fit_f, psd_funcs, delta_f, xp=cp)
    denom = cp.sqrt(
        inner_prod(h_true_f, h_true_f, psd_funcs, delta_f, xp=cp) *
        inner_prod(h_fit_f, h_fit_f, psd_funcs, delta_f, xp=cp)
    )
    return float(num / denom)


def main():
    args = parse_args()
    sig_arr = np.load(args.signal_file)
    res_arr = np.load(args.result_file, allow_pickle=True)
    n = sig_arr.shape[0]
    nchannels = args.nchannels

    T = float(sig_arr[0, 15])
    dt = float(sig_arr[0, 14])

    print(f"Signal : {args.signal_file}")
    print(f"Results: {args.result_file}")
    print(f"Cases  : {n}  |  nchannels={nchannels}\n")

    print("Building ResponseWrapper (once)...")
    response = build_response(T, dt)

    print("Generating reference waveform for PSD...")
    s = sig_arr[0]
    m1, m2, a, p0, e0, Y0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 = s[:14]
    chi2_ref = float(s[16])
    h_ref = cp.array(response(
        m1, m2, a, p0, e0, Y0, dist, qS, phiS, qK, phiK,
        Phi_phi0, Phi_theta0, Phi_r0,
        chi2_ref, True, False, True,
        T=T, dt=dt, chi2=chi2_ref,
        evolve_1PA=True, evolve_primary=False, evolve_2PA=True,
    ))[0:nchannels, :]
    n_fiducial = h_ref.shape[1]

    channels = [A2TDISens, E2TDISens, T2TDISens][:nchannels]
    noise_kwargs = [{"sens_fn": ch} for ch in channels]
    psd_funcs = cp.array(generate_PSD(
        waveform=h_ref, dt=dt, noise_PSD=get_sensitivity,
        channels=channels, noise_kwargs=noise_kwargs, use_gpu=True,
    ))
    freq = np.fft.rfftfreq(n_fiducial, dt)
    delta_f = freq[1] - freq[0]
    print(f"N={n_fiducial}, delta_f={delta_f:.6e}\n")

    print(f"{'Case':>5}  {'Overlap':>14}  Status")
    print("-" * 36)
    overlaps = []
    for i in range(n):
        fit_params = row_to_params(res_arr[i])
        try:
            ov = compute_overlap(response, sig_arr[i], fit_params, psd_funcs, delta_f, n_fiducial, nchannels)
        except Exception as e:
            print(f"{i:>5}  {'ERROR':>14}  {e}")
            overlaps.append(float('nan'))
            continue
        overlaps.append(ov)
        flag = "OK" if ov >= args.threshold else f"<{args.threshold} !"
        print(f"{i:>5}  {ov:>14.10f}  {flag}", flush=True)

    arr = np.array([o for o in overlaps if not np.isnan(o)])
    good = int(np.sum(arr >= args.threshold))
    print("-" * 36)
    print(f"\nCases >= {args.threshold}: {good}/{n}")
    if arr.size:
        print(f"mean={arr.mean():.10f}  min={arr.min():.10f}  max={arr.max():.10f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
