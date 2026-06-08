"""
Compute 1PA vs 2PA overlap at injected true parameters for all EMRI grid points.
Overlap = <s_2PA | h_1PA> / (||s_2PA|| * ||h_1PA||)
Results saved to overlap_1pa_vs_2pa.txt
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import cupy as cp
    xp = cp
    use_gpu = True
    print("[INFO] Using GPU (CuPy)")
except ImportError:
    xp = np
    use_gpu = False
    print("[INFO] CuPy not found, using CPU")

from inference import build_waveform_response
from misc import compute_fft_with_windowing, inner_prod
from stableemrifisher.utils import generate_PSD
from lisatools.sensitivity import get_sensitivity, A2TDISens, E2TDISens

PARAM_FILE = "/scratch/josh.mat/bias_inference_emri/data/signal_parameter_array_EMRI.npy"
OUT_FILE   = "/scratch/josh.mat/bias_inference_emri/src/overlap_1pa_vs_2pa.txt"
NCHANNELS  = 2


def compute_overlap_at_true(row):
    m1, m2, a, p0, e0, Y0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 = row[:14]
    dt = row[14]
    T  = row[15]
    chi2 = row[16]

    wr = build_waveform_response(T=T, dt=dt, use_gpu=use_gpu)

    base_wave_params = [m1, m2, a, p0, e0, Y0, dist, qS, phiS, qK, phiK,
                        Phi_phi0, Phi_theta0, Phi_r0]

    # 2PA signal (true)
    sig_params  = base_wave_params + [chi2, True, False, True]
    sig_kwargs  = {"T": T, "dt": dt, "chi2": chi2,
                   "evolve_1PA": True, "evolve_primary": False, "evolve_2PA": True}
    s = xp.array(wr(*sig_params, **sig_kwargs))[:NCHANNELS, :]

    # 1PA template (bias run template)
    tmpl_params = base_wave_params + [chi2, True, False, False]
    tmpl_kwargs = {"T": T, "dt": dt, "chi2": chi2,
                   "evolve_1PA": True, "evolve_primary": False, "evolve_2PA": False}
    h = xp.array(wr(*tmpl_params, **tmpl_kwargs))[:NCHANNELS, :]

    N = s.shape[1]
    freq    = np.fft.rfftfreq(N, dt)
    delta_f = freq[1] - freq[0]

    channels = [A2TDISens, E2TDISens]
    noise_kwargs = [{"sens_fn": ch} for ch in channels]
    PSD = xp.array(generate_PSD(
        waveform=s, dt=dt, noise_PSD=get_sensitivity,
        channels=[A2TDISens, E2TDISens],
        noise_kwargs=noise_kwargs,
        use_gpu=use_gpu,
    ))

    s_f = compute_fft_with_windowing(s, dt, N, use_gpu=use_gpu, n_channels=NCHANNELS)
    h_f = compute_fft_with_windowing(h, dt, N, use_gpu=use_gpu, n_channels=NCHANNELS)

    ss = float(xp.sqrt(inner_prod(s_f, s_f, PSD, delta_f, xp=xp)))
    hh = float(xp.sqrt(inner_prod(h_f, h_f, PSD, delta_f, xp=xp)))
    sh = float(inner_prod(s_f, h_f, PSD, delta_f, xp=xp))

    snr_signal   = ss
    overlap      = sh / (ss * hh)
    mismatch     = 1.0 - overlap

    return snr_signal, overlap, mismatch, s.shape[1], h.shape[1]


def main():
    params = np.load(PARAM_FILE)
    print(f"[INFO] Parameter array shape: {params.shape}")
    n_pts = params.shape[0]

    results = []
    with open(OUT_FILE, "w") as f:
        header = (f"{'idx':>4}  {'a':>7}  {'e0':>7}  {'p0':>8}  "
                  f"{'SNR':>9}  {'overlap':>18}  {'mismatch':>14}  "
                  f"{'N_sig':>8}  {'N_tmpl':>8}\n")
        f.write(header)
        f.write("-" * len(header) + "\n")
        print(header, end="")

        for i in range(n_pts):
            row = params[i]
            a, e0, p0 = row[2], row[4], row[3]
            print(f"[{i:2d}/{n_pts}] a={a:.3f}  e0={e0:.3f}  p0={p0:.4f} ...", flush=True)
            try:
                snr, ov, mm, N_s, N_h = compute_overlap_at_true(row)
                line = (f"{i:4d}  {a:7.4f}  {e0:7.4f}  {p0:8.5f}  "
                        f"{snr:9.4f}  {ov:18.15f}  {mm:14.6e}  "
                        f"{N_s:8d}  {N_h:8d}\n")
            except Exception as e:
                line = f"{i:4d}  {a:7.4f}  {e0:7.4f}  {p0:8.5f}  ERROR: {e}\n"
                print(f"  ERROR: {e}")
            f.write(line)
            f.flush()
            print(line, end="")
            results.append(line)

        # Summary statistics
        mismatch_vals = []
        overlap_vals  = []
        for line in results:
            parts = line.split()
            if len(parts) >= 7 and parts[0].isdigit():
                try:
                    overlap_vals.append(float(parts[5]))
                    mismatch_vals.append(float(parts[6]))
                except ValueError:
                    pass

        if mismatch_vals:
            mm_arr = np.array(mismatch_vals)
            ov_arr = np.array(overlap_vals)
            summary = (
                f"\n{'='*70}\n"
                f"SUMMARY ({len(mm_arr)} points)\n"
                f"  Mismatch  min={mm_arr.min():.4e}  max={mm_arr.max():.4e}  "
                f"median={np.median(mm_arr):.4e}\n"
                f"  Overlap   min={ov_arr.min():.15f}  max={ov_arr.max():.15f}  "
                f"median={np.median(ov_arr):.15f}\n"
            )
            f.write(summary)
            print(summary)


if __name__ == "__main__":
    main()
