"""
Compute overlaps for the IMRI TAIL configuration
(m2=10000 M_sun, T=0.25 yr, epsilon=1e-2, Grid C).

(A) Injection-point overlaps (0PA vs 2PA and 1PA vs 2PA at injected parameters).
    Output: /scratch/josh.mat/plot_scripts/data/signal/injection_overlaps_IMRI_TAIL.npy
      shape (25, 2): col 0 = mismatch_0pa,  col 1 = mismatch_1pa

(B) Recovered-point final overlap:
      1PA template at Rishav's recovered parameters vs 2PA signal at injected parameters.
    Output: updates col 17 of
      /scratch/josh.mat/plot_scripts/data/recovered/recovered_parameter_array_IMRI_TAIL_1pa.npy

A log is written alongside this script as compute_overlaps_IMRI_TAIL.txt.
Results are saved after each grid point so partial output survives interruption.
"""
import sys
import os
import numpy as np

try:
    import cupy as cp
    xp = cp
    use_gpu = True
    print("[INFO] Using GPU (CuPy)")
except ImportError:
    xp = np
    use_gpu = False
    print("[INFO] CuPy not found, using CPU")

import few as _few
_few_cfg = _few.get_config_setter(reset=True)
_few_cfg.enable_backends("cuda12x")

from few.waveform import GenerateEMRIWaveform
from few.waveform.waveform import SuperKludgeWaveform
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity, A2TDISens, E2TDISens
from stableemrifisher.utils import generate_PSD

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from misc import compute_fft_with_windowing, inner_prod, inner_prod_without_phase

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SIG_FILE = "/scratch/josh.mat/plot_scripts/data/signal/signal_parameter_array_IMRI_TAIL.npy"
REC_FILE = "/scratch/josh.mat/plot_scripts/data/recovered/recovered_parameter_array_IMRI_TAIL_1pa.npy"
OUT_INJ  = "/scratch/josh.mat/plot_scripts/data/signal/injection_overlaps_IMRI_TAIL.npy"
OUT_TXT  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "compute_overlaps_IMRI_TAIL.txt")

NCHANNELS = 2   # A and E only
T_LISA    = 0.25
dt        = 10.0

# ---------------------------------------------------------------------------
# Build LISA response wrapper (T=0.25 yr for IMRI TAIL)
# ---------------------------------------------------------------------------
_waveform_model = GenerateEMRIWaveform(
    SuperKludgeWaveform,
    sum_kwargs=dict(pad_output=True, odd_len=True),
    return_list=False,
    use_gpu=use_gpu,
)
wr = ResponseWrapper(
    waveform_gen=_waveform_model,
    Tobs=T_LISA, dt=dt, t0=10000.0,
    index_lambda=8, index_beta=7,
    flip_hx=True, is_ecliptic_latitude=False,
    remove_garbage="zero",
    orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
    force_backend="cuda12x" if use_gpu else "cpu",
    order=20,
    tdi="2nd generation",
    tdi_chan="AET",
)
print("[INFO] ResponseWrapper built (T=0.25 yr)")

channels     = [A2TDISens, E2TDISens]
noise_kwargs = [{"sens_fn": ch} for ch in channels]


def _waveform(params_14, chi2, ev1pa, evprim, ev2pa):
    raw = xp.array(
        wr(*params_14, chi2, ev1pa, evprim, ev2pa,
           T=T_LISA, dt=dt,
           evolve_1PA=ev1pa, evolve_primary=evprim, evolve_2PA=ev2pa)
    )[:NCHANNELS, :]
    return raw


def _setup_psd_and_signal(s_raw):
    N       = s_raw.shape[1]
    freq    = np.fft.rfftfreq(N, dt)
    delta_f = freq[1] - freq[0]
    PSD = xp.array(generate_PSD(
        waveform=s_raw, dt=dt, noise_PSD=get_sensitivity,
        channels=channels, noise_kwargs=noise_kwargs, use_gpu=use_gpu,
    ))
    s_f = compute_fft_with_windowing(s_raw, dt, N, use_gpu=use_gpu, n_channels=NCHANNELS)
    ss  = float(xp.sqrt(inner_prod(s_f, s_f, PSD, delta_f, xp=xp)))
    return s_f, PSD, ss, N, delta_f


def _overlap(h_raw, s_f, PSD, ss, N, delta_f):
    Nh = h_raw.shape[1]
    if Nh < N:
        h_raw = xp.concatenate([h_raw, xp.zeros((NCHANNELS, N - Nh), dtype=h_raw.dtype)], axis=1)
    elif Nh > N:
        h_raw = h_raw[:, :N]
    h_f = compute_fft_with_windowing(h_raw, dt, N, use_gpu=use_gpu, n_channels=NCHANNELS)
    hh  = float(xp.sqrt(inner_prod(h_f, h_f, PSD, delta_f, xp=xp)))
    sh  = float(inner_prod_without_phase(s_f, h_f, PSD, delta_f, xp=xp))
    return sh / (ss * hh)


# ---------------------------------------------------------------------------
# Load signal array
# ---------------------------------------------------------------------------
RCOL = dict(m1=0, m2=1, a=2, p0=3, e0=4,
            x0=5, dist=6, qS=7, phiS=8, qK=9, phiK=10,
            Phi_phi0=11, Phi_theta0=12, Phi_r0=13,
            dt=14, T=15, chi2=16, final_overlap=17)

sig_params = np.load(SIG_FILE)   # (25, 17)
rec_params = np.load(REC_FILE)   # (25, 18)
n_pts      = sig_params.shape[0]
print(f"[INFO] Signal shape: {sig_params.shape}  Recovered shape: {rec_params.shape}")

# Initialise output (NaN); resume from partial file if present
out_inj = np.full((n_pts, 2), np.nan)
if os.path.exists(OUT_INJ):
    out_inj = np.load(OUT_INJ)
    n_done = int(np.sum(~np.isnan(out_inj[:, 0])))
    print(f"[INFO] Resuming: {n_done}/{n_pts} injection overlaps already computed")

# Work on a copy of the recovered array so we can update col 17 in-place
rec_out = rec_params.copy()

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
with open(OUT_TXT, "w") as log:
    hdr = (f"{'idx':>4}  {'a':>7}  {'e0':>7}  {'p0':>8}  "
           f"{'ov_0pa_inj':>20}  {'mis_0pa_inj':>13}  "
           f"{'ov_1pa_inj':>20}  {'mis_1pa_inj':>13}  "
           f"{'ov_1pa_rec':>20}  {'mis_1pa_rec':>13}\n")
    sep = "-" * len(hdr) + "\n"
    log.write(hdr + sep)
    print(hdr, end="")

    for i in range(n_pts):
        sig = sig_params[i]
        rec = rec_params[i]
        a, e0, p0 = sig[2], sig[4], sig[3]
        print(f"[{i:2d}/{n_pts}] a={a:+.2f}  e0={e0:.1f}  p0={p0:.4f} ...", flush=True)

        base_inj = [sig[RCOL['m1']], sig[RCOL['m2']], sig[RCOL['a']],
                    sig[RCOL['p0']], sig[RCOL['e0']], sig[RCOL['x0']],
                    1.0,
                    sig[RCOL['qS']], sig[RCOL['phiS']],
                    sig[RCOL['qK']], sig[RCOL['phiK']],
                    sig[RCOL['Phi_phi0']], sig[RCOL['Phi_theta0']], sig[RCOL['Phi_r0']]]
        chi2_inj = float(sig[RCOL['chi2']])

        base_rec = [rec[RCOL['m1']], rec[RCOL['m2']], rec[RCOL['a']],
                    rec[RCOL['p0']], rec[RCOL['e0']], rec[RCOL['x0']],
                    1.0,
                    rec[RCOL['qS']], rec[RCOL['phiS']],
                    rec[RCOL['qK']], rec[RCOL['phiK']],
                    rec[RCOL['Phi_phi0']], rec[RCOL['Phi_theta0']], rec[RCOL['Phi_r0']]]
        chi2_rec = float(rec[RCOL['chi2']])

        ov_0pa_inj = ov_1pa_inj = ov_1pa_rec = np.nan
        mis_0pa_inj = mis_1pa_inj = mis_1pa_rec = np.nan

        try:
            s_raw = _waveform(base_inj, chi2_inj, True, False, True)
            s_f, PSD, ss, N, delta_f = _setup_psd_and_signal(s_raw)

            # --- (A) Injection-point overlaps (skip if already done) ---
            if np.isnan(out_inj[i, 0]):
                h0_raw = _waveform(base_inj, chi2_inj, False, False, False)
                ov_0pa_inj  = _overlap(h0_raw, s_f, PSD, ss, N, delta_f)
                mis_0pa_inj = 1.0 - ov_0pa_inj
                out_inj[i, 0] = mis_0pa_inj
            else:
                mis_0pa_inj = out_inj[i, 0]; ov_0pa_inj = 1.0 - mis_0pa_inj

            if np.isnan(out_inj[i, 1]):
                h1_raw = _waveform(base_inj, chi2_inj, True, False, False)
                ov_1pa_inj  = _overlap(h1_raw, s_f, PSD, ss, N, delta_f)
                mis_1pa_inj = 1.0 - ov_1pa_inj
                out_inj[i, 1] = mis_1pa_inj
            else:
                mis_1pa_inj = out_inj[i, 1]; ov_1pa_inj = 1.0 - mis_1pa_inj

            # --- (B) Final overlap at recovered point ---
            hr_raw      = _waveform(base_rec, chi2_rec, True, False, False)
            ov_1pa_rec  = _overlap(hr_raw, s_f, PSD, ss, N, delta_f)
            mis_1pa_rec = 1.0 - ov_1pa_rec
            rec_out[i, RCOL['final_overlap']] = ov_1pa_rec

            line = (f"{i:4d}  {a:7.4f}  {e0:7.4f}  {p0:8.5f}  "
                    f"{ov_0pa_inj:20.15f}  {mis_0pa_inj:13.6e}  "
                    f"{ov_1pa_inj:20.15f}  {mis_1pa_inj:13.6e}  "
                    f"{ov_1pa_rec:20.15f}  {mis_1pa_rec:13.6e}\n")

        except Exception as e:
            line = f"{i:4d}  {a:7.4f}  {e0:7.4f}  {p0:8.5f}  ERROR: {e}\n"
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

        log.write(line); log.flush()
        print(line, end="")

        # Save partial results after each point
        np.save(OUT_INJ, out_inj)
        np.save(REC_FILE, rec_out)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
with open(OUT_TXT, "a") as log:
    for col, tag in [(0, "0PA inj"), (1, "1PA inj")]:
        vals = out_inj[:, col]
        v    = vals[~np.isnan(vals)]
        if v.size:
            s = (f"\n{tag} mismatch: min={v.min():.4e}  max={v.max():.4e}  "
                 f"median={np.median(v):.4e}  n={v.size}")
            log.write(s + "\n"); print(s)
    final_ov = rec_out[:, RCOL['final_overlap']]
    v = final_ov[~np.isnan(final_ov)]
    if v.size:
        s = (f"\n1PA rec overlap: min={v.min():.10f}  max={v.max():.10f}  "
             f"median={np.median(v):.10f}")
        log.write(s + "\n"); print(s)

print(f"\n[INFO] Saved injection overlaps: {OUT_INJ}  shape={out_inj.shape}")
print(f"[INFO] Updated recovered array:  {REC_FILE}")
print(f"[INFO] Log: {OUT_TXT}")
