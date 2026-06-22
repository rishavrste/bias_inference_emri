"""
Check whether rho(lambda_*)/rho_true ≈ overlap(lambda_*) for the 0PA IMRI_TAIL best-fit points.

If the two are equal at the best-fit, maximising the overlap is self-consistent with
maximising the likelihood (Chua & Cutler Eq. 24).  Significant differences indicate
the -½ rho² correction in the likelihood matters and a proper MLE run is needed.

Saves: data/snr_ratio_check_IMRI_TAIL_0pa.npy
  shape (25, 4): [rho_true, rho_template, overlap_recomputed, rho_template/rho_true]
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cupy as cp

from inference import build_waveform_response, prepare_true_waveform, compute_fft_with_windowing
from misc import inner_prod

# ── data ──────────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), '..')
sig_grid = np.load(os.path.join(ROOT, 'data', 'signal_parameter_array_IMRI_TAIL.npy'))
res_0pa  = np.load(os.path.join(ROOT, 'data', 'result_parameter_array_IMRI_TAIL_0pa_overlap_opt.npy'))
ov_0pa   = np.load(os.path.join(ROOT, 'data', 'result_parameter_array_IMRI_TAIL_0pa_overlaps_overlap_opt.npy'))

# Signal col order: m1,m2,a,p0,e0,Y0,dist,qS,phiS,qK,phiK,Phi_phi0,Phi_theta0,Phi_r0,dt,T,chi2
NCHANNELS = 2  # A, E only — T-channel excluded

results = np.zeros((25, 4))  # [rho_true, rho_template, overlap_recomputed, ratio]

for i in range(25):
    sig = sig_grid[i]
    bf  = res_0pa[i]

    m1_s,m2_s,a_s,p0_s,e0_s,Y0,dist,qS,phiS,qK,phiK,phi_phi0_s,phi_theta0_s,phi_r0_s,dt,T,chi2 = sig
    m1_b,m2_b,a_b,p0_b,e0_b = bf[0],bf[1],bf[2],bf[3],bf[4]
    phi_phi0_b,phi_theta0_b,phi_r0_b = bf[11],bf[12],bf[13]

    print(f"\n── pt{i:02d}  a={a_s:.2f}  e0={e0_s:.2f}  stored_overlap={ov_0pa[i]:.6f} ──")

    # 2PA true waveform
    emri_kwargs = {"T": float(T), "dt": float(dt), "chi2": float(chi2),
                   "evolve_1PA": True, "evolve_primary": False, "evolve_2PA": True}
    add_kwargs  = {"chi2": float(chi2), "evolve_1PA": True,
                   "evolve_primary": False, "evolve_2PA": True}

    ctx = prepare_true_waveform(sig[0:14], emri_kwargs, add_kwargs,
                                use_gpu=True, nchannels=NCHANNELS)

    rho_true_sq = float(inner_prod(ctx['waveform_true_fft'], ctx['waveform_true_fft'],
                                   ctx['PSD_funcs'], ctx['delta_f'], xp=cp))
    rho_true = np.sqrt(rho_true_sq)

    # 0PA template at best-fit parameters
    add_kwargs_0pa = {"chi2": float(chi2), "evolve_1PA": False,
                      "evolve_primary": False, "evolve_2PA": False}
    emri_kwargs_0pa = {"T": float(T), "dt": float(dt), "chi2": float(chi2),
                       "evolve_1PA": False, "evolve_primary": False, "evolve_2PA": False}

    wave_params_bf = [m1_b, m2_b, a_b, p0_b, e0_b, float(Y0),
                      float(dist), float(qS), float(phiS), float(qK), float(phiK),
                      phi_phi0_b, phi_theta0_b, phi_r0_b,
                      float(chi2), False, False, False]

    h_bf = cp.array(ctx['waveform_response'](*wave_params_bf, **emri_kwargs_0pa))[0:NCHANNELS, :]
    h_bf_fft = compute_fft_with_windowing(h_bf, float(dt), ctx['N_fiducial'],
                                          use_gpu=True, n_channels=NCHANNELS)

    rho_tmpl_sq = float(inner_prod(h_bf_fft, h_bf_fft,
                                   ctx['PSD_funcs'], ctx['delta_f'], xp=cp))
    rho_tmpl = np.sqrt(rho_tmpl_sq)

    # overlap: <h_true | h_bf> / (rho_true * rho_tmpl)
    num = float(inner_prod(ctx['waveform_true_fft'], h_bf_fft,
                           ctx['PSD_funcs'], ctx['delta_f'], xp=cp))
    overlap = num / (rho_true * rho_tmpl)
    ratio   = rho_tmpl / rho_true

    print(f"  rho_true   = {rho_true:.6f}")
    print(f"  rho_tmpl   = {rho_tmpl:.6f}")
    print(f"  ratio      = {ratio:.6f}")
    print(f"  overlap    = {overlap:.6f}  (stored: {ov_0pa[i]:.6f})")
    print(f"  ratio - overlap = {ratio - overlap:+.6f}")

    results[i] = [rho_true, rho_tmpl, overlap, ratio]

out = os.path.join(ROOT, 'data', 'snr_ratio_check_IMRI_TAIL_0pa.npy')
np.save(out, results)
print(f"\nSaved {out}")
print("\n── Summary ──")
print(f"{'pt':>3} {'a':>6} {'e0':>5} | {'rho_t':>8} {'rho_bf':>8} {'ratio':>8} {'overlap':>9} | ratio-overlap")
print('-'*75)
for i in range(25):
    r = results[i]
    s = sig_grid[i]
    print(f"{i:3d} {s[2]:>6.3f} {s[4]:>5.2f} | {r[0]:>8.4f} {r[1]:>8.4f} {r[3]:>8.6f} {r[2]:>9.6f} | {r[3]-r[2]:+.6f}")
