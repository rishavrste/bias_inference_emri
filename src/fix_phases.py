"""
Wrap Phi_phi0 and Phi_r0 to [0, 2*pi] in all starting_point_*.npy files
and in the global result_parameter_array npy for IMRI_TAIL 0pa runs.

Run after jobs complete to ensure bias calculations use wrapped phases.
"""
import numpy as np
import glob
import os

TWO_PI = 2 * np.pi
RESULTS_BASE = "/scratch/josh.mat/opt_grid/results/IMRI_TAIL_0pa"
GLOBAL_RESULT = "/home/svu/josh.mat/git_repos/bias_inference_emri/data/result_parameter_array_IMRI_TAIL_0pa.npy"

def wrap(val):
    return float(val) % TWO_PI

fixed_count = 0

# --- Fix per-point starting_point files ---
for f in sorted(glob.glob(os.path.join(RESULTS_BASE, "**/starting_point_*.npy"), recursive=True)):
    d = np.load(f, allow_pickle=True).item()
    phi0 = float(d['Phi_phi0'])
    phir0 = float(d['Phi_r0'])
    wrapped0 = wrap(phi0)
    wrappedr = wrap(phir0)
    if abs(phi0 - wrapped0) > 1e-9 or abs(phir0 - wrappedr) > 1e-9:
        d['Phi_phi0'] = wrapped0
        d['Phi_r0']   = wrappedr
        np.save(f, d)
        print(f"[FIXED] {f}")
        print(f"  Phi_phi0: {phi0:.6f} -> {wrapped0:.6f}")
        print(f"  Phi_r0:   {phir0:.6f} -> {wrappedr:.6f}")
        fixed_count += 1
    else:
        print(f"[OK]    {f}  (Phi_phi0={phi0:.4f}, Phi_r0={phir0:.4f})")

# --- Fix global result array ---
# Row layout: m1,m2,a,p0,e0,xI0,dist,qS,phiS,qK,phiK,Phi_phi0,Phi_theta0,Phi_r0,...
PHI_PHI0_IDX = 11
PHI_R0_IDX   = 13
arr = np.load(GLOBAL_RESULT, allow_pickle=True)
global_fixed = 0
for i in range(len(arr)):
    phi0  = float(arr[i, PHI_PHI0_IDX])
    phir0 = float(arr[i, PHI_R0_IDX])
    wrapped0 = wrap(phi0)
    wrappedr = wrap(phir0)
    if abs(phi0 - wrapped0) > 1e-9 or abs(phir0 - wrappedr) > 1e-9:
        arr[i, PHI_PHI0_IDX] = wrapped0
        arr[i, PHI_R0_IDX]   = wrappedr
        print(f"[FIXED] global result pt {i}: Phi_phi0 {phi0:.6f}->{wrapped0:.6f}, Phi_r0 {phir0:.6f}->{wrappedr:.6f}")
        global_fixed += 1

if global_fixed:
    np.save(GLOBAL_RESULT, arr)
    print(f"[SAVED] global result array ({global_fixed} points updated)")
else:
    print("[OK]    global result array — no unwrapped phases found")

print(f"\nDone. {fixed_count} starting_point files fixed, {global_fixed} global result entries fixed.")
