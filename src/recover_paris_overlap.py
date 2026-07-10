"""
One-shot GPU script to recover PARIS results after the fix-chi2 chi2 bookkeeping bug.

For each point in RECOVER_POINTS:
  1. Load the PARIS starting_point file (chi2 already corrected to 0.95 externally).
  2. Evaluate calculate_detection_overlap on GPU.
  3. If the overlap beats the global array, update both the result and overlap arrays.

Designed to be submitted as a single-GPU PBS job.
"""

import os, sys, json, datetime
import numpy as np

# --- configuration ------------------------------------------------------------
GRID_TYPE      = os.environ.get("GRID_TYPE",  "IMRI_TAIL")
RUN_TYPE       = os.environ.get("RUN_TYPE",   "1pa_vs_2pa")
RECOVER_POINTS = [int(x) for x in os.environ.get("RECOVER_POINTS", "18,21").split(",")]

REPO_DIR  = "/home/svu/josh.mat/git_repos/bias_inference_emri"
SCRATCH   = "/scratch/josh.mat/opt_grid"
RESULTS   = os.path.join(SCRATCH, "results", f"{GRID_TYPE}_1pa")
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

from misc import calculate_detection_overlap

# --- global arrays ------------------------------------------------------------
result_file  = os.path.join(REPO_DIR, "data",
               f"result_parameter_array_{GRID_TYPE}_1pa.npy")
overlap_file = os.path.join(REPO_DIR, "data",
               f"result_parameter_array_{GRID_TYPE}_1pa_overlaps.npy")

disk_result  = np.load(result_file)
disk_overlap = np.load(overlap_file)

param_cols = ["m1","m2","a","p0","e0","xI0","dist","qS","phiS","qK","phiK",
              "Phi_phi0","Phi_theta0","Phi_r0","dt","T","chi2"]

signal_arr = np.load(os.path.join(REPO_DIR, "data",
             f"signal_parameter_array_{GRID_TYPE}.npy"))

# --- fixed waveform kwargs (same as inference.py) ----------------------------
fixed_kwargs = {
    "dt"       : 10.0,
    "T"        : 0.25,
    "nchannels": 2,
    "use_gpu"  : True,
}
if RUN_TYPE == "1pa_vs_2pa":
    fixed_kwargs["use_1PA"] = True
    fixed_kwargs["evolve_1PA"]     = False
    fixed_kwargs["evolve_primary"] = False
    fixed_kwargs["evolve_2PA"]     = False

for pt in RECOVER_POINTS:
    print(f"\n{'='*60}")
    print(f"Recovering pt{pt} ...")

    sig = {k: signal_arr[pt, i] for i, k in enumerate(param_cols)}

    sp_path = (f"{RESULTS}/{GRID_TYPE}_{pt}"
               "/paris_optimal_snr_id_0/starting_point_1.npy")
    if not os.path.exists(sp_path):
        print(f"  [SKIP] starting_point file not found: {sp_path}")
        continue

    sp = np.load(sp_path, allow_pickle=True).item()
    print(f"  Loaded starting_point: chi2={sp['chi2']:.4f}  "
          f"Phi_phi0={sp['Phi_phi0']:.6f}  Phi_r0={sp['Phi_r0']:.6f}")

    if abs(sp['chi2'] - 0.95) > 1e-6:
        print(f"  [WARN] chi2={sp['chi2']:.6f} not 0.95 — proceeding anyway")

    add_kwargs = {"chi2": float(sp['chi2'])}
    if RUN_TYPE == "1pa_vs_2pa":
        add_kwargs["evolve_1PA"]     = False
        add_kwargs["evolve_primary"] = False
        add_kwargs["evolve_2PA"]     = False

    try:
        overlap = calculate_detection_overlap(
            float(sp['m1']), float(sp['m2']), float(sp['a']),
            float(sp['p0']), float(sp['e0']), float(sig['xI0']),
            float(sig['dist']), float(sig['qS']), float(sig['phiS']),
            float(sig['qK']), float(sig['phiK']),
            float(sp['Phi_phi0']), float(sig['Phi_theta0']),
            float(sp['Phi_r0']),
            add_kwargs,
            maximize_phase=False,
            **fixed_kwargs,
        )
        overlap = float(overlap)
    except Exception as exc:
        print(f"  [ERROR] calculate_detection_overlap failed: {exc}")
        import traceback; traceback.print_exc()
        continue

    existing = float(disk_overlap[pt])
    print(f"  Evaluated overlap: {overlap:.6f}  (existing global: {existing:.6f})")

    if overlap > existing:
        row = np.array([sp[k] if k in sp else sig[k] for k in param_cols], dtype=float)
        disk_result[pt]  = row
        disk_overlap[pt] = overlap
        np.save(result_file,  disk_result)
        np.save(overlap_file, disk_overlap)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log = {
            "pt": pt, "overlap": overlap, "existing_overlap": existing,
            "params": row.tolist(), "param_names": param_cols, "ts": ts,
        }
        log_path = (f"{RESULTS}/{GRID_TYPE}_{pt}"
                    f"/recovered_overlap{overlap:.6f}_{ts}.json")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"  [SAVE] Global array updated. Log → {log_path}")
    else:
        print(f"  [SKIP] Not an improvement; global array unchanged.")

print(f"\n{'='*60}")
print("Recovery complete.")
final_overlaps = np.load(overlap_file)
for pt in RECOVER_POINTS:
    print(f"  pt{pt}: {final_overlaps[pt]:.6f}")
