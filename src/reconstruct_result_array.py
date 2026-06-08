"""
Reconstruct result_parameter_array_EMRI_1pa.npy from per-point output files.

For each grid point i, finds the most recent results_refined_*.npy (DE+NM best)
or falls back to results_paris_*.npy (PARIS/checkpoint best) in the per-point
subdirectory.  Assembles them into the global (25, 17) result array.

Usage:
    python reconstruct_result_array.py [--dry-run]
"""
import os, glob, re, argparse
import numpy as np

PARAM_FILE   = "/scratch/josh.mat/bias_inference_emri/data/signal_parameter_array_EMRI.npy"
RESULT_FILE  = "/scratch/josh.mat/bias_inference_emri/data/result_parameter_array_EMRI_1pa.npy"
RESULTS_BASE = "/scratch/josh.mat/opt_grid/results/EMRI_1pa"
COL_ORDER    = ["m1","m2","a","p0","e0","xI0","dist","qS","phiS","qK","phiK",
                "Phi_phi0","Phi_theta0","Phi_r0","dt","T","chi2"]

def latest_file(pattern):
    matches = glob.glob(pattern)
    return max(matches, key=os.path.getmtime) if matches else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be loaded without writing')
    args = parser.parse_args()

    param_array = np.load(PARAM_FILE)
    n_pts = param_array.shape[0]
    result_array = np.zeros_like(param_array)

    for i in range(n_pts):
        pt_dir = os.path.join(RESULTS_BASE, f"EMRI_{i}")
        a, e0 = param_array[i, 2], param_array[i, 4]

        # Prefer the DE+NM refined result; fall back to PARIS/checkpoint
        refined = latest_file(os.path.join(pt_dir, "paris_optimal_snr_id_*",
                                           "results_refined_*.npy"))
        paris   = latest_file(os.path.join(pt_dir, "paris_optimal_snr_id_*",
                                           "results_paris_*.npy"))
        chosen  = refined or paris

        if chosen is None:
            print(f"  [{i:2d}] a={a:+.1f} e0={e0:.1f}  MISSING — row left as zeros")
            continue

        source = "refined" if chosen == refined else "paris  "
        d = np.load(chosen, allow_pickle=True).item()
        row = [float(d[k]) for k in COL_ORDER]
        result_array[i] = row

        delta_m1 = row[0] - param_array[i, 0]
        print(f"  [{i:2d}] a={a:+.1f} e0={e0:.1f}  [{source}]  "
              f"Δm1={delta_m1:+.4e}  file={os.path.basename(chosen)}")

    if not args.dry_run:
        np.save(RESULT_FILE, result_array)
        print(f"\nSaved → {RESULT_FILE}")
    else:
        print("\n[dry-run] Not written.")

if __name__ == "__main__":
    main()
