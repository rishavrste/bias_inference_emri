"""
Sync result_parameter_array_IMRI_TAIL_with_phase_1PA.npy from the verified
best-fit npy files scattered across phase directories.

The central result file gets stale entries when concurrent PBS jobs each load
their own in-memory copy and clobber each other's writes for cases outside
their own index range. This rebuilds the file from scratch using the same
best-overlap selection logic as check_overlaps.py (highest final_overlap JSON
per case, across all phases).
"""

import os, glob, json
import numpy as np

SIGNAL_FILE = "/scratch/e1583490/SuperKludege_Optimizations/opt_grid/signal_parameter_array_IMRI_TAIL.npy"
RESULT_FILE = "/scratch/e1583490/SuperKludege_Optimizations/opt_grid/result_parameter_array_IMRI_TAIL_with_phase_1PA.npy"

PHASE_DIRS = [
    "/scratch/e1583490/SuperKludege_Optimizations/IMRI_TAIL_1PA_with_Phase_15/",
    "/scratch/e1583490/SuperKludege_Optimizations/IMRI_TAIL_1PA_with_Phase_14/",
    "/scratch/e1583490/SuperKludege_Optimizations/IMRI_TAIL_1PA_with_Phase_13/",
    "/scratch/e1583490/SuperKludege_Optimizations/IMRI_TAIL_1PA_with_Phase_12/",
    "/scratch/e1583490/SuperKludege_Optimizations/IMRI_TAIL_1PA_with_Phase_11/",
    "/scratch/e1583490/SuperKludege_Optimizations/IMRI_TAIL_1PA_with_Phase_9/",
    "/scratch/e1583490/SuperKludege_Optimizations/IMRI_TAIL_1PA_with_Phase_8/",
    "/scratch/e1583490/SuperKludege_Optimizations/IMRI_TAIL_1PA_with_Phase_6/",
    "/scratch/e1583490/SuperKludege_Optimizations/IMRI_TAIL_1PA_with_Phase_3/",
]

COLUMN_ORDER = ["m1","m2","a","p0","e0","xI0","dist","qS","phiS","qK","phiK",
                "Phi_phi0","Phi_theta0","Phi_r0","dt","T","chi2"]


def find_best_npy(case_idx):
    best_ov = -np.inf
    best_params = None
    for d in PHASE_DIRS:
        case_dir = os.path.join(d, f"IMRI_TAIL_1PA_{case_idx}")
        if not os.path.isdir(case_dir):
            continue
        for jf in glob.glob(os.path.join(case_dir, "**", "*.json"), recursive=True):
            try:
                with open(jf) as fh:
                    jdata = json.load(fh)
                ov = float(jdata.get('results', {}).get('final_overlap', -1))
                if ov <= best_ov:
                    continue
                result_dir = os.path.dirname(jf)
                npy_files = (glob.glob(os.path.join(result_dir, "results_differential_evolution_*.npy")) +
                             glob.glob(os.path.join(result_dir, "results_paris_*.npy")) +
                             glob.glob(os.path.join(result_dir, "results_nelder_mead_*.npy")))
                if not npy_files:
                    continue
                npy = max(npy_files, key=os.path.getmtime)
                params = np.load(npy, allow_pickle=True).item()
                best_ov = ov
                best_params = params
            except Exception:
                pass
    return best_ov, best_params


def main():
    sig_arr = np.load(SIGNAL_FILE)
    n = sig_arr.shape[0]
    result_arr = sig_arr.copy()

    print(f"{'Case':>5}  {'best overlap':>14}  {'status'}")
    print("-" * 40)
    for i in range(n):
        ov, params = find_best_npy(i)
        if params is None:
            print(f"{i:>5}  {'NO DATA':>14}  keeping truth row")
            continue
        row = [float(params[k]) for k in COLUMN_ORDER]
        result_arr[i] = row
        print(f"{i:>5}  {ov:>14.10f}  synced")

    np.save(RESULT_FILE, result_arr)
    print(f"\nSaved synced result array to {RESULT_FILE}")


if __name__ == "__main__":
    main()
