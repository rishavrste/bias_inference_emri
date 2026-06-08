"""
Plot the residual (post-bias-correction) 1PA-vs-2PA mismatch across the
(a, e0) grid -- a smoothness check on the refined optimization results.

For each grid point, the best-fit 1PA template (found by PARIS+DE+NM) is
compared to the 2PA injection; the residual mismatch = 1 - final_overlap
is the part of the systematic 1PA/2PA difference that bias alone cannot
absorb. A smooth trend across (a, e0) is evidence the optimization has
genuinely converged rather than landing in scattered local optima.

Usage:
  python plot_mismatch_grid.py
"""
import glob
import json
import os
import re

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams.update({
    'font.family':        'serif',
    'mathtext.fontset':   'cm',      # Computer Modern math, no latex binary needed
    'font.size':          8,
    'axes.labelsize':     8,
    'axes.titlesize':     8,
    'xtick.labelsize':    7,
    'ytick.labelsize':    7,
    'legend.fontsize':    7,
    'figure.dpi':         300,
    'text.usetex':        False,
})

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir   = '/scratch/josh.mat/opt_grid/results/EMRI_1pa'

# Grid layout: idx = ie * Na + ia  (a fastest-varying, matching overlap_1pa_vs_2pa.txt)
Na, Ne = 5, 5
a_vals  = np.array([-0.9, -0.5, 0.0, 0.5, 0.9])
e0_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
n_pts   = Na * Ne

_id_re = re.compile(r'id_(\d+)/opt_refined')


def _latest_refined_json(idx):
    """Return the path to the highest-id, most-recent refined-result JSON for grid point idx."""
    matches = glob.glob(os.path.join(
        base_dir, f'EMRI_{idx}', 'paris_optimal_snr_id_*', 'opt_refined_optimal_snr_id_*.json'))
    if not matches:
        return None
    max_id = max(int(_id_re.search(p).group(1)) for p in matches)
    cands  = [p for p in matches if int(_id_re.search(p).group(1)) == max_id]
    return max(cands, key=os.path.getmtime)


# ── Load final mismatch = 1 - final_overlap for every grid point ─────────────
mismatch = np.full(n_pts, np.nan)
for idx in range(n_pts):
    path = _latest_refined_json(idx)
    if path is None:
        print(f'idx {idx}: no refined result found -- skipping')
        continue
    with open(path) as fh:
        d = json.load(fh)
    mismatch[idx] = 1.0 - d['results']['final_overlap']

# Reshape to (Na, Ne): M[ia, ie] = mismatch at (a_vals[ia], e0_vals[ie])
M = mismatch.reshape(Ne, Na).T
logM = np.log10(M)

# ── PRL single-column figure ─────────────────────────────────────────────────
col_width = 3.375
fig, ax = plt.subplots(figsize=(col_width, 2.6))

x_ticks = np.arange(0.1, 0.51, 0.1)
ax.set_xticks(x_ticks)
ax.set_yticks(a_vals)
ax.tick_params(axis='both', labelsize=7, length=2, pad=2)

vmin, vmax = np.nanmin(logM), np.nanmax(logM)
im = ax.pcolormesh(e0_vals, a_vals, logM, cmap='viridis', shading='auto',
                   vmin=vmin, vmax=vmax)

for ia, a in enumerate(a_vals):
    for ie, e0 in enumerate(e0_vals):
        v = logM[ia, ie]
        if not np.isnan(v):
            ax.text(e0, a, f'{v:.1f}', ha='center', va='center', fontsize=6,
                    color='white' if v < vmin + 0.45 * (vmax - vmin) else 'black')

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=6, length=2, pad=1)
cbar.ax.set_title(r'$\log_{10}\mathcal{M}$', fontsize=7, pad=3)

ax.set_xlabel(r'$e_0$', fontsize=7)
ax.set_ylabel(r'$\chi_1$', fontsize=7)
ax.set_title('Residual 1PA/2PA mismatch after bias optimization', fontsize=7.5, pad=4)

fig.tight_layout()

out_path = os.path.join(script_dir, 'mismatch_grid.pdf')
fig.savefig(out_path, bbox_inches='tight')
print(f'Figure saved: {out_path}')

print('\nGrid of log10(mismatch)  [rows = a, cols = e0]:')
print('       ' + '  '.join(f'e0={e:.1f}' for e in e0_vals))
for ia, a in enumerate(a_vals):
    print(f'a={a:+.1f}  ' + '   '.join(f'{logM[ia, ie]:6.2f}' for ie in range(Ne)))
