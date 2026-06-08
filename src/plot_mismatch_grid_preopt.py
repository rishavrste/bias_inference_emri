"""
Plot the pre-optimization (raw injection-vs-template) 1PA-vs-2PA mismatch
across the (a, e0) grid -- a smoothness check on the *un-corrected* systematic
mismatch, for comparison against the post-optimization residual mismatch
(see plot_mismatch_grid.py).

Data source: overlap_1pa_vs_2pa.txt (already computed, no optimization).

Usage:
  python plot_mismatch_grid_preopt.py
"""
import os

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
data_path  = os.path.join(script_dir, 'overlap_1pa_vs_2pa.txt')

# Grid layout: idx = ie * Na + ia  (a fastest-varying, matching overlap_1pa_vs_2pa.txt)
Na, Ne = 5, 5
a_vals  = np.array([-0.9, -0.5, 0.0, 0.5, 0.9])
e0_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
n_pts   = Na * Ne

# ── Load pre-optimization mismatch from the text table ───────────────────────
mismatch = np.full(n_pts, np.nan)
with open(data_path) as fh:
    for line in fh:
        parts = line.split()
        if len(parts) != 9 or not parts[0].isdigit():
            continue
        idx = int(parts[0])
        mismatch[idx] = float(parts[6])

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
ax.set_title('Raw 1PA/2PA mismatch (pre-optimization)', fontsize=7.5, pad=4)

fig.tight_layout()

out_path = os.path.join(script_dir, 'mismatch_grid_preopt.pdf')
fig.savefig(out_path, bbox_inches='tight')
print(f'Figure saved: {out_path}')

print('\nGrid of log10(mismatch)  [rows = a, cols = e0]:')
print('       ' + '  '.join(f'e0={e:.1f}' for e in e0_vals))
for ia, a in enumerate(a_vals):
    print(f'a={a:+.1f}  ' + '   '.join(f'{logM[ia, ie]:6.2f}' for ie in range(Ne)))
