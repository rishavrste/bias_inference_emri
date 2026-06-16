# bias-inference-emri

Bias-inference tooling for LISA EMRI/IMRI waveforms: given a "truth" waveform
generated at a higher post-adiabatic order (e.g. 2PA), find the best-fit
parameters of a lower-order template (e.g. 1PA, or non-spinning 0PA) that
maximizes the noise-weighted overlap against that truth. This is used to
quantify systematic bias from using an approximate waveform model in LISA
parameter estimation.

The current branch implements the **1PA-vs-2PA** (and **0PA-vs-2PA**)
overlap-maximization pipeline: a Fisher-matrix-guided Nelder-Mead /
Differential-Evolution / PARIS optimizer search, run independently per
parameter-grid case, that pushes the overlap between the template and truth
waveform as close to 1 as possible.

## How it works

1. **Truth waveform.** For each case in a signal-parameter grid, build the
   fiducial (2PA) waveform through the LISA response (`SuperKludgeWaveform`
   + `fastlisaresponse.ResponseWrapper`, `EqualArmlengthOrbits` orbits), its
   PSD, and its windowed FFT.
2. **Fisher prior.** Compute a local Fisher-matrix "parallelotope" prior
   around the truth point (`compute_fisher_parallelotope` in `misc.py`),
   using `stableemrifisher`. This gives per-parameter 1σ widths used to set
   physically-motivated search bounds.
3. **Optimize.** Maximize overlap (or another supported objective — optimal
   SNR, time-max correlation, chi-square match) over the template
   parameters using one of:
   - `differential_evolution` — Fisher-bounded global search, optionally
     preceded by a short Nelder-Mead pre-polish (the default, most-tested
     path for this campaign);
   - `nelder-mead` — local simplex search from a starting point;
   - `paris` — Fisher-ellipse-seeded PARIS sampler (`parismc`) with a
     post-hoc local Gaussian polish and optional DE/NM refinement stage.
4. **Persist.** Each case writes its best-fit parameters, an `opt_*.json`
   summary (config + results), and the resulting overlap into its own
   per-case output directory under `Config.basedir`.

Both `run_type` (`0pa_vs_2pa` vs `1pa_vs_2pa`) and `parameter_selected`
(`intrinsic` vs `intrinsic_phase`, i.e. whether `Phi_phi0`/`Phi_r0` are also
inferred) are supported; together they select a 5/6/7/8-dimensional search
space, encoded once in `inference.py` as `PARAM_KEYS_BY_NDIM`.

## Repo layout

```
src/
  inference.py        Main optimization pipeline (entry point: `python inference.py`)
  misc.py              Waveform/PSD/overlap helper functions, Fisher-prior construction
  config_paris.py      `Config` class — all run parameters (paths, optimizer, priors, ...)
  batch.sh             Example PBS batch script (NUS Hopper cluster, see below)
  tests/               Diagnostic notebooks + standalone verification scripts
    check_overlaps.py     Recompute overlaps from a result file independently of inference.py
    sync_result_array.py  Rebuild a result file from per-case phase-directory outputs
    *.ipynb                Exploratory/diagnostic notebooks (Fisher checks, PSD checks, ...)
  requirement.txt      Snapshot of a working `pip list` for this project (see Dependencies)
```

## Dependencies

This project depends on FastEMRIWaveforms (`few`), `fastlisaresponse`,
`lisaanalysistools` / `lisatools`, `stableemrifisher`, and `parismc`, plus a
CUDA-enabled `cupy` for GPU acceleration. The authoritative dependency list
is `pyproject.toml` / `uv.lock` at the repo root (managed with
[`uv`](https://docs.astral.sh/uv/)); `src/requirement.txt` is a frozen
`pip list` snapshot of a known-working environment, kept for reference when
debugging version mismatches — it is not a `pip install -r` manifest.

```bash
uv sync   # from the repo root, using pyproject.toml / uv.lock
```

`parismc` is only required if you intend to use `optimizer = 'paris'`.

## Configuration

All run parameters live in `src/config_paris.py`'s `Config` class:
data paths, `run_type` / `parameter_selected`, which optimizer to use
(`differential_evolution` / `nelder-mead` / `paris`), Fisher prior widths
(`prior_sigma_range`), DE/NM/PARIS iteration budgets, and the per-case
output directory (`basedir`).

The data paths and output directory default to this project's own campaign
layout but can be overridden without editing the file, via environment
variables:

| Env var               | Overrides                                          |
|------------------------|-----------------------------------------------------|
| `OPT_PARAM_FILE`        | `Config.param_file` (signal/truth parameter grid)   |
| `OPT_RESULT_FILE`       | `Config.result_file` (best-fit result array)        |
| `OPT_CASE_PREFIX`       | `Config.TYPE` (per-case output directory prefix)    |
| `OPT_BASEDIR`           | `Config.basedir` (this run's output root)            |
| `OPT_PREV_BASEDIRS`     | `Config.prev_basedir` (comma-separated list of prior-run output roots to check for warm-start/skip) |
| `START_INDEX`, `END_INDEX` | which case indices (rows of the parameter grid) to process |

## Running a job

`inference.py`'s `__main__` block loops over `Config.start_index` to
`Config.end_index`, and for each case:
- skips it if a previous run (one of `Config.prev_basedir`) already reached
  `Config.good_overlap_threshold`;
- otherwise warm-starts from the best previous result (if above
  `Config.warm_start_threshold`) or starts fresh from the 2PA truth, then
  runs the configured optimizer and writes the result back into
  `Config.result_file`.

```bash
cd src
START_INDEX=0 END_INDEX=5 python inference.py
```

### Submitting on a cluster

`batch.sh` is an example PBS submission script **written for the NUS HPC
"Hopper" cluster** — it assumes the PBS scheduler, `module load
singularity`, and the CUDA 12.4 singularity image shown in the script.
It will need adapting (PBS directives, container path, module commands) to
run on any other cluster. Customize via env vars or by editing the script:

```bash
# replace #PBS -P <YOUR_PROJECT_CODE> with your own PBS allocation first
START_INDEX=0 END_INDEX=5 SCRIPT=inference.py qsub batch.sh
```

To run the overlap-verification script as a batch job instead, set
`SCRIPT=tests/check_overlaps.py`.

## Verifying results

- `python tests/check_overlaps.py` independently recomputes, on the GPU,
  the 2PA-truth-vs-1PA-best-fit overlap for every case in a result file —
  use this to sanity-check `Config.result_file` without trusting the
  overlap values `inference.py` already wrote.
- `python tests/sync_result_array.py` rebuilds a result file from scratch
  by scanning every case's output directory (across one or more phase
  directories) for the JSON with the highest recorded `final_overlap`, and
  taking its sibling `.npy` parameters. This recovers from the case where
  several concurrent jobs, each holding their own in-memory copy of the
  result array, clobber each other's writes for cases outside their own
  index range — symptom: a stale/wrong row in `Config.result_file` despite
  a good per-case JSON existing on disk.

Both scripts hardcode this project's default `OPT_PARAM_FILE` /
`OPT_RESULT_FILE` paths but accept CLI args / honor the same env vars to
point at a different grid.

## Diagnostic notebooks

`src/tests/*.ipynb` contains exploratory and diagnostic notebooks (Fisher
matrix checks, PSD/noise-model checks, TDI response checks, early
parameter-optimization experiments). They are not part of the production
pipeline and are not required to run `inference.py`.
