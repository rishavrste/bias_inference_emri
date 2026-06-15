---
name: feedback-de-workers
description: FEW waveform parallelism constraints for scipy DE
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b3132178-fa52-41c3-b128-9eed71c247a3
---

FEW CUDA waveforms (`interpolate.cu`) are NOT thread-safe and NOT fork-safe with Python 3.13 multiprocessing. Always use `de_workers=1` (serial DE).

**Why:**
- `multiprocessing.Pool` with workers=-1: fails because closures inside `main()` cannot be pickled in Python 3.13
- `ThreadPoolExecutor`: causes `GPUassert: an illegal memory access` in `interpolate.cu:771`
- Serial (workers=1) works correctly at ~1.6s/waveform on dedicated GPU

**How to apply:** When modifying DE configuration, never set `de_workers` to anything other than 1 for FEW-based objectives. The `workers` parameter is kept in the function signature for potential future use with non-CUDA objectives.
