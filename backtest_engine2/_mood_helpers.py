import numpy as np
from numba import njit, prange
import time


@njit(cache=True, fastmath=True)
def _mood_d_max(buf, n):
    """Mood scale-shift D_max on buf[0:n]. Equations (3)(4) from Ross (2011)."""
    if n < 3: return 0.0
    view = buf[:n]
    sorted_idx = np.argsort(view)
    ranks = np.empty(n, dtype=np.float64)
    for i in range(n): ranks[sorted_idx[i]] = float(i + 1)
    n_f = float(n); cum = 0.0; dmax = 0.0
    for j in range(n - 1):
        cum += (ranks[j] - (n_f + 1.0) * 0.5) ** 2
        k = j + 1
        if k < 2: continue
        k_f = float(k)
        mu  = k_f * (n_f * n_f - 1.0) / 12.0
        var = k_f * (n_f - k_f) * (n_f + 1.0) * (n_f * n_f - 4.0) / 180.0
        if var <= 0.0: continue
        m = abs(cum - mu) / var ** 0.5
        if m > dmax: dmax = m
    return dmax


@njit(parallel=True, cache=True, fastmath=True)
def _simulate_batch(sequences, h, warmup):
    n_sim, max_n = sequences.shape
    run_lengths = np.empty(n_sim, dtype=np.float64)
    for i in prange(n_sim):
        row = sequences[i, :]           # full row extracted once — stable 'A' type
        fired = False
        for t in range(warmup + 1, max_n + 1):
            if _mood_d_max(row, t) > h:  # no variable-length slice passed cross-JIT
                run_lengths[i] = float(t); fired = True; break
        if not fired: run_lengths[i] = float(max_n)
    return run_lengths


@njit(cache=True, fastmath=True)
def _mood_d_max_tau(buf, n):
    """Like _mood_d_max but also returns the split position tau (1-indexed)."""
    if n < 3: return 0.0, 0.0
    view = buf[:n]
    sorted_idx = np.argsort(view)
    ranks = np.empty(n, dtype=np.float64)
    for i in range(n): ranks[sorted_idx[i]] = float(i + 1)
    n_f = float(n); cum = 0.0; dmax = 0.0; best_k = 0.0
    for j in range(n - 1):
        cum += (ranks[j] - (n_f + 1.0) * 0.5) ** 2
        k = j + 1
        if k < 2: continue
        k_f = float(k)
        mu  = k_f * (n_f * n_f - 1.0) / 12.0
        var = k_f * (n_f - k_f) * (n_f + 1.0) * (n_f * n_f - 4.0) / 180.0
        if var <= 0.0: continue
        m = abs(cum - mu) / var ** 0.5
        if m > dmax: dmax = m; best_k = k_f
    return dmax, best_k


def _warmup_jit():
    dummy = np.random.default_rng(0).standard_normal(50).astype(np.float64)
    _mood_d_max(dummy, 50)
    _mood_d_max_tau(dummy, 50)
    tiny = np.random.default_rng(0).standard_normal((4, 50)).astype(np.float64)
    _simulate_batch(tiny, 4.0, 21)


def calibrate_mood_threshold(
    arl: int = 10_000,
    n_sim: int = 500,
    max_n: int = 30_000,
    warmup: int = 21,
    seed: int = 42,
    n_workers: int = None,
) -> float:
    print("JIT warm-up ...", end=" ", flush=True)
    _warmup_jit()
    print("done", flush=True)

    rng_seed = np.random.default_rng(seed)
    iter_seeds = rng_seed.integers(0, 2**31, size=(12, 1))

    lo, hi = 3.5, 6.0
    print(f"\nCalibrating  ARL={arl:,}  n_sim={n_sim}  max_n={max_n:,}", flush=True)
    print(f"{'iter':>4}  {'h':>7}  {'est_ARL':>10}  {'time':>6}", flush=True)

    for i in range(12):
        mid = (lo + hi) / 2.0
        t0 = time.perf_counter()
        rng = np.random.default_rng(int(iter_seeds[i, 0]))
        seqs = rng.standard_normal((n_sim, max_n)).astype(np.float64)
        rl = _simulate_batch(seqs, mid, warmup)
        est = float(np.mean(rl))
        print(f"{i+1:>4}  {mid:>7.4f}  {est:>10.0f}  {time.perf_counter()-t0:>5.1f}s", flush=True)
        if est < arl: lo = mid
        else: hi = mid

    h = (lo + hi) / 2.0
    print(f"\nMOOD_THRESHOLD = {h:.6f}", flush=True)
    return h
