"""
One-time precomputation of sliding-window FC matrices for all subjects.
Saves (T_w, N, N) float32 arrays per subject to fc_dir.
With 16 workers on the VM this takes ~1-2 minutes for 1080 subjects.
"""
import argparse
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

WINDOW = 50
STRIDE = 23


def _compute_and_save(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    idx, roi_ts, out_path = args
    if os.path.exists(out_path):
        return idx, "skip"
    N, T = roi_ts.shape
    starts = list(range(0, T - WINDOW + 1, STRIDE))
    windows = []
    for start in starts:
        seg = roi_ts[:, start:start + WINDOW]
        fc = np.corrcoef(seg).astype(np.float32)
        windows.append(np.nan_to_num(fc, nan=0.0))
    v = np.stack(windows, axis=0)  # (T_w, N, N)
    np.save(out_path, v)
    return idx, v.shape


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roi-ts",  default="data/fmri/hcp/roi/roi_timeseries.npy")
    p.add_argument("--out-dir", default="data/fmri/hcp/roi/fc")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = np.load(args.roi_ts)  # (N, 333, 1200)
    N = data.shape[0]

    worker_args = [
        (i, data[i], os.path.join(args.out_dir, f"fc_{i:04d}.npy"))
        for i in range(N)
    ]

    print(f"Precomputando FC para {N} sujeitos "
          f"(WINDOW={WINDOW}, STRIDE={STRIDE}, T_w={(1200-WINDOW)//STRIDE}) "
          f"com {args.workers} workers...")
    sys.stdout.flush()

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for future in as_completed(ex.submit(_compute_and_save, a) for a in worker_args):
            idx, result = future.result()
            done += 1
            if done % 100 == 0 or done == N:
                print(f"  {done}/{N}  idx={idx}  shape={result}")
                sys.stdout.flush()

    # Quick sanity check
    sample = np.load(os.path.join(args.out_dir, "fc_0000.npy"))
    print(f"\nOK — fc_0000.npy shape: {sample.shape}  dtype: {sample.dtype}")
    total_gb = N * sample.nbytes / 1e9
    print(f"Total em disco estimado: {total_gb:.1f} GB")


if __name__ == "__main__":
    main()
