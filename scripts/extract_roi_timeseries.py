import argparse
import csv
import os
import subprocess
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

ATLAS_PATH = "HCP_S1200_Atlas_Z4_pkXDZ/Gordon333.32k_fs_LR.dlabel.nii"
S3_TEMPLATE = (
    "s3://hcp-openaccess/HCP_1200/{subj}/MNINonLinear/Results/"
    "rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii"
)
N_WORKERS = 8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject-list", required=True)
    p.add_argument("--labels-csv",   required=True)
    p.add_argument("--out-dir",      required=True)
    p.add_argument("--workers",      type=int, default=N_WORKERS)
    return p.parse_args()


def load_labels(csv_path):
    labels = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            labels[row["Subject"]] = 1 if row["Gender"] == "M" else 0
    return labels


def load_atlas(atlas_path):
    import nibabel as nib
    img = nib.load(atlas_path)
    return np.asarray(img.dataobj, dtype=np.int32).squeeze()


def extract_subject(args):
    """Worker function: downloads, parcellates, saves. Returns (subj, shape) or raises."""
    subj, out_dir, atlas_path = args
    import nibabel as nib

    atlas_labels = load_atlas(atlas_path)

    tmp_path = f"/tmp/{subj}.dtseries.nii"
    s3_path = S3_TEMPLATE.format(subj=subj)

    subprocess.run(
        ["aws", "s3", "cp", s3_path, tmp_path, "--profile", "hcp"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    ts_img = nib.load(tmp_path)
    # dtseries CIFTI-2: dataobj shape (T, grayordinates)
    ts_data = np.asarray(ts_img.dataobj, dtype=np.float32)
    os.remove(tmp_path)

    parcels = np.unique(atlas_labels[atlas_labels > 0])
    n_parcels = len(parcels)
    T = ts_data.shape[0]

    roi_ts = np.zeros((n_parcels, T), dtype=np.float32)
    for i, p in enumerate(parcels):
        mask = atlas_labels == p
        if mask.sum() > 0:
            roi_ts[i] = ts_data[:, mask].mean(axis=1)

    # z-score per ROI over time
    std = roi_ts.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    roi_ts = (roi_ts - roi_ts.mean(axis=1, keepdims=True)) / std

    out_path = os.path.join(out_dir, f"{subj}.npy")
    np.save(out_path, roi_ts)  # (333, T)
    return subj, roi_ts.shape


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.subject_list) as f:
        subjects = [l.strip() for l in f if l.strip()]

    labels = load_labels(args.labels_csv)

    # Filter subjects not yet processed
    pending = [s for s in subjects
               if not os.path.exists(os.path.join(args.out_dir, f"{s}.npy"))]
    already = len(subjects) - len(pending)
    if already:
        print(f"{already} sujeitos já processados, pulando.")

    total = len(subjects)
    done_count = already

    worker_args = [(s, args.out_dir, ATLAS_PATH) for s in pending]

    print(f"Iniciando {len(pending)} sujeitos com {args.workers} workers paralelos...")
    sys.stdout.flush()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_subject, a): a[0] for a in worker_args}
        for future in as_completed(futures):
            subj = futures[future]
            try:
                _, shape = future.result()
                done_count += 1
                print(f"[{done_count}/{total}] {subj} → {shape}")
            except Exception as e:
                print(f"ERRO {subj}: {e}", file=sys.stderr)
            sys.stdout.flush()

    # Stack all processed subjects
    done = [s for s in subjects if os.path.exists(os.path.join(args.out_dir, f"{s}.npy"))]
    if not done:
        print("Nenhum sujeito processado com sucesso.", file=sys.stderr)
        sys.exit(1)

    print(f"\nEmpilhando {len(done)} sujeitos...")
    arrays, label_list = [], []
    for subj in done:
        arrays.append(np.load(os.path.join(args.out_dir, f"{subj}.npy")))
        label_list.append(labels[subj])

    roi_ts = np.stack(arrays, axis=0)   # (N, 333, T)
    label_arr = np.array(label_list, dtype=np.int64)

    np.save(os.path.join(args.out_dir, "roi_timeseries.npy"), roi_ts)
    np.save(os.path.join(args.out_dir, "labels.npy"), label_arr)
    print(f"Salvo: roi_timeseries.npy {roi_ts.shape}, labels.npy {label_arr.shape}")
    print(f"M={label_arr.sum()}  F={(label_arr==0).sum()}")


if __name__ == "__main__":
    main()
