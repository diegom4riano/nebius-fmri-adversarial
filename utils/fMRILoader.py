import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

WINDOW = 50   # TRs per window (original STAGIN)
STRIDE = 23   # stride: (1200-50)//23 ≈ 50 windows per subject, matching ABIDE count in paper


def _make_windows(roi_ts):
    """
    roi_ts: (N_rois, T)
    Returns:
        v:         (T_w, N, N)  — FC matrices per window
        a:         (T_w, N, N)  — copy of v for sparsification
        endpoints: list[int]    — 1-indexed window end positions
    """
    N, T = roi_ts.shape
    starts = list(range(0, T - WINDOW + 1, STRIDE))
    windows = []
    for start in starts:
        seg = roi_ts[:, start:start + WINDOW]   # (N, WINDOW)
        fc  = np.corrcoef(seg).astype(np.float32)
        windows.append(np.nan_to_num(fc, nan=0.0))
    v = np.stack(windows, axis=0)              # (T_w, N, N)
    endpoints = [s + WINDOW for s in starts]
    return v, v.copy(), endpoints


class HCPDataset(Dataset):
    def __init__(self, roi_ts_path, labels_path, indices=None, fc_dir=None):
        # Memory-map the large array so workers share OS page cache
        all_data = np.load(roi_ts_path, mmap_mode='r')  # (N_subj, N_rois, T)
        all_labels = np.load(labels_path)                # (N_subj,)

        if indices is not None:
            self.orig_indices = np.array(indices)
        else:
            self.orig_indices = np.arange(len(all_labels))

        self.data   = all_data    # mmap — no per-worker duplication
        self.labels = all_labels[self.orig_indices]
        self.fc_dir = fc_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        orig_idx = int(self.orig_indices[idx])
        roi_ts   = np.array(self.data[orig_idx], dtype=np.float32)  # (N_rois, T)
        label    = int(self.labels[idx])

        # Load precomputed FC if available, else compute on the fly
        fc_path = None if self.fc_dir is None else \
                  os.path.join(self.fc_dir, f"fc_{orig_idx:04d}.npy")

        if fc_path and os.path.exists(fc_path):
            v = np.load(fc_path)                         # (T_w, N, N)
            a = v.copy()
            endpoints = [WINDOW + i * STRIDE for i in range(v.shape[0])]
        else:
            v, a, endpoints = _make_windows(roi_ts)

        t = roi_ts.T  # (T, N_rois) — raw time series for STAGIN temporal encoder

        return {
            'v':         torch.from_numpy(v),
            'a':         torch.from_numpy(a),
            't':         torch.from_numpy(t),
            'endpoints': endpoints,
            'label':     label,
        }


def hcp_collate(batch):
    """Truncates all samples to the minimum window/time count in the batch."""
    min_tw = min(b['v'].shape[0] for b in batch)
    min_t  = min(b['t'].shape[0] for b in batch)

    v_list, a_list, t_list, labels = [], [], [], []
    endpoints = None

    for b in batch:
        v_list.append(b['v'][:min_tw])
        a_list.append(b['a'][:min_tw])
        t_list.append(b['t'][:min_t])
        labels.append(b['label'])
        if endpoints is None:
            endpoints = b['endpoints'][:min_tw]

    return (
        torch.stack(v_list, dim=0),                        # (B, T_w, N, N)
        torch.stack(a_list, dim=0),                        # (B, T_w, N, N)
        torch.stack(t_list, dim=1),                        # (T, B, N) — seq-first for GRU
        endpoints,                                         # list[int]
        torch.tensor(labels, dtype=torch.long),
    )


def make_loaders(roi_ts_path, labels_path, batch_size=4, seed=42):
    labels = np.load(labels_path)
    idx    = np.arange(len(labels))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_val_idx, test_idx = next(sss.split(idx, labels))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
    rel_train, rel_val = next(sss2.split(train_val_idx, labels[train_val_idx]))
    train_idx = train_val_idx[rel_train]
    val_idx   = train_val_idx[rel_val]

    fc_dir = os.path.join(os.path.dirname(roi_ts_path), "fc")
    fc_dir = fc_dir if os.path.isdir(fc_dir) else None

    def _loader(indices, shuffle):
        ds = HCPDataset(roi_ts_path, labels_path, indices=indices, fc_dir=fc_dir)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            collate_fn=hcp_collate,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    return _loader(train_idx, True), _loader(val_idx, False), _loader(test_idx, False)
