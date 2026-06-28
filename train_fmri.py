import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score

from model.STAGIN import ModelSTAGIN
from utils.fMRILoader import make_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters from original STAGIN paper (egyptdj/stagin, NeurIPS 2021)
DEFAULTS = dict(
    data_dir     = "data/fmri/hcp/roi",
    out_dir      = "saved_model",
    epochs       = 200,
    batch        = 4,
    lr           = 5e-4,
    max_lr       = 1e-3,
    weight_decay = 0.0,
    hidden_dim   = 64,
    num_heads    = 1,
    num_layers   = 4,
    sparsity     = 30,
    dropout      = 0.5,
    readout      = "sero",
    cls_token    = "sum",
    reg_lambda   = 1e-5,
    patience     = 30,
    seed         = 42,
)


def parse_args():
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    return p.parse_args()


def run_epoch(loader, model, optimizer, criterion, device, reg_lambda,
              train=True, scheduler=None):
    model.train(train)
    total_loss, preds_all, labels_all = 0.0, [], []

    for v, a, t, endpoints, labels in loader:
        v, a, t = v.to(device), a.to(device), t.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits, _, _, reg_ortho = model(v, a, t, endpoints)
            loss = criterion(logits, labels) + reg_lambda * reg_ortho

        if train:
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        if torch.isnan(loss):
            continue
        total_loss += loss.item() * len(labels)
        preds_all.extend(logits.argmax(1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

    if not labels_all:
        return float("nan"), 0.5, 0.0
    avg_loss = total_loss / len(loader.dataset)
    bacc = balanced_accuracy_score(labels_all, preds_all)
    f1   = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return avg_loss, bacc, f1


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device: {DEVICE}  |  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    roi_ts_path = os.path.join(args.data_dir, "roi_timeseries.npy")
    labels_path = os.path.join(args.data_dir, "labels.npy")
    train_loader, val_loader, test_loader = make_loaders(
        roi_ts_path, labels_path, batch_size=args.batch, seed=args.seed
    )
    print(f"train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  test={len(test_loader.dataset)}")

    input_dim = np.load(roi_ts_path).shape[1]
    model = ModelSTAGIN(
        input_dim   = input_dim,
        hidden_dim  = args.hidden_dim,
        num_classes = 2,
        num_heads   = args.num_heads,
        num_layers  = args.num_layers,
        sparsity    = args.sparsity,
        dropout     = args.dropout,
        cls_token   = args.cls_token,
        readout     = args.readout,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}  |  reg_lambda={args.reg_lambda:.0e}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # OneCycleLR: matches paper (pct_start=0.2, final_div_factor=1000)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = args.max_lr,
        steps_per_epoch = len(train_loader),
        epochs          = args.epochs,
        pct_start       = 0.2,
        final_div_factor= 1000,
    )

    ckpt_path  = os.path.join(args.out_dir, "best_model_fmri.pth")
    best_bacc  = 0.0
    best_epoch = 0
    no_improve = 0
    nan_streak = 0

    print(f"\n{'Ep':>4}  {'TrLoss':>8}  {'VaLoss':>8}  {'VaBAcc':>7}  {'VaF1':>6}  {'GPU MB':>8}  {'Time':>6}")
    print("-" * 58)

    epoch_log = []

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, _, _ = run_epoch(
            train_loader, model, optimizer, criterion, DEVICE,
            args.reg_lambda, train=True, scheduler=scheduler)
        va_loss, va_bacc, va_f1 = run_epoch(
            val_loader, model, optimizer, criterion, DEVICE,
            args.reg_lambda, train=False)

        elapsed = time.time() - t0
        gpu_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
        va_loss_str = f"{va_loss:>8.4f}" if va_loss == va_loss else "     nan"
        print(f"{ep:>4}  {tr_loss:>8.4f}  {va_loss_str}  {va_bacc:>7.4f}  {va_f1:>6.4f}  {gpu_mb:>7.0f}M  {elapsed:>5.1f}s")
        sys.stdout.flush()
        epoch_log.append({
            "epoch": ep,
            "train_loss": round(float(tr_loss), 6),
            "val_loss":   round(float(va_loss), 6) if va_loss == va_loss else None,
            "val_bacc":   round(float(va_bacc), 6),
            "val_f1":     round(float(va_f1), 6),
            "gpu_memory_mb": round(gpu_mb, 1),
            "time_s":     round(elapsed, 2),
        })

        is_nan_epoch = (va_bacc == 0.5 and va_f1 == 0.0)

        if va_bacc > best_bacc and not is_nan_epoch:
            best_bacc, best_epoch, no_improve, nan_streak = va_bacc, ep, 0, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            if is_nan_epoch and os.path.exists(ckpt_path):
                nan_streak += 1
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
                optimizer.state.clear()
                print(f"  [warn] NaN ep {ep} (streak={nan_streak}) — reset to ep {best_epoch}")
            else:
                nan_streak = 0
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {ep}")
                break

    print(f"\nBest val balanced accuracy: {best_bacc:.4f}  (epoch {best_epoch})")

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    te_loss, te_bacc, te_f1 = run_epoch(
        test_loader, model, optimizer, criterion, DEVICE,
        args.reg_lambda, train=False)
    print(f"Test  balanced accuracy: {te_bacc:.4f}   F1: {te_f1:.4f}   Loss: {te_loss:.4f}")
    print(f"Model saved: {ckpt_path}")

    # Save training metrics JSON for reproducibility
    metrics = {
        "best_val_bacc": round(best_bacc, 6),
        "best_epoch": best_epoch,
        "test_bacc": round(float(te_bacc), 6),
        "test_f1": round(float(te_f1), 6),
        "test_loss": round(float(te_loss), 6) if te_loss == te_loss else None,
        "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if torch.cuda.is_available() else 0.0,
        "epoch_log": epoch_log,
    }
    json_path = os.path.join(args.out_dir, "train_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {json_path}")


if __name__ == "__main__":
    main()
