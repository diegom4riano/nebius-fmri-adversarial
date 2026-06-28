import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

import torch.nn as nn
from model.CNN import CNN
from utils.DataLoader import ECGDataset, ecg_collate_func
from hessian import targeted_attack, pgd_attack

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

MODEL_PATH = '/Users/diegomariano/Documents/precision-med/saved_model/best_model.pth'
DATA_DIR   = '/Users/diegomariano/Documents/precision-med/data/'

BATCH_SIZE  = 16
NUM_CLASSES = 4   # Normal, AF, Other, Noisy — igual ao treino

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Device: {device}')

# ── Data — mesmo split usado no treino (PERMUTATION + 10% val) ───────────────
X = np.load(os.path.join(DATA_DIR, 'raw_data.npy'),          allow_pickle=True)
y = np.load(os.path.join(DATA_DIR, 'raw_labels.npy'),        allow_pickle=True)
P = np.load(os.path.join(DATA_DIR, 'random_permutation.npy'), allow_pickle=True)

X, y = X[P], y[P]
mid  = int(len(X) * 0.97)
X, y = X[mid:], y[mid:]
logging.info(f'{len(X)} validation samples, 4-class labels')

dataset    = ECGDataset(X, y)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=ecg_collate_func)

# ── Model ───────────────────────────────────────────────────────────────────
model = CNN(num_classes=NUM_CLASSES)
state_dict = torch.load(MODEL_PATH, map_location=device)
state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.to(device)
model.eval()


# ── True ASR: apenas amostras não-alvo que foram flipadas pelo ataque ────────
def compute_true_asr(labels, preds_clean, preds_adv, target_class=0):
    labels      = np.array(labels)
    preds_clean = np.array(preds_clean)
    preds_adv   = np.array(preds_adv)
    # pool: amostras que o modelo limpo NÃO prediz como classe alvo
    pool = preds_clean != target_class
    if pool.sum() == 0:
        return 0.0
    flipped = (preds_adv[pool] == target_class).sum()
    return flipped / pool.sum()


# ── Evaluation helper ────────────────────────────────────────────────────────
def evaluate(dataloader, model, epsilon, attack_name):
    all_labels, preds_clean_all, preds_hess_all, preds_pgd_all = [], [], [], []
    total_batches = len(dataloader)
    t_start = time.time()

    for batch_idx, (data, lengths, labels) in enumerate(dataloader):
        elapsed = time.time() - t_start
        eta = (elapsed / (batch_idx + 1)) * (total_batches - batch_idx - 1) if batch_idx > 0 else 0
        print(f'[{attack_name}] batch {batch_idx+1:3d}/{total_batches} '
              f'| elapsed: {elapsed:5.1f}s | ETA: {eta:5.1f}s', flush=True)

        data, labels = data.to(device), labels.to(device)
        targets = torch.zeros_like(labels)   # targeted: push toward class 0 (Normal)

        def fwd(x):
            return model(x)

        # Clean predictions
        with torch.no_grad():
            preds_clean = fwd(data).argmax(dim=1)

        # Hessian attack
        data_hess = targeted_attack(fwd, data, targets,
                                    lambda_reg=1e-6, epsilon=epsilon,
                                    max_iter=50, num_steps=5, verbose=False)
        with torch.no_grad():
            preds_hess = fwd(data_hess).argmax(dim=1)

        # PGD attack (mesma implementação de hessian.py)
        data_pgd = pgd_attack(fwd, data, targets, epsilon=epsilon)
        with torch.no_grad():
            preds_pgd = fwd(data_pgd).argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        preds_clean_all.extend(preds_clean.cpu().numpy())
        preds_hess_all.extend(preds_hess.cpu().numpy())
        preds_pgd_all.extend(preds_pgd.cpu().numpy())

    labels_arr      = np.array(all_labels)
    preds_clean_arr = np.array(preds_clean_all)
    preds_hess_arr  = np.array(preds_hess_all)
    preds_pgd_arr   = np.array(preds_pgd_all)

    acc_clean = np.mean(preds_clean_arr == labels_arr)
    asr_hess  = compute_true_asr(labels_arr, preds_clean_arr, preds_hess_arr)
    asr_pgd   = compute_true_asr(labels_arr, preds_clean_arr, preds_pgd_arr)

    print(f'\n── ε={epsilon} ──')
    print(f'  Clean accuracy          : {acc_clean * 100:.2f}%')
    print(f'  Hessian True ASR        : {asr_hess  * 100:.2f}%')
    print(f'  PGD     True ASR        : {asr_pgd   * 100:.2f}%')
    print(f'  Advantage (Hessian−PGD) : {(asr_hess - asr_pgd) * 100:+.2f}%')

    return acc_clean, asr_hess, asr_pgd


# ── Epsilon sweep ─────────────────────────────────────────────────────────────
print('\n' + '='*55)
print('Clean + Adversarial Evaluation — ECG CNN')
print('='*55)

results = []
for eps in [10, 50, 100, 200]:
    acc, asr_h, asr_p = evaluate(dataloader, model, epsilon=eps, attack_name=f'eps={eps}')
    results.append((eps, acc, asr_h, asr_p))

print('\n' + '='*55)
print('Summary')
print('='*55)
print(f'{"ε":>6}  {"Clean Acc":>10}  {"Hessian ASR":>12}  {"PGD ASR":>9}  {"Advantage":>10}')
print('-' * 55)
for eps, acc, asr_h, asr_p in results:
    print(f'{eps:>6}  {acc*100:>9.2f}%  {asr_h*100:>11.2f}%  {asr_p*100:>8.2f}%  {(asr_h-asr_p)*100:>+9.2f}%')
print('='*55)
