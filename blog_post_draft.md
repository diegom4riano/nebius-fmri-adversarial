# Can You Fool a Brain Scan Model? Building Adversarial Attacks on Clinical AI with Nebius H200

*Nebius Serverless AI Builders Challenge — Healthcare & Life Sciences*

---

## The Starting Point: Two Clinical Models

The question I wanted to answer was simple: how vulnerable are medical AI models to small, targeted input perturbations? I had two models already trained from a previous research project:

**Model 1 — ECG rhythm classifier (CNN)**
A 13-block dilated 1D CNN following the Han et al. architecture, trained on the PhysioNet/CinC 2017 challenge dataset (4-class rhythm classification: Normal, AF, Other, Noisy). Training ran for 50 epochs with the standard Adam optimizer and a step LR scheduler, reaching **87.5% test accuracy**. The model uses Batch Normalization after every convolutional block.

**Model 2 — fMRI sex classifier (STAGIN)**
A Spatio-Temporal Attention Graph Isomorphism Network trained on 1,080 resting-state fMRI scans from the Human Connectome Project (HCP). The training pipeline was significantly more complex:
- Raw CIFTI files were parcellated into 333 ROIs using the Gordon atlas
- Sliding windows (50 TRs, stride 3) generate dynamic functional connectivity matrices (~50 windows per subject)
- The model combines 4 GIN layers with multi-head self-attention and a GRU across time windows
- Training used `OneCycleLR` scheduler, L2 regularization (λ=1e-5), and early stopping with patience=30
- Final test split: 216 subjects, **BACC = 77.2%**

Both models were already checkpointed and ready. The new work was building the attack pipeline and running it at scale.

---

## The Attack Pipeline

I implemented six attacks, all targeting the same threat model: **L∞ perturbations** on the raw input, small enough to be imperceptible, large enough to flip the predicted class.

- **Newton-CG** — a second-order attack that solves `(H + λI)δ = −∇L` at each step using Conjugate Gradient, where H is approximated via Hessian-Vector Products (HVPs) computed with double-backward passes
- **PGD-40** — classic Projected Gradient Descent, 40 steps
- **PGD-500** — same as PGD-40 but with 5× more steps, budget-matched to Newton-CG
- **AutoAttack** — ensemble of APGD-CE + Square attack (binary-safe configuration)
- **APGD-CE** — Auto-PGD with adaptive step size from `torchattacks`
- **C&W L2** — Carlini-Wagner L2 norm attack (50 steps)

The metric is **True Attack Success Rate (True ASR)**: the fraction of subjects in the non-target class whose prediction flips after the attack. For the fMRI model, this is Male subjects (predicted class 1) whose prediction flips to Female (class 0).

---

## The Engineering Challenges

Getting six different attack libraries to agree on a single STAGIN forward pass turned out to be the hardest part.

**The batch size problem.** AutoAttack internally splits batches into sub-batches and calls `forward(v_sub)` where `v_sub` has fewer samples than the original batch. STAGIN's GRU uses batch-first tensors for the adjacency matrix (`[B, T_w, N, N]`) but seq-first for the temporal input (`[T, B, N_rois]`). When `v_sub.shape[0] < B`, the `torch.cat([v, time_encoding], dim=3)` inside STAGIN throws a size mismatch. The fix: a `ForwardWrapper` that pads `v` with zeros back to the original batch size B, runs the full forward pass, and returns only `logits[:n]`.

**The cuDNN RNN backward problem.** C&W L2 requires second-order differentiation through the GRU. PyTorch's cuDNN backend only supports RNN backward in training mode — calling `model.eval()` inside `forward()` (a common pattern for inference) breaks C&W's backward pass because `eval()` gets called after `forward()` returns but before backward runs. Fix: keep the model in `train()` mode throughout the entire attack sweep, never restore `eval()`.

**The AutoAttack binary problem.** AutoAttack's standard version includes DLR loss and FAB-T, which require at least 3 classes. For a binary classifier, these are silently skipped or misconfigured. Fix: `version='custom', attacks_to_run=['apgd-ce', 'square']`.

**Partial saves and resume.** The full sweep (6 attacks × 5 epsilons) takes ~12 hours on a single H200. Each epsilon is saved to S3 immediately after completion. If a job crashes at ε=0.05, the next job resumes from where it left off: `make deploy-attack RESUME_RUN_ID=<old_run_id>`.

---

## Why the H200 Was Not Optional

The STAGIN forward pass is memory-intensive: batch=32, each sample is a `[T_w=50, N=333, N=333]` sparse adjacency + `[T=1200, N_rois=333]` timeseries. The peak VRAM during an attack (which requires storing intermediate activations for backward passes) hit **86,876 MB — 86.9 GB**.

An A100 has 80 GB HBM2e. This experiment cannot run on an A100.

The **Nebius H200 SXM** has 141 GB HBM3e. Setup was a single CLI command:

```bash
nebius ai job create \
  --platform gpu-h200-sxm \
  --preset 1gpu-16vcpu-200gb \
  --volume storagebucket-...:/workspace/data \
  --container-command bash \
  --args '-c "pip install -r requirements.txt && python test_fmri_model.py \
              --config configs/config.yaml \
              --output-dir /workspace/data/output \
              --run-id $(date +%Y%m%d_%H%M%S)"'
```

The S3 bucket mounts at `/workspace/data`, results write directly to S3, and the hardware is released as soon as the job finishes. No persistent VM billing, no cluster management. Total job runtime: **~12 hours**. The final `attack_results.json` (5.1 KB) is in the repository.

---

## Results: ECG vs fMRI

Before running the fMRI sweep, I had ECG baseline results from the precision-med project. The comparison reveals something unexpected.

### ECG CNN — PhysioNet/CinC 2017 (ε in ADC units)

| Method | ε | Steps | True ASR |
|---|---|---|---|
| PGD | 10 | 40 | **86.5%** |
| Newton-CG | 10 | 5 outer × 10 CG | 72.9% |
| PGD | 2 | 40 | 24.0% |
| Newton-CG | 2 | 5 outer × 50 CG | 21.9% |

On the ECG model, **PGD outperforms Newton-CG**. More CG iterations don't help. Second-order information is useless here.

### STAGIN — HCP fMRI (ε in correlation units, 84 Male test subjects)

| Attack | ε=0.001 | ε=0.005 | ε=0.01 | ε=0.05 | ε=0.1 |
|---|---|---|---|---|---|
| **Newton-CG** | **60.7%** | **58.5%** | **50.0%** | **92.7%** | **93.9%** |
| APGD-CE | 22.6% | 30.5% | 29.3% | 86.6% | 74.4% |
| PGD-40 | 31.0% | 35.4% | 41.5% | 54.9% | 53.7% |
| AutoAttack | 17.9% | 15.9% | 28.1% | 62.2% | 65.9% |
| C&W L2 | 16.7% | 18.3% | 18.3% | 18.3% | 18.3% |
| PGD-500 | 13.1% | 29.3% | 42.7% | 51.2% | 53.7% |

On the fMRI model, the picture flips: **Newton-CG achieves 60.7% ASR at ε=0.001 while PGD-500 achieves only 13.1%** — using 5× more gradient steps and 5× more wall-clock time (741s vs 3,771s).

The gap closes as epsilon grows (at ε=0.01, PGD-500 catches up to 42.7% vs Newton-CG's 50%), but reopens at ε=0.05 where Newton-CG jumps to 93% while PGD plateaus at ~53%.

### What Explains the Difference?

The ECG CNN uses Batch Normalization after every layer, which normalizes gradient directions and makes the loss landscape well-conditioned. The estimated Hessian condition number for the ECG model is κ ≈ 1. PGD works perfectly.

The STAGIN model has no batch normalization. The estimated condition number is κ = **178,695**. The loss landscape has directions 178,000× steeper than others. Gradient descent oscillates in these directions; Newton-CG solves for the correct step direction. This is why the same attack code produces completely different results on the two models.

C&W L2 is a special case: it stays at ~18% regardless of ε because it minimizes L2 perturbation, and the fMRI model's vulnerabilities are aligned with L∞ directions, not L2.

---

## The Infrastructure in Practice

The entire workflow lives in a single `Makefile`:

```bash
make upload-data     # sync HCP preprocessed FC matrices to S3
make deploy-attack   # launch Nebius H200 job, write results to S3
make download-results # pull attack_results.json from S3
```

Total cost for the full 12-hour H200 job: well under the $100 Nebius Builder Challenge budget. The S3 storage for all 1,113 subjects' FC matrices (~488 GB) is separate and was pre-uploaded from an earlier experiment.

The resume mechanism meant that when we needed to iterate on the attack code (fixing the ForwardWrapper and cuDNN bugs), we could redeploy and skip already-completed epsilons instead of restarting from scratch.

---

## Conclusion

Two models, same attack code, completely different results — because architecture choices (Batch Normalization vs none) change the loss geometry in ways that flip which optimizer wins. PGD-based robustness evaluations that skip second-order methods may miss real vulnerabilities in models without normalization layers, which are common in graph neural networks and other non-standard architectures used in medical imaging.

All code, configs, trained checkpoints, and the full results JSON are open-source:
**[github.com/diegom4riano/nebius-fmri-adversarial](https://github.com/diegom4riano/nebius-fmri-adversarial)**

---

*#NebiusServerlessChallenge — Healthcare & Life Sciences*
