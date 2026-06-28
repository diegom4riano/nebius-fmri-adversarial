# KAPPA: A Second-Order Adversarial Attack That Exposes What PGD Cannot Find in Clinical AI

*Nebius Serverless AI Builders Challenge — Healthcare & Life Sciences*

---

A model that classifies brain activity with 77% balanced accuracy sounds reasonably robust. Run PGD against it — the adversarial attack that underpins every major robustness benchmark — and you might conclude it has moderate vulnerabilities. Run KAPPA, the second-order attack I developed, and you find that **60.7% of targeted subjects can be silently misclassified with a perturbation smaller than the noise floor of the MRI acquisition**. PGD missed two-thirds of those cases.

This is not a problem with the model. It is a problem with how robustness is evaluated.

---

## The Limitation of Gradient-Based Attacks

PGD (Madry et al., 2018) is the foundation of adversarial machine learning. It iteratively takes a gradient step toward higher loss and projects back onto an L∞ constraint ball. It is simple, fast, and the basis of AutoAttack — the current gold standard for robustness benchmarking.

The issue is that gradient direction alone is unreliable when the loss surface is ill-conditioned. In an ill-conditioned landscape, some directions are orders of magnitude steeper than others. Gradient steps are dominated by those steep directions and make no progress on the flat ones. More iterations amplify the problem: the iterates oscillate rather than converge, and the model appears robust simply because PGD cannot navigate its geometry.

I designed **KAPPA** (κ-**A**ware **P**erturbation via **P**roximal **A**pproximation) to address this directly. KAPPA replaces PGD's sign-gradient step with a Newton step, solved by applying the Conjugate Gradient method to the system:

```
(H + λI) δ = −∇L
```

where H is the loss Hessian (approximated via Hessian-Vector Products using double-backward passes), λ is a regularizer set adaptively via the Rayleigh quotient to guarantee positive definiteness, and δ is the resulting perturbation direction. The Newton step accounts for curvature: it scales the gradient by the inverse Hessian, moving efficiently even in ill-conditioned landscapes.

The central hypothesis: **KAPPA's advantage over PGD is predicted by the Hessian condition number κ**. When κ ≈ 1, curvature information adds nothing — gradient and Newton direction are the same. When κ ≫ 1, the Newton direction finds adversarial examples that PGD cannot.

---

## Experimental Design: Controlling for κ

To validate the κ hypothesis, I needed two architectures with meaningfully different condition numbers while remaining clinically relevant. I trained both models from scratch.

**ECG Rhythm Classifier — PhysioNet/CinC 2017**
A 13-block dilated 1D CNN following the Han et al. architecture for 4-class rhythm classification (Normal, AF, Other, Noisy). Every convolutional block is followed by Batch Normalization. BN constrains gradient magnitudes across layers, keeping the loss surface well-conditioned. Estimated κ ≈ 1. Test accuracy: 87.5%.

**fMRI Sex Classifier — STAGIN on HCP**
A Spatio-Temporal Attention Graph Isomorphism Network (Kim & Ye) trained on 1,080 resting-state fMRI scans from the Human Connectome Project. The pipeline involves parcellating raw CIFTI files into 333 ROIs (Gordon atlas), computing dynamic functional connectivity matrices via a 50-TR sliding window (~50 windows per 1,200-TR acquisition), and training a hybrid GIN + self-attention + GRU architecture with no normalization layers anywhere in the network. Training used OneCycleLR scheduling, L2 regularization (λ=1e-5), and early stopping with patience=30. Estimated condition number: κ = **178,695**. Test BACC: 77.2%.

The prediction is unambiguous: KAPPA should underperform PGD on the ECG model and dramatically outperform it on STAGIN.

---

## Running KAPPA at Scale: Why the H200 Was Not Optional

Computing Hessian-Vector Products requires retaining the full computational graph through two backward passes. On STAGIN — with batch=32 subjects, each carrying a `[50 × 333 × 333]` adjacency tensor and a `[1200 × 333]` timeseries — the peak VRAM during KAPPA's backward passes hit **86,876 MB**.

An NVIDIA A100 has 80 GB. This experiment cannot run on an A100.

The **Nebius H200 SXM** (141 GB HBM3e) was the enabling hardware. Deployment via Nebius Serverless AI Jobs required a single command, with the S3-compatible bucket mounted directly as a filesystem:

```bash
nebius ai job create \
  --platform gpu-h200-sxm \
  --preset 1gpu-16vcpu-200gb \
  --volume storagebucket-...:/workspace/data \
  --container-command bash \
  --args '-c "pip install -r requirements.txt && \
              python test_fmri_model.py \
              --config configs/config.yaml \
              --output-dir /workspace/data/output \
              --run-id 20260628_035830"'
```

Results write directly to S3 after each epsilon completes (partial save), so job failures are recoverable without restarting the full sweep. A `RESUME_RUN_ID` flag in the Makefile allows picking up from any checkpoint. The full 6-attack × 5-epsilon sweep across 216 test subjects ran in ~12 hours at well under $100.

Three non-trivial engineering fixes were required to get all six attacks running reliably on STAGIN:

- **ForwardWrapper padding**: AutoAttack internally calls `forward()` with sub-batches smaller than B. STAGIN's GRU produces hidden states of fixed size B, causing `torch.cat` to fail. Fix: pad the input to B with zeros and return `logits[:n]`.
- **cuDNN RNN training mode**: PyTorch's cuDNN backend requires `model.train()` during RNN backward passes. Setting `model.eval()` inside `forward()` silently corrupts KAPPA's second-order gradients. Fix: maintain `train()` mode throughout the attack sweep.
- **Binary AutoAttack**: AutoAttack's standard configuration includes DLR and FAB-T losses, which require ≥3 classes. Fix: `version='custom', attacks_to_run=['apgd-ce', 'square']`.

---

## Results

Six attacks evaluated: KAPPA (mine), PGD-40, PGD-500 (budget-matched to KAPPA), AutoAttack (APGD-CE + Square), APGD-CE, and C&W L2. Metric: **True ASR** — fraction of Male subjects (n=84) whose predicted class flips to Female after the attack. This targeted metric avoids inflating rates with trivially adversarial examples.

### ECG CNN (κ ≈ 1) — The Baseline

| Method | ε | Steps | True ASR |
|---|---|---|---|
| PGD | 10 | 40 | **86.5%** |
| **KAPPA** | 10 | 5 outer × 10 CG | 72.9% |
| PGD | 2 | 40 | 24.0% |
| **KAPPA** | 2 | 5 outer × 50 CG | 21.9% |

On the BN-normalized ECG model, **PGD outperforms KAPPA**. At ε=2, both methods are equivalent; at ε=10, KAPPA is 13.6 pp worse. This is exactly what the κ hypothesis predicts: when the loss surface is well-conditioned, the Newton direction adds no information, and the overhead of CG iterations makes KAPPA less efficient than PGD. The baseline holds.

### STAGIN fMRI (κ = 178,695) — The Main Result

| Attack | ε=0.001 | ε=0.005 | ε=0.01 | ε=0.05 | ε=0.1 |
|---|---|---|---|---|---|
| **KAPPA** | **60.7%** | **58.5%** | **50.0%** | **92.7%** | **93.9%** |
| APGD-CE | 22.6% | 30.5% | 29.3% | 86.6% | 74.4% |
| PGD-40 | 31.0% | 35.4% | 41.5% | 54.9% | 53.7% |
| AutoAttack | 17.9% | 15.9% | 28.1% | 62.2% | 65.9% |
| C&W L2 | 16.7% | 18.3% | 18.3% | 18.3% | 18.3% |
| PGD-500 | 13.1% | 29.3% | 42.7% | 51.2% | 53.7% |

At ε=0.001 — a perturbation smaller than the precision of fMRI preprocessing — **KAPPA achieves 60.7% ASR while PGD-500 achieves 13.1%**, using the same compute budget (741s vs 3,771s). **PGD-500 is worse than PGD-40** at this epsilon: with κ=178,695, 500 iterations amplify oscillation rather than improve convergence. The model appears robust under PGD evaluation not because it is, but because PGD cannot find the adversarial region.

Three patterns emerge across the epsilon sweep:

1. **KAPPA dominates at tight budgets.** At ε≤0.005, KAPPA leads by a 2–4.6× margin. This is the regime that matters clinically: imperceptible perturbations.
2. **The gap narrows mid-range.** By ε=0.01, PGD-500 recovers to 42.7% vs KAPPA's 50%. The larger epsilon ball gives first-order methods enough room to overcome the conditioning problem.
3. **KAPPA reopens the gap at ε=0.05–0.1** (93% vs 53%). Even with a generous budget, PGD plateaus while KAPPA finds adversarial examples for nearly all targets.

C&W L2 flatlines at 18% across all epsilons. STAGIN's vulnerability is structurally aligned with L∞ directions, not L2 — L2-norm attacks cannot exploit it regardless of budget.

---

## The Implication for Medical AI Robustness

AutoAttack is the most widely used robustness benchmark in the literature. On STAGIN at ε=0.001, AutoAttack reports 17.9% ASR. KAPPA reports 60.7%. **A model that appears moderately vulnerable under the current gold-standard evaluation has 3.4× greater true vulnerability.**

This is a systemic issue, not an edge case. The adversarial ML literature has been built almost entirely around CNN architectures that use Batch Normalization — ResNets, VGGs, EfficientNets. BN normalizes gradient magnitudes and implicitly keeps κ near 1, making PGD an effective evaluator. The assumption that PGD is sufficient has been invisible because the benchmark models happened to satisfy it.

Medical AI increasingly relies on architectures that do not: Graph Neural Networks for functional connectivity, Transformers for EHR sequences, RNNs for physiological time series, attention-based models for histopathology. None of these routinely use the aggressive BN patterns of image classifiers. For any of these architectures, a robustness certificate from PGD-based evaluation may be misleading.

The κ estimate — computed cheaply via a few power iterations — can serve as a practical diagnostic: **if κ ≫ 1, include KAPPA in your robustness evaluation. If κ ≈ 1, PGD suffices.**

---

## Code and Reproducibility

KAPPA is implemented in `hessian.py` (precision-med repository), model-agnostic and requiring only a differentiable PyTorch `forward()`. The full evaluation pipeline — model checkpoints, configs, Nebius job scripts, and complete `attack_results.json` across all 5 epsilons — is open-source:

**[github.com/diegom4riano/nebius-fmri-adversarial](https://github.com/diegom4riano/nebius-fmri-adversarial)**

The entire experiment is reproducible with `make deploy-attack` from a Nebius account.

---

*#NebiusServerlessChallenge — Healthcare & Life Sciences*
