# Validating Newton-CG: A Second-Order Adversarial Attack on Clinical AI at Scale

*Nebius Serverless AI Builders Challenge — Healthcare & Life Sciences*

---

## Why I Built a Second-Order Attack

The standard adversarial attack against neural networks is PGD (Projected Gradient Descent, Madry et al. 2018): take a gradient step, project back onto the constraint set, repeat. It is simple, fast, and the foundation of almost every robustness benchmark in the literature.

The problem with PGD is that it only uses first-order information — the gradient. It ignores the curvature of the loss surface. In a well-conditioned landscape, this doesn't matter. In an ill-conditioned one — where some directions are 100,000× steeper than others — gradient steps are systematically misleading. PGD oscillates in the steep directions and makes no progress in the flat ones.

I developed **Newton-CG**, a second-order adversarial attack that replaces the sign-gradient step with a Newton step obtained by solving the linear system `(H + λI)δ = −∇L` via Conjugate Gradient, where H is the loss Hessian approximated through Hessian-Vector Products (HVPs). The regularizer λ is set adaptively via the Rayleigh quotient to ensure positive definiteness. The central hypothesis: **Newton-CG should outperform PGD when the Hessian condition number κ is large, and not otherwise**.

This post describes the large-scale empirical validation of that hypothesis on two clinical neural networks, run on a Nebius H200 GPU.

---

## Two Clinical Models, Two Architectures

To test the κ hypothesis, I needed two models that differed in their expected condition number while remaining clinically relevant.

**Model 1 — ECG Rhythm Classifier (CNN)**
A 13-block dilated 1D CNN (Han et al. architecture) trained on PhysioNet/CinC 2017 (4-class: Normal, AF, Other, Noisy). Every convolutional block is followed by Batch Normalization. BN constrains the gradient magnitudes layer-by-layer, keeping the loss surface well-conditioned: estimated κ ≈ 1. Test accuracy: 87.5%.

**Model 2 — fMRI Sex Classifier (STAGIN)**
A Spatio-Temporal Attention Graph Isomorphism Network (Kim & Ye) trained on 1,080 resting-state fMRI scans from the Human Connectome Project. The architecture combines 4 GIN layers with multi-head self-attention and a GRU over time — no batch normalization anywhere in the pipeline. Inputs are 333×333 dynamic functional connectivity matrices computed via a 50-TR sliding window over 1,200-TR acquisitions. Training used OneCycleLR scheduling with L2 regularization (λ=1e-5) and patience-based early stopping. Estimated condition number: κ = **178,695**. Test BACC: 77.2%.

The prediction: Newton-CG should fail to improve on PGD for the ECG model, and dramatically outperform PGD for the STAGIN model.

---

## The Computational Infrastructure

Running Newton-CG at scale requires double-backward passes to compute HVPs — each outer iteration performs `K` CG steps, each of which costs one HVP. The memory footprint is large: storing intermediate activations for second-order differentiation through a GRU + 4-layer GIN on batch=32 subjects peaked at **86,876 MB (86.9 GB)**.

An NVIDIA A100 has 80 GB HBM2e. This experiment cannot run on an A100.

I deployed the evaluation on a **Nebius H200 SXM** (141 GB HBM3e) via Nebius Serverless AI Jobs:

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

Results write directly to an S3-compatible bucket mounted at `/workspace/data`. A partial save mechanism writes results after each epsilon completes, so a job crash loses at most one epsilon's worth of compute. The entire 6-attack × 5-epsilon sweep ran in ~12 hours. Total cost: under $100.

---

## Engineering the Attack Pipeline

Adapting Newton-CG to work alongside standard attack libraries (AutoAttack, torchattacks) on STAGIN required solving three non-trivial engineering problems.

**Sub-batch size mismatch.** AutoAttack calls `forward(v_sub)` with sub-batches smaller than the original batch. STAGIN's internal `torch.cat([v, time_encoding], dim=3)` requires that the batch dimension of `v` matches the batch dimension produced by the GRU's hidden state — which is fixed at training-time batch size B. Fix: a `ForwardWrapper` that pads `v` with zeros to size B, runs the full forward pass, and returns `logits[:n]`.

**cuDNN RNN and second-order backward.** PyTorch's cuDNN backend requires the model to be in `train()` mode for RNN backward passes. Calling `model.eval()` inside `forward()` — common in inference code — silently corrupts C&W L2 and Newton-CG gradients because `eval()` executes before backward runs. Fix: keep the model in `train()` throughout the entire attack sweep without restoring `eval()`.

**Binary AutoAttack.** AutoAttack's standard configuration includes DLR loss and FAB-T, which require ≥3 classes. For the binary fMRI classifier these components throw or produce undefined behavior. Fix: `version='custom', attacks_to_run=['apgd-ce', 'square']`.

---

## Results

I evaluate six attacks: Newton-CG (mine), PGD-40, PGD-500 (budget-matched to Newton-CG), AutoAttack (APGD-CE + Square), APGD-CE, and C&W L2. Metric: **True ASR** — fraction of Male subjects (84 total) whose prediction flips to Female. I use Male→Female as the targeted direction to avoid inflating rates with trivially easy examples.

### ECG CNN (κ ≈ 1)

| Method | ε | Steps | True ASR |
|---|---|---|---|
| PGD | 10 | 40 | **86.5%** |
| Newton-CG | 10 | 5 outer × 10 CG | 72.9% |
| PGD | 2 | 40 | 24.0% |
| Newton-CG | 2 | 5 outer × 50 CG | 21.9% |

On the BN-normalized ECG model, **PGD outperforms Newton-CG**. At ε=2, both methods achieve roughly the same ASR; at ε=10, Newton-CG is actually 13.6 pp worse despite using second-order information. This matches the κ ≈ 1 prediction: when the loss surface is well-conditioned, the Newton direction adds no information over the gradient, and the overhead of CG iterations makes it strictly less efficient.

### STAGIN — fMRI (κ = 178,695)

| Attack | ε=0.001 | ε=0.005 | ε=0.01 | ε=0.05 | ε=0.1 |
|---|---|---|---|---|---|
| **Newton-CG** | **60.7%** | **58.5%** | **50.0%** | **92.7%** | **93.9%** |
| APGD-CE | 22.6% | 30.5% | 29.3% | 86.6% | 74.4% |
| PGD-40 | 31.0% | 35.4% | 41.5% | 54.9% | 53.7% |
| AutoAttack | 17.9% | 15.9% | 28.1% | 62.2% | 65.9% |
| C&W L2 | 16.7% | 18.3% | 18.3% | 18.3% | 18.3% |
| PGD-500 | 13.1% | 29.3% | 42.7% | 51.2% | 53.7% |

On the un-normalized STAGIN model, **Newton-CG achieves 60.7% ASR at ε=0.001 while PGD-500 achieves only 13.1%** — a 4.6× gap using the same compute budget (741s vs 3,771s).

Notably, **PGD-500 is worse than PGD-40** at ε=0.001 (13.1% vs 31.0%). This is a clean demonstration of the oscillation pathology: with κ=178,695 and a tiny epsilon ball, more gradient steps amplify divergence rather than improve convergence. The model appears robust under PGD evaluation precisely because PGD cannot navigate its loss surface.

The gap narrows as ε grows (at ε=0.01, PGD-500 recovers to 42.7% vs Newton-CG's 50%), confirming the mechanism: a larger epsilon ball gives first-order methods enough room to work around the ill-conditioned directions. At ε=0.05–0.1, Newton-CG jumps to 93%+ while PGD plateaus at ~53%, suggesting a regime where the Newton direction finds adversarial examples that PGD misses even with a generous budget.

C&W L2 flatlines at ~18% across all epsilon values. The STAGIN model's adversarial vulnerability is aligned with L∞ directions, not L2 — so an L2-norm attack, regardless of epsilon, cannot exploit the same vulnerabilities.

---

## What This Means for Robustness Evaluation

AutoAttack is the current gold standard for robustness evaluation (Croce & Hein, 2020). On STAGIN at ε=0.001, AutoAttack reports 17.9% ASR. Newton-CG reports 60.7%. **A model evaluated as relatively robust under AutoAttack has a 3.4× higher true vulnerability.**

This is not a failure of AutoAttack — it is a property of the model. STAGIN's lack of normalization layers creates a loss landscape that first-order methods systematically fail to explore. Any robustness benchmark that uses only PGD-family attacks is implicitly assuming that the models being evaluated are well-conditioned, an assumption that does not hold for graph neural networks, RNNs, and other architectures common in medical imaging that are not descended from the BN-heavy ResNet family.

The practical implication: robustness evaluations of clinical AI models should include at least one second-order attack, particularly for architectures without batch normalization. The κ estimate (cheaply computed via a few Rayleigh quotient power iterations) can serve as a proxy for whether PGD-based evaluation is trustworthy.

---

## Code and Results

All code, trained checkpoints, configs, and the full results JSON are open-source:
**[github.com/diegom4riano/nebius-fmri-adversarial](https://github.com/diegom4riano/nebius-fmri-adversarial)**

The Newton-CG implementation (`hessian.py` in the precision-med repository) is self-contained and model-agnostic — it requires only a differentiable `forward()` function and works with any PyTorch model.

---

*#NebiusServerlessChallenge — Healthcare & Life Sciences*
