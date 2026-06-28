# Validating KAPPA: A Second-Order Adversarial Attack on Clinical AI at Scale

*Nebius Serverless AI Builders Challenge — Healthcare & Life Sciences*

---

A model that classifies brain activity with 77% balanced accuracy sounds reasonably robust. Run AutoAttack against it — the current gold standard for adversarial robustness evaluation (Croce & Hein, 2020), trusted by the community precisely because it combines the best first-order and black-box strategies into a single ensemble — and it reports 17.9% attack success rate. Run KAPPA, the second-order attack I developed, and the same model shows **60.7% vulnerability** with a perturbation smaller than the noise floor of the MRI acquisition.

AutoAttack missed more than two-thirds of those cases.

This is not a problem with the model. It is a structural limitation of every attack that relies solely on gradient direction — including the current state of the art.

---

## The Shared Limitation of First-Order Attacks

The adversarial attack landscape today spans a spectrum of sophistication. At the classic end, **PGD** (Madry et al., 2018) takes a sign-gradient step and projects onto the L∞ ball — fast, simple, foundational. At the state of the art, **AutoAttack** (Croce & Hein, 2020) combines four strategies: APGD-CE (adaptive step-size PGD with cross-entropy loss), APGD-DLR (PGD with difference-of-logits ratio loss), FAB-T (boundary attack minimizing perturbation norm), and Square Attack (black-box random search). AutoAttack was specifically designed to overcome the known weaknesses of vanilla PGD — step size sensitivity, loss function choice, and local optima.

Yet all of these attacks share a fundamental property: **they use only first-order information**. Gradient direction. No curvature. APGD's adaptive step scheduler makes PGD steps smarter, but it cannot fix a misleading gradient direction — and gradient directions are systematically misleading on ill-conditioned loss surfaces.

When the Hessian condition number κ is large — some loss directions are orders of magnitude steeper than others — gradient steps oscillate in the steep directions and stall in the flat ones. An adaptive step size applied to a misleading direction just oscillates more efficiently. More iterations amplify the problem: at ε=0.001, PGD-500 achieves *lower* ASR than PGD-40. The model appears robust not because it is, but because no first-order method can navigate its geometry.

I designed **KAPPA** (κ-**A**ware **P**erturbation via **P**roximal **A**pproximation) to address this at the root. KAPPA replaces the gradient step with a Newton step, solved by applying the Conjugate Gradient method to the system:

```
(H + λI) δ = −∇L
```

where H is the loss Hessian (approximated via Hessian-Vector Products using double-backward passes), λ is a regularizer set adaptively via the Rayleigh quotient to guarantee positive definiteness, and δ is the resulting search direction. The Newton step accounts for curvature explicitly: it rescales the gradient by the inverse Hessian, moving efficiently even where the landscape is maximally ill-conditioned.

The central hypothesis: **KAPPA's advantage over the state of the art is predicted by the Hessian condition number κ**. When κ ≈ 1, curvature adds no information and KAPPA is equivalent to — or slower than — first-order attacks. When κ ≫ 1, KAPPA finds adversarial examples that the entire first-order family cannot.

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

Six attacks evaluated: KAPPA (mine), AutoAttack (APGD-CE + Square — current state of the art), APGD-CE (AutoAttack's strongest component), PGD-40, PGD-500 (budget-matched to KAPPA), and C&W L2. Metric: **True ASR** — fraction of Male subjects (n=84) whose predicted class flips to Female after the attack. This targeted metric avoids inflating rates with trivially adversarial examples.

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

At ε=0.001 — a perturbation smaller than the precision of fMRI preprocessing — the contrast is stark:

- **AutoAttack** (state of the art): **17.9% ASR** in 2,762s
- **APGD-CE** (AutoAttack's strongest component): **22.6%** in 783s
- **KAPPA**: **60.7%** in 741s

KAPPA finds **3.4× more adversarial examples than AutoAttack** in less wall-clock time. And this is with AutoAttack's full adaptive machinery — adaptive step sizes, restart strategies, and two complementary attack objectives. The advantage is not computational: KAPPA uses roughly the same budget as APGD-CE. The advantage is structural: it uses curvature information that no first-order method has access to.

The budget-matched baseline makes this precise: **PGD-500 achieves 13.1%** spending 3,771s — 5× more time than KAPPA at 741s. Remarkably, PGD-500 is also worse than PGD-40 (31.0%) at this epsilon. With κ=178,695, more gradient iterations amplify oscillation; the model appears robust under any first-order evaluation simply because the attacks cannot navigate its loss surface.

Three patterns emerge across the full epsilon sweep:

1. **KAPPA dominates at tight budgets.** At ε≤0.005, KAPPA leads AutoAttack by 2.7–3.7×. This is the clinically relevant regime: perturbations small enough to evade human review.
2. **The gap narrows mid-range.** At ε=0.01, first-order methods partially recover (APGD-CE reaches 29.3%, KAPPA 50%) as the larger epsilon ball gives gradient methods enough room to work. The advantage drops to ~1.7×.
3. **KAPPA reopens the gap at large epsilon.** At ε=0.05–0.1, KAPPA jumps to 93%+ while first-order methods plateau near 53–66%. Even with a generous budget, no first-order attack saturates the model's vulnerability — KAPPA does.

C&W L2 flatlines at 18% across all epsilons. STAGIN's vulnerability is structurally aligned with L∞ directions, not L2 — L2-norm attacks cannot exploit it regardless of budget.

---

## The Implication for Medical AI Robustness

AutoAttack was designed to be the hardest reasonable first-order benchmark — an ensemble that patches the known weaknesses of vanilla PGD. On STAGIN at ε=0.001, it reports 17.9% ASR. KAPPA reports 60.7%. **A model evaluated as having moderate vulnerability under the current gold standard has 3.4× greater true vulnerability.**

This is not a failure of AutoAttack. AutoAttack is exactly what it claims to be: the best practical first-order evaluator. The gap is a consequence of the model architecture, not the attack design. And this is a systemic issue.

The adversarial ML literature has been built almost entirely on CNN architectures that use Batch Normalization — ResNets, VGGs, EfficientNets, the CIFAR-10 and ImageNet benchmarks. BN normalizes gradient magnitudes layer-by-layer, implicitly keeping κ near 1 and making every first-order attack a fair evaluator. The assumption that gradient-based attacks are sufficient has been invisible for years because the benchmark architectures happened to satisfy it.

Medical AI operates in a different regime. Graph Neural Networks for functional connectivity (like STAGIN), Transformers for EHR sequences, RNNs for physiological time series, attention models for histopathology — none of these routinely use the aggressive normalization patterns of image classifiers. For any of these architectures, a robustness certificate issued by AutoAttack may be systematically optimistic.

The κ estimate — computed cheaply via a few Rayleigh quotient power iterations before running any attack — can serve as a practical diagnostic: **if κ ≫ 1, include KAPPA in your robustness evaluation. If κ ≈ 1, AutoAttack suffices.**

---

## Code and Reproducibility

KAPPA is implemented in `hessian.py` (precision-med repository), model-agnostic and requiring only a differentiable PyTorch `forward()`. The full evaluation pipeline — model checkpoints, configs, Nebius job scripts, and complete `attack_results.json` across all 5 epsilons — is open-source:

**[github.com/diegom4riano/nebius-fmri-adversarial](https://github.com/diegom4riano/nebius-fmri-adversarial)**

The entire experiment is reproducible with `make deploy-attack` from a Nebius account.

---

*#NebiusServerlessChallenge — Healthcare & Life Sciences*
