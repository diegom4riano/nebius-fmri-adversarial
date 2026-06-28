# Validating KAPPA: A Second-Order Adversarial Attack on Clinical AI at Scale

*Nebius Serverless AI Builders Challenge — Healthcare & Life Sciences*

---

A model that classifies brain activity with 77% balanced accuracy sounds reasonably robust. Run AutoAttack against it — the current gold standard for adversarial robustness evaluation (Croce & Hein, 2020) — and it reports 17.9% attack success. Run KAPPA, the second-order attack I developed, and the same model shows **60.7% vulnerability** with a perturbation smaller than the noise floor of the MRI acquisition.

AutoAttack missed more than two-thirds of those cases. This is not a problem with the model. It is a structural limitation of every attack that relies solely on gradient direction — including the current state of the art.

---

## The Shared Limitation of First-Order Attacks

The adversarial attack landscape spans a spectrum of sophistication. At the classic end, **PGD** (Madry et al., 2018) takes a sign-gradient step and projects onto the L∞ ball. At the state of the art, **AutoAttack** (Croce & Hein, 2020) combines four strategies: APGD-CE (adaptive step-size PGD), APGD-DLR, FAB-T (boundary minimization), and Square Attack (black-box random search). AutoAttack was specifically designed to overcome the known weaknesses of vanilla PGD — step size sensitivity, loss function choice, and local optima.

Yet all of these attacks share a fundamental property: **they use only first-order information**. APGD's adaptive step scheduler makes gradient steps smarter, but it cannot fix a misleading gradient direction — and gradient directions are systematically misleading on ill-conditioned loss surfaces.

When the Hessian condition number κ is large, some loss directions are orders of magnitude steeper than others. Gradient steps oscillate in the steep directions and stall in the flat ones. More iterations amplify the problem rather than fixing it.

I designed **KAPPA** (κ-**A**ware **P**erturbation via **P**roximal **A**pproximation) to address this at the root. Instead of following the gradient, KAPPA computes the Newton direction at each outer iteration — a search direction that accounts for curvature rather than just slope. The Hessian is never explicitly materialized; it is approximated on-the-fly via Hessian-Vector Products computed through double-backward passes, making KAPPA tractable on large models. A proximal regularization term ensures numerical stability in ill-conditioned regions. The full method is described in an upcoming paper and is model-agnostic: it requires only a differentiable PyTorch `forward()`.

The central hypothesis: **KAPPA's advantage over the state of the art is predicted by κ**. When κ ≈ 1, Newton and gradient directions coincide — KAPPA adds overhead without benefit. When κ ≫ 1, the Newton direction finds adversarial examples that no first-order method can reach.

---

## Two Clinical Models, Designed to Test the Hypothesis

To validate the κ hypothesis, I needed two architectures with meaningfully different condition numbers. I trained both from scratch.

**ECG Rhythm Classifier — PhysioNet/CinC 2017**
A 13-block dilated 1D CNN (Han et al. architecture) for 4-class rhythm classification. Every convolutional block is followed by Batch Normalization. BN normalizes gradient magnitudes layer-by-layer, keeping the loss surface well-conditioned. Estimated κ ≈ 1. Test accuracy: 87.5%.

**fMRI Sex Classifier — STAGIN on HCP**
A Spatio-Temporal Attention Graph Isomorphism Network (Kim & Ye) trained on 1,080 resting-state fMRI scans from the Human Connectome Project. The preprocessing pipeline: CIFTI files → 333 ROIs (Gordon atlas) → 50-TR sliding-window functional connectivity matrices (~50 windows per 1,200-TR acquisition). The model combines 4 GIN layers, multi-head self-attention, and a GRU over time — **no normalization layers anywhere**. Training used OneCycleLR scheduling, L2 regularization (λ=1e-5), and early stopping with patience=30. Estimated condition number: κ = **178,695**. Test BACC: 77.2%.

> *Figure 1 — Training curves for both models. The STAGIN loss shows the characteristic noisy convergence of un-normalized GNN+RNN architectures; the ECG CNN converges smoothly under BN.*

![training_curve](figures/training_curve.png)

The prediction is unambiguous: KAPPA should underperform first-order attacks on the ECG model (κ ≈ 1) and dramatically outperform them on STAGIN (κ ≫ 1).

---

## The System Architecture

The full pipeline runs across three components:

```
Local Machine                      Nebius Cloud
─────────────────                  ──────────────────────────────────
precision-med/                     H200 SXM (141 GB HBM3e)
  hessian.py  ──┐                  ┌── test_fmri_model.py
  STAGIN.py   ──┤                  │     └─ KAPPA × 6 attacks × 5 ε
  configs/    ──┤                  │     └─ partial save after each ε
                │   Nebius AI Job  │
                ├─── deploy ──────►│
                │                  └── results → S3
                │
                │   S3 (488 GB HCP data + results)
                ├─── upload-data ──► precision-med-hcp/
                └─── download ◄──── output/attack_results.json
```

The entire workflow is three Makefile targets:

```makefile
make upload-data       # sync preprocessed HCP FC matrices to S3 (~488 GB, once)
make deploy-attack     # launch H200 job, write results directly to S3
make download-results  # pull attack_results.json when job completes

# Resume a failed job from the last completed epsilon:
make deploy-attack RESUME_RUN_ID=20260628_035830
```

Results write directly to the S3-mounted filesystem after each epsilon completes. A job crash loses at most one epsilon's work — the resume mechanism reloads the partial JSON and skips already-completed runs.

---

## Why the H200 Was Not Optional

Computing Hessian-Vector Products requires retaining the full computational graph through two backward passes. On STAGIN with batch=32:

| Resource | Value |
|---|---|
| Input per subject | `[50 × 333 × 333]` adjacency + `[1200 × 333]` timeseries |
| Peak VRAM (KAPPA backward) | **86,876 MB** |
| A100 SXM capacity | 80,000 MB |
| H200 SXM capacity | 141,000 MB |
| Headroom on H200 | ~54 GB |

An A100 cannot run this experiment. The H200 SXM on Nebius Serverless AI was not a preference — it was the minimum viable GPU.

Deployment was a single CLI call:

```bash
nebius ai job create \
  --platform gpu-h200-sxm \
  --preset 1gpu-16vcpu-200gb \
  --volume storagebucket-e005764888512084834516:/workspace/data \
  --container-image pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime \
  --container-command bash \
  --args '-c "pip install -r requirements.txt && \
              python test_fmri_model.py \
              --config configs/config.yaml \
              --output-dir /workspace/data/output \
              --run-id 20260628_035830"' \
  --timeout 86400
```

No cluster setup, no persistent VM billing, no storage provisioning beyond the S3 bucket. Total job runtime: ~12 hours. Total cost: under $100.

---

## The Honest Part: Three Bugs That Cost a Full H200 Job

Getting six attack libraries to cooperate on a GRU-based GNN took more iteration than expected. Each of the three failures below was silent — no crash, no error, just wrong results — which made them expensive to discover on a 6-hour H200 job.

**1. Sub-batch size mismatch (discovered after 6 hours)**

The first full job completed cleanly — then I checked the output and found that AutoAttack had reported 0% ASR across all epsilons. No exception was raised. Digging into AutoAttack's internals revealed that it internally splits batches into sub-batches and calls `forward(v_sub)` with fewer samples than the original batch B. STAGIN's GRU produces hidden states fixed at size B; `torch.cat([v, time_encoding], dim=3)` silently skipped the incompatible tensors.

The fix: a `ForwardWrapper` that pads the input back to B with zeros and returns only the relevant logits:

```python
class ForwardWrapper(torch.nn.Module):
    def forward(self, v):
        n, B = v.shape[0], self._B
        if n < B:
            pad = torch.zeros((B - n,) + v.shape[1:], device=v.device, dtype=v.dtype)
            v = torch.cat([v, pad], dim=0)
        self.model.train()
        logits, _, _, _ = self.model(v, self._a, self._t, self.endpoints)
        return logits[:n]
```

**2. cuDNN RNN backward requires training mode (discovered mid-sweep)**

C&W L2 completed but reported 0% ASR on every batch. The root cause: PyTorch's cuDNN backend requires `model.train()` during RNN backward passes. The original code called `model.eval()` inside `forward()` — a standard inference pattern — which restores eval mode before backward runs, silently zeroing the gradients. Fix: keep `train()` throughout the entire attack sweep, never restoring `eval()`.

**3. Binary AutoAttack (warning buried in 10,000 lines of output)**

AutoAttack's standard configuration includes DLR loss and FAB-T, which require ≥3 classes. On a binary classifier these components issue a one-line warning — easily missed — and produce undefined results. Fix:

```python
adversary = AutoAttack(wrapper, norm="Linf", eps=epsilon,
                       version="custom",
                       attacks_to_run=["apgd-ce", "square"],
                       verbose=False)
```

After fixing all three, I added a partial save after each epsilon and a `RESUME_RUN_ID` flag so future failures could continue from the last completed checkpoint rather than restarting from zero.

---

## Results

Six attacks evaluated: KAPPA (mine), AutoAttack (APGD-CE + Square — current state of the art), APGD-CE, PGD-40, PGD-500 (budget-matched to KAPPA), and C&W L2. Metric: **True ASR** — fraction of Male subjects (n=84) whose prediction flips to Female. This targeted metric avoids inflating rates with trivially adversarial examples.

### ECG CNN (κ ≈ 1) — Baseline Validation

| Method | ε | True ASR |
|---|---|---|
| PGD | 10 | **86.5%** |
| **KAPPA** | 10 | 72.9% |
| PGD | 2 | 24.0% |
| **KAPPA** | 2 | 21.9% |

On the BN-normalized ECG model, **PGD outperforms KAPPA**. At ε=2, both are equivalent; at ε=10, KAPPA is 13.6 pp worse. This is exactly what the κ ≈ 1 prediction requires: when the loss surface is well-conditioned, the Newton direction adds no information, and CG overhead makes KAPPA strictly less efficient. The baseline holds — KAPPA does not claim universal superiority.

> *Figure 2 — Confusion matrices: clean model (left), after KAPPA attack (center), after PGD attack (right) on the fMRI test set at ε=0.001. KAPPA flips 51/84 Male subjects; PGD flips 11.*

![confusion matrices](figures/confusion_matrix_clean.png)

### STAGIN fMRI (κ = 178,695) — Main Result

| Attack | ε=0.001 | ε=0.005 | ε=0.01 | ε=0.05 | ε=0.1 |
|---|---|---|---|---|---|
| **KAPPA** | **60.7%** | **58.5%** | **50.0%** | **92.7%** | **93.9%** |
| APGD-CE | 22.6% | 30.5% | 29.3% | 86.6% | 74.4% |
| PGD-40 | 31.0% | 35.4% | 41.5% | 54.9% | 53.7% |
| AutoAttack | 17.9% | 15.9% | 28.1% | 62.2% | 65.9% |
| C&W L2 | 16.7% | 18.3% | 18.3% | 18.3% | 18.3% |
| PGD-500 | 13.1% | 29.3% | 42.7% | 51.2% | 53.7% |

> *Figure 3 — True ASR vs epsilon for all six attacks on STAGIN. KAPPA (solid line) consistently sits above the first-order family, with the gap most pronounced at tight epsilon budgets.*

![asr_vs_epsilon](figures/asr_vs_epsilon.png)

At ε=0.001 — a perturbation imperceptible to preprocessing pipelines:

| Comparison | KAPPA | Competitor | Gap |
|---|---|---|---|
| vs AutoAttack (SOTA) | 60.7% | 17.9% | **3.4×** |
| vs APGD-CE (SOTA component) | 60.7% | 22.6% | **2.7×** |
| vs PGD-500 (budget-matched) | 60.7% | 13.1% | **4.6×** |
| Wall-clock time | 741s | 3,771s (PGD-500) | KAPPA is **5× faster** |

**PGD-500 is worse than PGD-40** (13.1% vs 31.0%) at this epsilon. With κ=178,695, more gradient iterations amplify oscillation; the model appears robust under any first-order evaluation simply because the attacks cannot navigate its loss surface.

Three patterns across the sweep:
1. **Tight budgets (ε≤0.005):** KAPPA leads AutoAttack by 2.7–3.7×. This is the clinically relevant regime: perturbations small enough to evade human inspection.
2. **Mid-range (ε=0.01):** First-order methods partially recover. APGD-CE reaches 29.3% vs KAPPA's 50% — a 1.7× gap — as the larger epsilon ball gives gradient methods more room.
3. **Large epsilon (ε=0.05–0.1):** KAPPA saturates at 93%+. First-order attacks plateau near 53–66%, unable to reach the adversarial region even with an unconstrained budget.

C&W L2 flatlines at ~18% across all epsilons — STAGIN's vulnerability is structurally aligned with L∞ directions, not L2.

---

## What This Means for Medical AI Robustness

AutoAttack was designed to be the hardest practical first-order benchmark. On STAGIN at ε=0.001, it reports 17.9%. KAPPA reports 60.7%. **A model evaluated as having moderate vulnerability under the current gold standard has 3.4× greater true vulnerability.**

This is not a failure of AutoAttack. It is a consequence of architecture. The adversarial ML literature has been built almost entirely on CNN architectures with Batch Normalization — ResNets, VGGs, EfficientNets, the CIFAR-10 and ImageNet benchmarks. BN keeps κ near 1, making every first-order attack a fair evaluator. The assumption that gradient-based attacks are sufficient has been invisible because benchmark architectures happened to satisfy it.

Medical AI operates in a different regime. Graph Neural Networks for functional connectivity, Transformers for EHR sequences, RNNs for physiological time series, attention models for histopathology — none of these routinely use the aggressive normalization of image classifiers. For any of these architectures, a robustness certificate from AutoAttack may be systematically optimistic.

The κ estimate — computed cheaply via a few Rayleigh quotient power iterations before running any attack — can serve as a practical diagnostic:

> **If κ ≫ 1, include KAPPA in your robustness evaluation. If κ ≈ 1, AutoAttack suffices.**

---

## Code and Reproducibility

KAPPA is implemented in `hessian.py`, model-agnostic and requiring only a differentiable PyTorch `forward()`. The full evaluation pipeline — model checkpoints, configs, Nebius job scripts, partial-save/resume logic, and `attack_results.json` — is open-source:

**[github.com/diegom4riano/nebius-fmri-adversarial](https://github.com/diegom4riano/nebius-fmri-adversarial)**

Reproduce the full experiment: `make deploy-attack` from a Nebius account with the HCP data in S3.

---

*#NebiusServerlessChallenge — Healthcare & Life Sciences*
