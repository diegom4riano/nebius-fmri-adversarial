# Geometry-Aware Robustness Testing for Clinical AI

> **Nebius Serverless AI Builders Challenge — Healthcare & Life Sciences**

**Blog post:** [Can Clinical AI Prove Its Robustness?](https://medium.com/@diegom4riano/clinical-ai-proves-its-robustness-it-shouldnt-have-e9b9a484561a)

---

Robustness certificates back real clinical decisions — procurement, regulatory clearance, clinical trust. The implicit promise is that the tool used to test the model was adequate *for that model*. That promise is shakier than it looks for medical AI: standard robustness toolkits were built and tuned on image classifiers, while clinical AI uses fundamentally different architectures — graph networks for brain connectivity, recurrent networks for physiological time series — with different input constraints and different failure modes. A test designed for vision can silently mis-measure a clinical model in either direction: **false confidence** (a vulnerable model gets a clean bill of health) or a **false alarm** (a safe model looks catastrophically vulnerable).

This project built three things to probe that gap: (1) **KAPPA**, a curvature-aware second-order attack; (2) a **reproducible evaluation pipeline** on Nebius Serverless AI H200s running KAPPA and five baselines against two clinical models at scale; and (3) a **geometry profiler**, a cheap triage tool that classifies each input's loss-surface terrain before any attack runs and predicts attack outcome without executing one.

---

**Central finding (two-part):**

1. **Geometry may predicts vulnerability**: the local loss-surface geometry of a clinical input — computable from ~k HVP operations before running any attack — reliably identifies inputs that are robust to *any* gradient-based attack (`flat_masked` → 100% robust in both models tested, χ² p ≤ 0.001).

2. **Architecture can shape attack selection**: BatchNorm CNNs (ECG) and GRU graph networks (STAGIN) differ in curvature *magnitude* (σ_max ~3,500×), not anisotropy (κ(H+λI@0.1) = 3.54 vs 1.0003). Newton-CG attacks exploit anisotropy, not magnitude — which explains why KAPPA does not outperform PGD on these models, and identifies the precise condition a model must satisfy for second-order attacks to have structural advantage.

---

## The Framework: Loss-Surface Triage

Clinical robustness evaluation typically applies a single attack uniformly to all inputs. But clinical inputs have heterogeneous loss-surface geometry — some are flat (immune to gradient-based attacks), others curved (vulnerable), and among the curved, some are isotropic (first-order attacks are optimal) vs anisotropic (Newton methods have structural advantage). Applying one attack to all of them either over-tests safe inputs or under-tests vulnerable ones.

```
input clínico → [PROFILER geométrico] → geometry class → [ROUTER] → attack / certificate
                (~k HVPs, before any attack)              (rule or learned)
```

**Profiler features** (single-pass Lanczos, ~k HVPs/subject):

| Feature | Cost | What it reveals |
|---|---|---|
| ‖∇‖ | 1 backward | first-order signal / gradient masking detector |
| σ_max | Lanczos | curvature magnitude |
| μ_min | Lanczos | non-convexity / saddle structure |
| κ(H+λI) | Lanczos | **anisotropy** — what Newton exploits |

**Geometry taxonomy and routing:**

| Class | Signature | Routed Attack |
|---|---|---|
| **flat-masked** | ‖∇‖≈0, σ_max≈0 | black-box (Square/SPSA) or skip |
| **flat-with-gradient** | ‖∇‖>0, σ_max≈0 | PGD/APGD (1st order sufficient) |
| **isotropic-curved** | σ_max high, κ≈1 | PGD large-step (Newton adds nothing) |
| **anisotropic** | σ_max high, κ>>1 | **KAPPA** (Newton-CG + min-PD damping) |
| **saddle** | μ_min<<0 | negative-curvature step |

### MVP Validation

Per-subject geometry profiled on 74 STAGIN + 85 ECG subjects (~30 HVPs/subject, single-pass Lanczos) and joined to per-subject attack outcomes.

**STAGIN ε=0.001 — geometry class × attack outcome:**

| Class | robust | FO only | KAPPA only | both | robust-rate [Wilson 95%] |
|---|---|---|---|---|---|
| **flat-masked** | 26 | 0 | 0 | 0 | **1.00 [0.87, 1.00]** |
| flat-with-gradient | 9 | 2 | 0 | 0 | 0.82 [0.52, 0.95] |
| isotropic-curved | 3 | 0 | 0 | 0 | 1.00 [0.44, 1.00] |
| anisotropic | 21 | 9 | 0 | 4 | 0.62 [0.45, 0.76] |

χ² p=0.020. **ECG ε=2:** flat-masked 1.00 [0.86, 1.00], χ² p=0.0005 (same pattern, independent dataset).

![Geometry routing scatter](figures/geometry_routing_stagin.png)
![Robustness by geometry class](figures/geometry_robustness_stagin.png)

**`anisotropic → KAPPA` (kappa_only=0):** neither ECG (BatchNorm collapses H to σ_max≈4×10⁻⁵) nor STAGIN (κ=3.54, CG residual 0.16–0.22) inhabits the class where Newton-CG has structural advantage. This is a model-selection constraint, not a falsification of the routing hypothesis — it identifies what a model must look like for KAPPA routing to activate.

---

## What Is KAPPA?

**KAPPA** (κ-**A**daptive **P**roximal **P**erturbation **A**ttack) is a second-order adversarial attack that replaces gradient steps with Newton steps, computed using Conjugate Gradient on Hessian-Vector Products. Unlike PGD and its variants (APGD, AutoAttack), KAPPA uses curvature information to navigate the loss surface.

The implementation in [`hessian.py`](hessian.py) is model-agnostic and requires only a differentiable PyTorch `forward()`.

**Hypothesis:** KAPPA's advantage over first-order attacks is predicted by κ — specifically, in the `anisotropic` geometry class (σ_max high, κ>>1, CG convergent).

**Result on these models:** neither ECG nor STAGIN lives in that class. The geometry analysis correctly predicts this post-hoc: KAPPA does not exceed the strongest first-order baseline at any ε tested.

---

## From Hypothesis to Corrected Result

**Original hypothesis:** KAPPA would add little on the ECG CNN (BatchNorm smooths the loss surface) but expose hidden vulnerability on STAGIN (graph architecture, rank-deficient fMRI inputs).

**What happened first:** the initial run showed a clear KAPPA advantage on STAGIN at small ε — consistent with the hypothesis.

**What went wrong:** holding that result to the standard required for a clinical robustness claim, it dissolved. Four methodological failures inflated the initial gap:

1. **Wrong input range for baselines** — off-the-shelf attacks assumed image-range inputs; STAGIN inputs are correlation matrices. Fixing the range alone erased most of the gap.
2. **No paired statistics** — different patient pools per run made comparisons unreliable. A paired test on a consistent intersection pool showed the "advantage" was within noise.
3. **Single-knob exploration** — KAPPA was tested at one λ value; sweeping all reasonable values as parallel H200 jobs removed the rest of the effect.
4. **Non-deterministic GPU kernels** — pinning seeds was required for a fair comparison.

**Corrected result (full damping sweep, McNemar paired tests):** KAPPA does not beat well-run gradient baselines on either model. On STAGIN it is clearly weaker at every λ tested (p ≤ 0.001). The geometry profiler is the contribution that survives: flat-and-silent inputs are 100% robust to every gradient-based attack; the terrain type predicts attack outcome before any attack runs.

The lesson generalises: clinical data's non-standard structure (correlation matrices, waveform amplitudes) creates silent mismatches with tools built for vision. An evaluation can hand you a confident number that flips under a stricter protocol. The methodology matters more than the attack choice.

---

## Models

| Model | Task | Dataset | Architecture | Test BACC | κ(H+λI) at λ=0.1 | σ_max | Geometry class |
|---|---|---|---|---|---|---|---|
| **STAGIN** | fMRI sex classification | HCP-Rest S1200, n=1,080 | GIN + Self-Attention + GRU | **77.2%** | 3.54 | ~0.133 | anisotropic (moderate κ, poor CG) |
| **ECG CNN** | Rhythm classification | PhysioNet/CinC 2017 | 13-block dilated 1D CNN + BN | 87.5% | 1.0003 | ~3.8×10⁻⁵ | flat (BatchNorm collapses H) |

σ_max differs ~3,500× between models; κ at a common λ is similar. BatchNorm is the mechanism that collapses the ECG Hessian — every convolutional block's output is re-normalized before the loss sees it.

![σ_max vs κ at common λ](figures/conditioning_sigma_vs_kappa.png)

---

## Results

### STAGIN — fMRI · Damping sweep (λ ∈ {0.13…0.50} + adaptive, n≈74 pool subjects)

| Attack | ε=0.001 | ε=0.01 | Note |
|---|---|---|---|
| **KAPPA (best λ)** | **4–5%** | **49–55%** | All λ ≥ 0.13 tested |
| APGD-CE | 14.9% | — | Best first-order at ε=0.001 |
| PGD-40 | — | 70.3% | Best first-order at ε=0.01 |

McNemar paired test: KAPPA < best first-order at all λ, p ≤ 0.001. CG did not converge (residual 0.16–0.22) — STAGIN's moderate κ=3.54 is insufficient for Newton to gain traction. Geometry routing correctly predicts this: STAGIN sits in `anisotropic` (moderate), not the high-κ regime where KAPPA activates.

### ECG CNN — PhysioNet 2017 · Untargeted robust-ASR

![ECG robust-ASR](figures/ecg_robust_asr.png)

| Attack | ε=2 | ε=10 |
|---|---|---|
| **KAPPA (ours)** | **12.5%** | **43.5%** |
| PGD-40 | 24.0% | 86.5% |
| PGD-500 | — | — |
| APGD-CE | — | — |
| AutoAttack | — | — |

KAPPA is the weakest attack at both budgets on the ECG CNN as well.

---

## Infrastructure

```
  Nebius S3 (precision-med-hcp/)
    data/fmri/hcp/roi/roi_timeseries.npy  ← 1,080 subjects · 333 ROIs · 1,200 TRs
    saved_model/best_model_fmri.pth       ← STAGIN checkpoint (BACC=77.2%)
        │ mount at /workspace/data
        ▼
  Nebius AI Job fleet (H200 SXM · 141 GB HBM3e)
    1 baselines job per ε  +  N KAPPA-only jobs per (λ, ε)
    Each job writes output/<run_id>/  ← no collision
        │ results → S3
        ▼
  Local machine
    make download-results  ← output/<run_id>/attack_results.json
    python scripts/analyze_sweep.py output/
```

| Resource | Value |
|---|---|
| GPU | H200 SXM — 141 GB HBM3e |
| Platform | `gpu-h200-sxm` |
| Preset | `1gpu-16vcpu-200gb` |
| Peak VRAM | ~86.9 GB (exceeds A100 80 GB limit) |
| Base image | `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime` |
| Sweep runtime | ~2–3h wall-clock (jobs run in parallel) |
| Total cost | < $100 |

**Why H200?** KAPPA requires double-backward HVPs through STAGIN's GRU. With batch=32, peak VRAM hits ~87 GB — beyond an A100's 80 GB. The H200 (141 GB) is the minimum viable GPU for this experiment.

---

## Reproduce

### Quick validation — smoke test (no GPU, no data, no accounts)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python test_fmri_model.py --smoke-test --smoke-samples 8 --smoke-epsilons 0.05
# Expected: smoke test PASSED
python scripts/profile_geometry.py --model stagin --smoke-test
# Expected: geometry records for synthetic subjects (grad_norm, sigma_max, kappa_at_lambda)
```

### Single job on Nebius H200

Prerequisites: Nebius account, nebius CLI, AWS CLI, `.env` with `PARENT_ID`/`BUCKET_ID`/`S3_BUCKET`.

```bash
make upload-data       # sync preprocessed inputs + checkpoints to S3 (once)
make deploy-attack     # launch H200 job; results stream to S3
make download-results  # pull results when job completes
```

### Full damping sweep on Nebius H200

Prerequisites: same as above.

```bash
# Preview (no submission)
DRY_RUN=1 bash scripts/deploy_sweep.sh

# Submit fleet (1 baselines job + N KAPPA-only jobs per ε)
bash scripts/deploy_sweep.sh

# Monitor until all jobs reach a terminal state
bash scripts/monitor_fleet.sh

# Download results
make download-results

# Analyze: McNemar per λ per ε, paired on intersection pool
python scripts/analyze_sweep.py output/
```

**Resume a failed job**

```bash
make deploy-attack RESUME_RUN_ID=<previous_run_id>
```

**Reproduce figures from existing results**

```bash
python generate_figures.py
# figures/stagin_damping_sweep.png
# figures/ecg_robust_asr.png
# figures/conditioning_sigma_vs_kappa.png

python scripts/geometry_routing.py output/
# figures/geometry_routing_stagin.png
# figures/geometry_robustness_stagin.png
# figures/geometry_routing_ecg.png
# figures/geometry_robustness_ecg.png
```

### Validate HVP and Hessian spectrum

```bash
# Smoke (CPU, synthetic data)
python scripts/hvp_validation.py --smoke-test

# Full (GPU, real checkpoint)
python scripts/hvp_validation.py --config configs/config.yaml --n-inputs 8
```

---

## Repository Structure

```
├── hessian.py              KAPPA + PGD implementations (core, model-agnostic)
├── test_fmri_model.py      Full adversarial evaluation sweep (STAGIN)
├── test_pytorch_model.py   ECG CNN evaluation
├── train_fmri.py           STAGIN training (OneCycleLR, early stopping)
├── generate_figures.py     Reproduce all result figures
├── Makefile                Nebius job orchestration (upload-data / deploy-attack / download-results / sweep)
├── Dockerfile              H200 container image (pytorch 2.2.2 + cuda 12.1)
├── .env.template           Required env vars: PARENT_ID, BUCKET_ID, S3_BUCKET
├── model/
│   ├── STAGIN.py           Spatio-Temporal Attention GIN (Kim & Ye, NeurIPS 2021)
│   └── CNN.py              Han et al. dilated 1D CNN (ECG)
├── utils/
│   ├── fMRILoader.py       HCP fMRI loader (sliding-window FC matrices)
│   └── DataLoader.py       ECG loader
├── scripts/
│   ├── profile_geometry.py     Per-subject geometry profiler (‖∇‖, σ_max, κ — ~k HVPs/subject)
│   ├── geometry_routing.py     Join geometry↔outcomes, separability analysis, routing figures
│   ├── paired_stats.py         McNemar + Wilson CI + bootstrap from per_subject JSON
│   ├── analyze_sweep.py        Damping sweep analysis (intersection pool, paired stats per λ)
│   ├── hvp_validation.py       HVP autodiff vs FD, κ multi-input, PD-check
│   ├── deploy_sweep.sh         Fan-out: one Nebius job per (λ, ε)
│   └── monitor_fleet.sh        Poll job states until all terminal
├── configs/
│   ├── config.yaml             Default attack + training hyperparameters
│   ├── config_ecg.yaml         ECG-specific config
│   └── config_damping_sweep.yaml  Damping sweep config
├── tests/                  Unit tests for KAPPA, HVP, and attack constraints
├── saved_model/            Pre-trained checkpoints (STAGIN BACC=77.2%)
├── output/                 Per-run results (attack_results.json, geometry_*.json per run)
├── figures/                Generated result plots (PNG)
└── requirements.txt
```

---

## Citation

KAPPA and Geometry Profiler method: *manuscript in preparation*

STAGIN: Kim et al., [*Understanding Graph Isomorphism Network for rs-fMRI Functional Connectivity Analysis*](https://arxiv.org/abs/2111.01543), NeurIPS 2021  
AutoAttack: Croce & Hein, [*Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks*](https://arxiv.org/abs/2003.01690), ICML 2020  
HCP dataset: Van Essen et al., [*The WU-Minn Human Connectome Project*](https://doi.org/10.1016/j.neuroimage.2013.05.041), NeuroImage 2013  
ECG CNN: Han et al., [*Deep learning models for electrocardiograms are susceptible to adversarial attack*](https://doi.org/10.1038/s41591-020-0791-x), Nature Medicine 26(3):360–363, 2020

## License

MIT
