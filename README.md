# Can Clinical AI Prove Its Robustness?

> **Nebius Serverless AI Builders Challenge — Healthcare & Life Sciences**

**Blog post:** [Can Clinical AI Prove Its Robustness?](https://medium.com/@diegom4riano/clinical-ai-proves-its-robustness-it-shouldnt-have-e9b9a484561a)

---

Before a hospital deploys an AI that reads a brain scan, or a regulator authorizes an ECG
classifier, someone must certify how robust that model is. A wave of methods proposes using the
*geometry* of a model's decision surface to do this. **This
project asks whether that sophisticated geometry actually tells you anything a simple robustness
check doesn't, on real clinical models and whether the standard tooling, built for photo
classifiers, even measures clinical models correctly.** Across three clinical AI systems
(brain-imaging, cardiac-signal, cancer-genomics), the sophisticated geometry adds nothing a simple
check misses, and the off-the-shelf toolkit can quietly return the wrong answer.

## Contribution

1. **No hidden vulnerability from "smarter" attacks.** The curvature-aware attack proposed to find
   weaknesses that simple attacks miss ([`hessian.py`](hessian.py)) was run head-to-head against
   standard attacks on all three models. It never found a weakness the simple attack missed — on the
   most curved model it attacked the *exact same patients*. **Practical takeaway: a well-run simple
   robustness test is sufficient for these clinical models.**
2. **The standard robustness toolkit misfires on clinical models — a safety issue.** Built for image
   classifiers, it mis-measures brain-connectivity graph networks and ECG networks in *both*
   directions: **false confidence** (a fragile model looks safe) and **false alarm** (a safe model
   looks broken). Certifying clinical AI with vision tools is not safe by default.
3. **First curvature characterization of real clinical AI** (brain imaging, cardiac signal, cancer
   genomics), with an actionable engineering finding: **BatchNorm flattens the decision surface** — a
   model built with it will look "smooth" regardless, which matters when choosing architectures for
   certifiable clinical AI.
4. **A cheap pre-screening tool** that flags, before any expensive test, which inputs are trivially
   robust (100% robust to every attack, both models, tight confidence intervals) — useful in a
   deployment/monitoring pipeline to skip needless testing. *Honest scope:* the flag recovers a known
   result (curvature relates to robustness), so it is an operational convenience, not a new predictor.
5. **A rigorous, reusable evaluation protocol** — the methodological backbone: honest held-out
   validation that avoids the self-fulfilling comparisons which make geometry claims look better than
   they are. This is what let the project catch and discard its own initial (wrong) positive result.

## Models

Three clinical models, chosen to span domains, architectures, and decision-surface conditioning:

| Model | Clinical task (data) | Architecture | Accuracy | Decision-surface conditioning |
|---|---|---|---|---|
| **STAGIN** | sex from brain fMRI (HCP-Rest S1200, n=1,080) | graph net + attention + GRU | 77.2% BACC | mildly stretched (κ≈3.5) |
| **ECG CNN** | heart-rhythm, 4-class (PhysioNet/CinC 2017) | dilated 1D-CNN + BatchNorm | 87.5% BACC | flat — BatchNorm smooths it (κ≈1) |
| **MaxNet** | glioma grade from genomics (TCGA-GBMLGG) | 4-layer network, no BatchNorm | — | strongly ill-conditioned (κ up to ~2260) |

*"Conditioning" (κ) = how stretched/anisotropic the decision surface is — higher is the regime where
sophisticated second-order geometry* should *help most. Only MaxNet reaches that regime, and even
there it didn't help.*

## Key results

**The sophisticated attack found nothing extra.** On the brain (STAGIN) and cardiac (ECG) models the
curvature-aware attack was *weaker* than standard attacks. On the genomics model (MaxNet) — the one in
the extreme-conditioning regime where it *should* win — a fully corrected version matched the standard
attack exactly, flipping the identical set of patients. Sophisticated geometry adds no attack power
for these clinical models.

**The pre-screening tool works — but it only recovers curvature.** Inputs the profiler labels "flat"
are 100% robust to every attack (both models, tight confidence intervals) — a reliable cheap
skip-the-test flag. But a rigorous held-out test showed the tool's terrain labels never beat a single
curvature number — it recovers known theory rather than adding new predictive power.

![Geometry routing](figures/geometry_routing_stagin.png)
![Robustness by geometry class](figures/geometry_robustness_stagin.png)
![σ_max vs κ at common λ](figures/conditioning_sigma_vs_kappa.png)

---

## Reproduce

Prerequisites for Nebius jobs: Nebius account, nebius CLI, AWS CLI, `.env` filled from `.env.template` (`PARENT_ID`, `BUCKET_ID`, `S3_BUCKET`).

### 0. Setup

Requires Python 3.11+. Local steps (smoke tests, figure generation) run on CPU. Cloud steps require a Nebius account with H200 SXM access.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.template .env   # fill in PARENT_ID, BUCKET_ID, S3_BUCKET, S3_ENDPOINT
```

### 1. Smoke test — local, no GPU, no accounts

```bash
python test_fmri_model.py --smoke-test --smoke-samples 8 --smoke-epsilons 0.05
# Expected: smoke test PASSED

python scripts/profile_geometry.py --model stagin --smoke-test
# Expected: geometry records for synthetic subjects (grad_norm, sigma_max, kappa_at_lambda)
```

### 2. Data access

The preprocessed fMRI timeseries (1.6 GB, HCP-Rest S1200) and model checkpoints are already hosted on a Nebius S3 bucket. To request read access, contact **diegocampos.br at gmail**.

Once you have credentials:
```bash
make upload-data          # fMRI timeseries + model checkpoints (if replicating your own bucket)
make upload-data-ecg      # ECG waveforms (separate, lighter)
```

### 3. STAGIN adversarial attack (KAPPA + baselines)

**Runtime:** ~10h on H200 (5 ε values × all attacks, sequential; dominated by PGD-500 and AutoAttack). **Output:** `output/<run_id>/attack_results.json` — per-subject attack outcomes for all attacks × all ε values.

```bash
make deploy-attack        # launch H200 job; results stream to S3
make logs                 # follow job output live
make download-results     # pull output/ from S3 when done
```

Resume a failed job:
```bash
make deploy-attack RESUME_RUN_ID=<previous_run_id>
```

### 4. ECG adversarial attack

**Runtime:** ~30 min on H200 (2 ε values × all attacks). **Output:** `output/<run_id>/attack_results_ecg.json`.

```bash
make deploy-attack-ecg
make logs-ecg
make download-results
```

### 5. HVP validation (verify KAPPA math)

```bash
make deploy-kappa-validation   # autodiff HVP vs finite-diff, κ distribution, PD-check
make logs-job JOB=kappa-validation
make download-results
```

Local smoke (CPU, synthetic data):
```bash
python scripts/hvp_validation.py --smoke-test
```

### 6. Full damping sweep (KAPPA vs baselines across all λ)

**Runtime:** ~2–3h wall-clock (all jobs run in parallel). **Output:** one `output/<run_id>/attack_results.json` per job; `analyze_sweep.py` prints McNemar p-values and Wilson CIs per (λ, ε) to stdout.

```bash
make deploy-sweep-dry          # preview: list jobs to submit, no submission
make deploy-sweep              # fan-out: 1 baselines job + N KAPPA jobs per (λ, ε)
make status                    # fleet status for all jobs
make download-results          # pull all run directories from S3

# runs locally after download:
python scripts/analyze_sweep.py output/   # McNemar per λ per ε, paired intersection pool
```

### 7. Figures

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

---

## Infrastructure

```
  Nebius S3 (precision-med-hcp/)
    data/fmri/hcp/roi/roi_timeseries.npy  ← 1,080 subjects · 333 ROIs · 1,200 TRs
    saved_model/best_model_fmri.pth       ← STAGIN checkpoint (BACC=77.2%)
        │ volume-mounted at /workspace/data
        ▼
  Nebius H200 SXM (141 GB HBM3e) — Serverless AI
    1 baselines job per ε  +  N KAPPA jobs per (λ, ε)
    each job writes output/<run_id>/  ← no collision; partial results survive failure
        │ → S3
        ▼
  Local
    make download-results → output/<run_id>/attack_results.json
```

| Resource | Value |
|---|---|
| GPU | H200 SXM — 141 GB HBM3e |
| Platform | `gpu-h200-sxm` · preset `1gpu-16vcpu-200gb` |
| Peak VRAM | ~87 GB (KAPPA double-backward through GRU) |
| Total Runtime | ~13h wall-clock (sweep jobs run in parallel) |
| Total cost | ~$100 |

---

## Repository Structure

```
├── hessian.py              KAPPA + PGD implementations (core, model-agnostic)
├── test_fmri_model.py      Full adversarial evaluation sweep (STAGIN)
├── test_pytorch_model.py   ECG CNN evaluation
├── test_pathomic_model.py  MaxNet (genomic SNN) evaluation — the κ≫1 model
├── test_pathomic_qp.py     L∞ box-QP ablation (second-order steelman: QP == PGD)
├── train_fmri.py           STAGIN training (OneCycleLR, early stopping)
├── generate_figures.py     Reproduce all result figures
├── Makefile                Nebius job orchestration (see targets above)
├── Dockerfile              H200 container image (pytorch 2.2.2 + cuda 12.1)
├── .env.template           Required env vars: PARENT_ID, BUCKET_ID, S3_BUCKET
├── model/
│   ├── STAGIN.py           Spatio-Temporal Attention GIN (Kim & Ye, NeurIPS 2021)
│   └── CNN.py              Han et al. dilated 1D CNN (ECG)
├── utils/
│   ├── fMRILoader.py       HCP fMRI loader (sliding-window FC matrices)
│   └── DataLoader.py       ECG loader
├── scripts/
│   ├── profile_geometry.py           Per-subject geometry profiler (‖∇‖, σ_max, κ)
│   ├── profile_geometry_pathomic.py  MaxNet geometry profiler
│   ├── geometry_routing.py           Join geometry↔outcomes, routing figures
│   ├── paired_stats.py               McNemar + Wilson CI + bootstrap
│   ├── analyze_sweep.py              Damping sweep analysis (intersection pool)
│   ├── hvp_validation.py             HVP autodiff vs FD, κ multi-input, PD-check
│   ├── deploy_sweep.sh               Fan-out: one Nebius job per (λ, ε)
│   └── monitor_fleet.sh              Poll job states until all terminal
├── configs/
│   ├── config.yaml             Default STAGIN config
│   ├── config_ecg.yaml         ECG-specific config
│   └── config_damping_sweep.yaml  Damping sweep config
├── output/                 Per-run results (downloaded via make download-results)
├── figures/                Generated result plots
└── requirements.txt
```

---

## Citation and References

KAPPA and Geometry Profiler: *manuscript in preparation*

STAGIN: Kim et al., [*Understanding Graph Isomorphism Network for rs-fMRI Functional Connectivity Analysis*](https://arxiv.org/abs/2111.01543), NeurIPS 2021  
AutoAttack: Croce & Hein, [*Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks*](https://arxiv.org/abs/2003.01690), ICML 2020  
HCP dataset: Van Essen et al., [*The WU-Minn Human Connectome Project*](https://doi.org/10.1016/j.neuroimage.2013.05.041), NeuroImage 2013  
ECG CNN: Han et al., [*Deep learning models for electrocardiograms are susceptible to adversarial attack*](https://doi.org/10.1038/s41591-020-0791-x), Nature Medicine 26(3):360–363, 2020

## License

MIT
