# KAPPA: Second-Order Adversarial Attacks on Clinical Neural Networks

> **Nebius Serverless AI Builders Challenge — Healthcare & Life Sciences**

**Blog post:** [Read on Medium](#) — full problem / method / results write-up *(link after publish)*  
**Execution:** see [Results](#results) and [output/attack_results.json](output/attack_results.json) — real run on Nebius H200 
**Reproduce:** `make deploy-attack` from a Nebius account with HCP data in S3

---

**Central finding:** AutoAttack — the current gold standard for adversarial robustness evaluation — reports **17.9% attack success rate** on a clinical fMRI model. **KAPPA, the second-order attack developed in this project, reports 60.7%** — a 3.4× gap explained by the Hessian condition number κ = 178,695.

![KAPPA vs all attacks across epsilon](figures/asr_vs_epsilon_kappa.png)

---

## What Is KAPPA?

**KAPPA** (κ-**A**daptive **P**roximal **P**erturbation **A**ttack) is a second-order adversarial attack that replaces gradient steps with Newton steps, computed using Conjugate Gradient on Hessian-Vector Products. Unlike PGD and its variants (APGD, AutoAttack), KAPPA uses curvature information and is therefore effective on ill-conditioned loss surfaces where gradient direction alone is misleading.

The implementation in [`hessian.py`](hessian.py) is model-agnostic and requires only a differentiable PyTorch `forward()`.

**Hypothesis:** KAPPA's advantage over first-order attacks is predicted by the Hessian condition number κ.
- κ ≈ 8,000 (moderately conditioned, e.g. BN-normalized CNNs): KAPPA ≈ PGD. Marginal advantage.
- κ ≈ 180,000 (severely ill-conditioned, e.g. GNNs with incomplete normalization like STAGIN): KAPPA >> all first-order attacks.

---

## Models

| Model | Task | Dataset | Architecture | Test BACC | κ |
|---|---|---|---|---|---|
| **STAGIN** | fMRI sex classification | HCP-Rest S1200, n=1,080 | GIN + Self-Attention + GRU | **77.2%** | **178,695** |
| **ECG CNN** | Rhythm classification | PhysioNet/CinC 2017 | 13-block dilated 1D CNN + BN | 87.5% | ≈ 8,000 |

The two models serve as a controlled experiment: same attack code, architectures that differ only in normalization, opposite results for KAPPA vs PGD.

---

## Results

### STAGIN — fMRI (κ = 178,695) · 82–84 Male test subjects (n varies by ε)

![Bar chart at eps 0.001](figures/bar_attack_eps001.png)

| Attack | ε=0.001 | ε=0.005 | ε=0.01 | ε=0.05 | ε=0.1 | Time @ ε=0.001 |
|---|---|---|---|---|---|---|
| **KAPPA (ours)** | **60.7%** | **58.5%** | **50.0%** | **92.7%** | **93.9%** | 741s |
| APGD-CE | 22.6% | 30.5% | 29.3% | 86.6% | 74.4% | 783s |
| PGD-40 | 31.0% | 35.4% | 41.5% | 54.9% | 53.7% | 308s |
| AutoAttack | 17.9% | 15.9% | 28.1% | 62.2% | 65.9% | 2,762s |
| C&W L2 | 16.7% | 18.3% | 18.3% | 18.3% | 18.3% | 52s |
| PGD-500 | 13.1% | 29.3% | 42.7% | 51.2% | 53.7% | 3,771s |

Full results: [`output/attack_results.json`](output/attack_results.json) · Peak VRAM: 86.9 GB · Job: `aijob-e00b1w63p1e576vgxc`

### ECG CNN — PhysioNet 2017 (κ ≈ 8,000) · Control experiment

| Method | ε | True ASR |
|---|---|---|
| PGD-40 | 10 | **86.5%** |
| KAPPA | 10 | 72.9% |
| PGD-40 | 2 | 24.0% |
| KAPPA | 2 | 21.9% |

On the BN-normalized ECG model, PGD outperforms KAPPA — as predicted by κ ≈ 8,000. The baseline validates the hypothesis.

---

## Infrastructure

```
  Nebius S3 (precision-med-hcp/)
    data/fmri/hcp/roi/roi_timeseries.npy  ← 1,080 subjects · 333 ROIs · 1,200 TRs
    saved_model/best_model_fmri.pth       ← STAGIN checkpoint (BACC=77.2%)
        │ mount at /workspace/data
        ▼
  Nebius AI Job (H200 SXM · 141 GB HBM3e)
    test_fmri_model.py   ← 6 attacks × 5 ε × 216 test subjects
    FC windows computed on-the-fly per batch (no precompute needed)
    partial save after each ε  ← resume-safe
        │ results → S3
        ▼
  Local machine
    make download-results  ← output/attack_results.json
```

| Resource | Value |
|---|---|
| GPU | H200 SXM — 141 GB HBM3e |
| Platform | `gpu-h200-sxm` |
| Preset | `1gpu-16vcpu-200gb` |
| Peak VRAM | 86,876 MB (exceeds A100 80 GB limit) |
| Base image | `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime` |
| Actual runtime | ~10h (6 attacks × 5 epsilons × 216 subjects) |
| Total cost | < $100 |

**Why H200?** KAPPA requires double-backward HVPs through STAGIN's GRU. With batch=32, peak VRAM hits 86.9 GB — beyond an A100's 80 GB. The H200 (141 GB) is the minimum viable GPU for this experiment.

---

## Reproduce

### Quick validation — smoke test (no GPU, no data, no accounts)

```bash
pip install -r requirements.txt
python test_fmri_model.py --smoke-test --smoke-samples 8 --smoke-epsilons 0.05
# Expected: smoke test PASSED — KAPPA and PGD ran without errors
```

---

### Full H200 attack sweep (~10h, ~$100)

The model checkpoint and preprocessed ROI timeseries are already in a shared Nebius S3 bucket accessible with the read-only key below. You only need your own Nebius account to submit the job.

#### Prerequisites

- Nebius account with credits ([console.eu-north1.nebius.cloud](https://console.eu-north1.nebius.cloud) — free trial available)
- nebius CLI and AWS CLI
- ~$100 in Nebius credits for H200 runtime

#### Step 1 — Install CLIs

```bash
# Nebius CLI
curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
exec -l $SHELL
nebius auth login

# AWS CLI
brew install awscli      # macOS
# or: sudo apt install awscli
```

#### Step 2 — Configure read-only access to shared data (Access available only for a short period of time for the challenge)

```bash
aws configure --profile nebius-readonly
# Access Key ID:     NAKI1GQ4OSO9U4Z1M465
# Secret Access Key: bJEWna+EL5PIxfWMHzK+v/lZjuCTpuxgIa4Crxwh
# Default region name:   eu-north1
# Default output format: (leave blank)
```

#### Step 3 — Create your Nebius resources (one-time console setup)

1. [console.eu-north1.nebius.cloud](https://console.eu-north1.nebius.cloud) → **Projects** → Create project → note your **Project ID**
2. **Storage → Object Storage** → Create bucket → note your **bucket name**
3. **IAM → Service Accounts** → Create SA with `storage.editor` role → **Static Keys** → note your **Key ID** and **Secret**

#### Step 4 — Configure your Nebius credentials

```bash
aws configure --profile nebius
# Access Key ID:     <your SA key ID>
# Secret Access Key: <your SA secret>
# Default region name:   eu-north1
# Default output format: (leave blank)
```

#### Step 5 — Configure .env

```bash
cp .env.template .env
# Edit .env: fill in PARENT_ID, BUCKET_ID, S3_BUCKET

# Get BUCKET_ID with:
nebius storage bucket get-by-name \
  --name <your-bucket-name> \
  --parent-id <your-project-id> \
  --format jsonpath='{.metadata.id}'
```

#### Step 6 — Sync shared data to your bucket (~1.8 GB)

```bash
mkdir -p data/fmri/hcp/roi saved_model

aws s3 sync s3://precision-med-hcp/data/ data/ \
  --profile nebius-readonly \
  --endpoint-url https://storage.eu-north1.nebius.cloud

aws s3 sync s3://precision-med-hcp/saved_model/ saved_model/ \
  --profile nebius-readonly \
  --endpoint-url https://storage.eu-north1.nebius.cloud

make upload-data        # upload to your bucket (one-time, ~1.8 GB)
```

#### Step 7 — Deploy and monitor

```bash
make deploy-attack      # uploads code, submits H200 job
make logs               # tail live output
make download-results   # fetch output/<run_id>/attack_results.json when done
```

**Resume a failed job**

```bash
make deploy-attack RESUME_RUN_ID=<previous_run_id>
# Reloads partial results from S3, skips completed epsilons
```

**Reproduce figures from existing results**

```bash
pip install matplotlib numpy
python generate_figures.py
# figures/asr_vs_epsilon_kappa.png
# figures/bar_attack_eps001.png
# figures/kappa_vs_autoattack_gap.png
```

---

## Repository Structure

```
├── hessian.py              KAPPA + PGD implementations (core, model-agnostic)
├── test_fmri_model.py      Full adversarial evaluation sweep (6 attacks × 5 ε)
├── train_fmri.py           STAGIN training (OneCycleLR, early stopping)
├── train.py                ECG CNN training
├── generate_figures.py     Reproduce all result figures from attack_results.json
├── model/
│   ├── STAGIN.py           Spatio-Temporal Attention GIN (Kim & Ye, NeurIPS 2021)
│   └── CNN.py              Han et al. dilated 1D CNN (ECG)
├── utils/
│   ├── fMRILoader.py       HCP fMRI loader (sliding-window FC matrices)
│   └── DataLoader.py       ECG loader
├── scripts/
│   ├── extract_roi_timeseries.py   CIFTI → 333 Gordon ROIs (streams from HCP S3)
│   ├── precompute_fc.py            ROI timeseries → sliding-window FC matrices
│   └── plot_results.py             Additional result plots
├── configs/config.yaml     Attack + training hyperparameters
├── saved_model/            Pre-trained checkpoints (STAGIN BACC=77.2%)
├── output/
│   └── attack_results.json Full 5-epsilon sweep results (real run, H200)
├── figures/                Generated result plots
├── Makefile                Nebius job orchestration (deploy / logs / download)
└── Dockerfile              Container image for Nebius AI Jobs
```

---

## Citation

KAPPA method: *manuscript in preparation (ICLR submission)*

STAGIN: Kim et al., [*Understanding Graph Isomorphism Network for rs-fMRI Functional Connectivity Analysis*](https://arxiv.org/abs/2111.01543), NeurIPS 2021  
AutoAttack: Croce & Hein, [*Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks*](https://arxiv.org/abs/2003.01690), ICML 2020  
C&W: Carlini & Wagner, [*Towards Evaluating the Robustness of Neural Networks*](https://arxiv.org/abs/1608.04644), IEEE S&P 2017  
HCP dataset: Van Essen et al., [*The WU-Minn Human Connectome Project*](https://doi.org/10.1016/j.neuroimage.2013.05.041), NeuroImage 2013  
ECG CNN: Han et al., [*Deep learning models for electrocardiograms are susceptible to adversarial attack*](https://doi.org/10.1038/s41591-020-0791-x), Nature Medicine 26(3):360–363, 2020

## License

MIT