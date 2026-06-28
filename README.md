# KAPPA: Second-Order Adversarial Attacks on Clinical Neural Networks

> **Nebius Serverless AI Builders Challenge — Healthcare & Life Sciences**

📝 **Blog post:** [Read on Medium](#) — full problem / method / results write-up *(link after publish)*  
🔎 **Proof of execution:** see [Results](#results) and [output/attack_results.json](output/attack_results.json) — real run on Nebius H200, job ID `aijob-e00b1w63p1e576vgxc`  
🚀 **Reproduce:** `make deploy-attack` from a Nebius account with HCP data in S3

---

**Central finding:** AutoAttack — the current gold standard for adversarial robustness evaluation — reports **17.9% attack success rate** on a clinical fMRI model. **KAPPA, the second-order attack developed in this project, reports 60.7%** — a 3.4× gap explained by the Hessian condition number κ = 178,695.

![KAPPA vs all attacks across epsilon](figures/asr_vs_epsilon_kappa.png)

---

## What Is KAPPA?

**KAPPA** (κ-**A**ware **P**erturbation via **P**roximal **A**pproximation) is a second-order adversarial attack that replaces gradient steps with Newton steps, computed using Conjugate Gradient on Hessian-Vector Products. Unlike PGD and its variants (APGD, AutoAttack), KAPPA uses curvature information and is therefore effective on ill-conditioned loss surfaces where gradient direction alone is misleading.

The method will be described in full in an upcoming paper. The implementation in [`hessian.py`](hessian.py) is model-agnostic and requires only a differentiable PyTorch `forward()`.

**Hypothesis:** KAPPA's advantage over first-order attacks is predicted by the Hessian condition number κ.
- κ ≈ 1 (well-conditioned, e.g. BN-normalized CNNs): KAPPA ≈ PGD. No advantage.
- κ ≫ 1 (ill-conditioned, e.g. GNNs with incomplete normalization like STAGIN): KAPPA >> all first-order attacks.

---

## Models

| Model | Task | Dataset | Architecture | Test BACC | κ |
|---|---|---|---|---|---|
| **STAGIN** | fMRI sex classification | HCP-Rest S1200, n=1,080 | GIN + Self-Attention + GRU | **77.2%** | **178,695** |
| **ECG CNN** | Rhythm classification | PhysioNet/CinC 2017 | 13-block dilated 1D CNN + BN | 87.5% | ≈ 1 |

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

### ECG CNN — PhysioNet 2017 (κ ≈ 1) · Control experiment

| Method | ε | True ASR |
|---|---|---|
| PGD-40 | 10 | **86.5%** |
| KAPPA | 10 | 72.9% |
| PGD-40 | 2 | 24.0% |
| KAPPA | 2 | 21.9% |

On the BN-normalized ECG model, PGD outperforms KAPPA — as predicted by κ ≈ 1. The baseline validates the hypothesis.

---

## Infrastructure

```
  AWS S3 (hcp-openaccess)
        │ stream 438 MB/subject (×1,080)
        ▼
  Nebius VM (H200, setup once)
    scripts/extract_roi_timeseries.py  ← CIFTI → 333 Gordon ROIs
    scripts/precompute_fc.py           ← ROI → FC matrices (51×333×333)
        │ sync ~24 GB
        ▼
  Nebius S3 (precision-med-hcp/)
    data/fmri/hcp/roi/fc/       ← 1,080 FC matrix files
    saved_model/                ← STAGIN checkpoint (BACC=77.2%)
        │ mount at /workspace/data
        ▼
  Nebius AI Job (H200 SXM · 141 GB HBM3e)
    test_fmri_model.py          ← 6 attacks × 5 ε × 216 subjects
    partial save after each ε   ← resume-safe
        │ results → S3
        ▼
  Local machine
    make download-results       ← output/attack_results.json
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

### Path A — Re-run the attack only (~10 min setup)

The model checkpoint and preprocessed FC matrices are already in Nebius S3. This path re-runs the full H200 attack sweep against the same data.

**Prerequisites**

```bash
# Nebius CLI
curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
exec -l $SHELL
nebius auth login

# AWS CLI (for S3 sync)
brew install awscli          # macOS; or: sudo apt install awscli

# Configure Nebius S3 profile
aws configure --profile nebius
# AWS Access Key ID:     <Nebius SA static key>
# AWS Secret Access Key: <Nebius SA secret>
# Default region:        eu-north1
```

**Configure and deploy**

```bash
cp .env.template .env
# Fill in: PARENT_ID, BUCKET_ID, S3_BUCKET, S3_ENDPOINT
# (Nebius console: Compute → project ID; Storage → bucket ID)

make deploy-attack      # uploads code, submits H200 job
make logs               # tail live output
make download-results   # fetch output/attack_results.json when done
```

**Resume a failed job**

```bash
make deploy-attack RESUME_RUN_ID=<previous_run_id>
# Reloads partial JSON from S3, skips completed epsilons
```

**Smoke test (no data or GPU needed)**

```bash
pip install -r requirements.txt
python test_fmri_model.py --smoke-test --smoke-samples 8 --smoke-epsilons 0.05
# Expected: smoke test PASSED — KAPPA and PGD ran without errors
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

### Path B — Full reproduction from scratch (~2–3 days)

This path reproduces everything: HCP preprocessing, STAGIN training, and the attack sweep. The FC matrices (~24 GB on disk) must be generated on a VM — a local laptop lacks both bandwidth and storage for the 438 MB/subject HCP downloads.

#### 1. Get HCP data access

1. Register at [db.humanconnectome.org](https://db.humanconnectome.org)
2. Accept the **WU-Minn HCP Data Use Terms**
3. Go to **Amazon S3 Access → Get AWS Credentials** (temporary, expire in a few hours)

#### 2. Provision a Nebius VM for preprocessing

Create a VM in the Nebius console (**Compute → Virtual Machines → Create**):

| Setting | Value |
|---|---|
| GPU | NVIDIA H200 NVLink |
| RAM | 196 GiB |
| Disk | 500 GiB NVMe |
| OS | Ubuntu 24.04 LTS (CUDA pre-installed) |
| Estimated cost | ~$3.55/h |

Also create a **Nebius S3 bucket** (Storage → Object Storage) and a **service account** with `storage.editor` role to generate static access keys.

#### 3. Set up the VM

```bash
ssh <user>@<VM_IP>

sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-venv awscli git

git clone https://github.com/diegom4riano/nebius-fmri-adversarial /opt/kappa
cd /opt/kappa
pip install -r requirements.txt

# Configure AWS profiles on the VM (credentials stay on VM only)
aws configure --profile hcp       # HCP key + secret + region us-east-1
aws configure set aws_session_token <TOKEN> --profile hcp

aws configure --profile nebius    # Nebius SA key + secret + region eu-north1
```

> HCP credentials expire in a few hours — regenerate at ConnectomeDB if you see `AccessDenied`.

#### 4. Preprocess HCP data (~6–8h)

```bash
cd /opt/kappa

# 4a. Extract 333 Gordon ROI timeseries from CIFTI files
#     Streams rfMRI_REST1_LR_hp2000_clean.dtseries.nii (~438 MB/subject) from HCP S3,
#     extracts ROIs, z-scores, deletes the .nii to stay within disk budget.
nohup python scripts/extract_roi_timeseries.py \
  --subjects data/HCP_YA_subjects_2026_04_26_22_26_40.csv \
  --out-dir data/fmri/hcp/roi \
  > logs/extract.log 2>&1 &

tail -f logs/extract.log   # monitor progress

# 4b. Precompute sliding-window FC matrices (~2 min, 16 workers)
#     Generates 1,080 × fc_{i:04d}.npy shape (51, 333, 333) — ~24 GB total
python scripts/precompute_fc.py --workers 16

# Verify
python - <<'EOF'
import numpy as np
roi = np.load('data/fmri/hcp/roi/roi_timeseries.npy')
fc  = np.load('data/fmri/hcp/roi/fc/fc_0000.npy')
print(roi.shape, fc.shape)   # (1080, 333, 1200)   (51, 333, 333)
EOF

# 4c. Sync FC matrices to Nebius S3
aws s3 sync data/fmri/hcp/roi/fc/ s3://<S3_BUCKET>/data/fmri/hcp/roi/fc/ \
  --profile nebius --endpoint-url https://storage.eu-north1.nebius.cloud
```

#### 5. Upload and run the attack

```bash
# From local machine
make upload-data        # sync saved_model/ and data CSV to Nebius S3
make deploy-attack      # submit H200 job (code uploaded automatically)
make logs
make download-results
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

---

## License

MIT
