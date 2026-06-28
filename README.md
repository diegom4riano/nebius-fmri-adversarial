# Second-Order Adversarial Attacks on Clinical Neural Networks

> **Nebius Serverless AI Builders Challenge** — Healthcare & Life Sciences

## Overview

This project investigates adversarial vulnerabilities in clinical neural networks using **second-order (Newton-CG) attacks** compared to first-order baselines (PGD, AutoAttack, APGD-CE, C&W L2).

**Central finding:** The advantage of Newton-CG over PGD is mediated by the condition number κ of the target model's loss Hessian. On a STAGIN model trained on fMRI functional connectivity matrices (κ ≫ 1), Newton-CG achieves **+25 percentage-point higher True ASR** at ε = 0.001 compared to PGD-40 (equal compute budget).

## Models

| Model | Task | Dataset | Balanced Acc |
|---|---|---|---|
| **STAGIN** | Sex classification | HCP fMRI (ROI) | 77.2% |
| **CNN** | Arrhythmia detection | ECG | — |

## Attacks

| Attack | Type | Source | Notes |
|---|---|---|---|
| Newton-CG | 2nd-order targeted | `hessian.py` | 5 outer steps × 50 CG iters |
| PGD-40 | 1st-order | `hessian.py` | Standard baseline |
| PGD-500 | 1st-order | `hessian.py` | Matched compute budget (500 backward passes) |
| AutoAttack | Ensemble L∞ | `autoattack` | APGD-CE + APGD-DLR + FAB + Square |
| APGD-CE | Adaptive L∞ | `torchattacks` | Auto-projected gradient descent |
| C&W L2 | Optimization | `torchattacks` | Carlini-Wagner L2 |

## Infrastructure

All jobs run on **Nebius Serverless AI** — no persistent VMs, billed per GPU-second.

| Resource | Value |
|---|---|
| GPU | H200 SXM (80 GB HBM3e) |
| Platform | `gpu-h200-sxm` |
| Preset | `1gpu-16vcpu-200gb` |
| Base image | `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime` |
| Data storage | Nebius Object Storage (S3-compatible) |
| Estimated cost | $32–64 for full 5-epsilon sweep |

## Quick Start

### Prerequisites

```bash
# Install Nebius CLI
curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
exec -l $SHELL

# Configure authentication
nebius profile create

# Configure AWS CLI for Nebius S3
# (set endpoint in ~/.aws/config: endpoint_url = https://storage.eu-north1.nebius.cloud)
```

### Setup

```bash
cp .env.template .env
# Edit .env: fill in BUCKET_ID
#   nebius storage bucket create --parent-id project-e00tza0vpr005zjp61embc --name precision-med-hcp
#   nebius storage bucket get-by-name --name precision-med-hcp \
#     --parent-id project-e00tza0vpr005zjp61embc --format jsonpath='{.metadata.id}'
```

### Local smoke test (no HCP data required)

```bash
pip install -r requirements.txt
python test_fmri_model.py --smoke-test --smoke-samples 8 --smoke-epsilons 0.05
```

### Deploy to Nebius (full run)

```bash
make upload-data          # once: sync HCP fMRI data to S3
make deploy-attack        # submit H200 job
make logs                 # tail job logs
make download-results     # fetch output/*/attack_results.json
```

### Local Docker build (optional)

```bash
docker build -t precision-med .
```

## Output

Results are saved to `output/{RUN_ID}/attack_results.json`:

```json
{
  "clean": {"balanced_accuracy": 0.772, "macro_f1": 0.771},
  "condition_number_kappa": 1847.3,
  "epsilon_results": [
    {
      "epsilon": 0.001,
      "attacks": {
        "newton_cg": {"asr": 0.63, "time_s": 420.5},
        "pgd_40":    {"asr": 0.38, "time_s": 12.1},
        "pgd_500":   {"asr": 0.41, "time_s": 151.3},
        "autoattack":{"asr": 0.44, "time_s": 89.2},
        "apgd_ce":   {"asr": 0.40, "time_s": 31.4},
        "cw_l2":     {"asr": 0.35, "time_s": 28.7}
      }
    }
  ]
}
```

## Repository Structure

```
├── hessian.py            Newton-CG + PGD attack implementations
├── test_fmri_model.py    Adversarial evaluation (all 6 attacks + epsilon sweep)
├── train_fmri.py         STAGIN training with JSON logging
├── train.py              ECG CNN training
├── model/STAGIN.py       Spatio-Temporal Attention Graph Isomorphism Network
├── utils/fMRILoader.py   HCP fMRI data loader
├── saved_model/          Pre-trained checkpoints (BACC=77.2%)
├── configs/config.yaml   Attack and training hyperparameters
├── Makefile              Nebius job orchestration
└── Dockerfile            Local build / challenge requirement
```

## Citation

STAGIN architecture: [Kim et al., NeurIPS 2021](https://arxiv.org/abs/2111.01543)  
AutoAttack: [Croce & Hein, ICML 2020](https://arxiv.org/abs/2003.01690)  
C&W attack: [Carlini & Wagner, IEEE S&P 2017](https://arxiv.org/abs/1608.04644)

## License

MIT
