#!/usr/bin/env bash
# Fan-out of the damping sweep on Nebius AI (one job per (λ, ε), unique names).
#
# λ does not affect the baselines → one baselines job per ε + N KAPPA-only jobs per (λ, ε).
# Each job writes its own output/<run_id>/ to S3 (no collision). Monitor with `make status`,
# download with `make download-results`, analyze with scripts/analyze_sweep.py.
#
# DRY_RUN=1 prints the commands without submitting (validates them without spending GPU):
#   DRY_RUN=1 bash scripts/deploy_sweep.sh
# Real run:
#   bash scripts/deploy_sweep.sh
#
# Overridable vars:
#   SWEEP_LAMBDAS="0.13 0.15 0.20 0.30 0.50"   SWEEP_EPS="0.001 0.01"
#   SWEEP_CONFIG=configs/config_damping_sweep.yaml   INCLUDE_ADAPTIVE=1
set -euo pipefail

# .env provides PARENT_ID, BUCKET_ID, S3_BUCKET, S3_ENDPOINT (same as the Makefile)
[ -f .env ] && set -a && . ./.env && set +a

NEBIUS="${NEBIUS:-$(command -v nebius || echo "$HOME/.nebius/bin/nebius")}"
IMAGE="pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime"
CONFIG="${SWEEP_CONFIG:-configs/config_damping_sweep.yaml}"
LAMBDAS="${SWEEP_LAMBDAS:-0.13 0.15 0.20 0.30 0.50}"
EPS="${SWEEP_EPS:-0.001 0.01}"
INCLUDE_ADAPTIVE="${INCLUDE_ADAPTIVE:-1}"
DRY_RUN="${DRY_RUN:-0}"

: "${PARENT_ID:?defina PARENT_ID (via .env)}"
: "${BUCKET_ID:?defina BUCKET_ID (via .env)}"

# Sync the code to S3 once before submitting the fleet.
if [ "$DRY_RUN" = "0" ]; then
  make upload-job-files
fi

_seq=0
submit() {   # $1 = job name ; $2 = python args (after 'python test_fmri_model.py ')
  local name="$1" pyargs="$2"
  local container="apt-get update -qq && apt-get install -y git -q && cd /workspace/data && \
pip install --no-cache-dir -r requirements.txt && python test_fmri_model.py ${pyargs} \
--output-dir /workspace/data/output"
  echo "───── job: ${name}"
  if [ "$DRY_RUN" = "1" ]; then
    echo "    python test_fmri_model.py ${pyargs}"
    return
  fi
  "$NEBIUS" ai job create \
    --parent-id "$PARENT_ID" --name "$name" \
    --image "$IMAGE" --platform gpu-h200-sxm --preset 1gpu-16vcpu-200gb \
    --disk-size 200Gi --shm-size 32Gi \
    --volume "$BUCKET_ID":/workspace/data \
    --container-command bash --args "-c \"${container}\""
}

ts() { date +%Y%m%d_%H%M%S; }

INCLUDE_BASE="${INCLUDE_BASE:-1}"
for eps in $EPS; do
  # 1 job de baselines por ε (independe de λ) — reutilizado por todos os λ na análise.
  # INCLUDE_BASE=0 skips it (reuse baselines from a previous run, e.g. an extra λ arm).
  if [ "$INCLUDE_BASE" = "1" ]; then
    submit "base-e${eps}" \
      "--config ${CONFIG} --only pgd,pgd500,apgd --epsilon ${eps} --run-id $(ts)_base_e${eps}_$((_seq++))"
  fi

  # N jobs KAPPA-only, um por λ (damping fixo).
  for lam in $LAMBDAS; do
    submit "kappa-l${lam}-e${eps}" \
      "--config ${CONFIG} --only kappa --damping-mode fixed --lambda-reg ${lam} --epsilon ${eps} \
--run-id $(ts)_kappa_l${lam}_e${eps}_$((_seq++))"
  done

  # Adaptive arm (λ = |μ_min|+λ_min, chosen automatically via exact Lanczos).
  if [ "$INCLUDE_ADAPTIVE" = "1" ]; then
    submit "kappa-adaptive-e${eps}" \
      "--config ${CONFIG} --only kappa --damping-mode adaptive_exact --epsilon ${eps} \
--run-id $(ts)_kappa_adaptive_e${eps}_$((_seq++))"
  fi
done

echo ""
echo "Submetidos $((_seq)) jobs. Monitore: make status  |  baixe: make download-results"
echo "Analise pareado: python scripts/paired_stats.py output/<run_id_kappa> (vs base)"
