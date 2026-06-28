include .env
export

# Add ~/.nebius/bin to PATH if needed: export PATH="$$HOME/.nebius/bin:$$PATH"
NEBIUS     := $(or $(shell which nebius 2>/dev/null), $(HOME)/.nebius/bin/nebius)
RUN_ID     := $(shell date +%Y%m%d_%H%M%S)
JOB_NAME   := fmri-adversarial-attack
TRAIN_NAME := fmri-train

.PHONY: all upload-job-files configure-s3-transfer upload-data \
        deploy-attack deploy-train logs logs-train \
        download-results delete-earliest-run check-env

all: check-env upload-job-files deploy-attack

check-env:
	@test -n "$(PARENT_ID)" || (echo "ERROR: BUCKET_ID not set — cp .env.template .env and fill it in" && exit 1)
	@test -n "$(BUCKET_ID)" || (echo "ERROR: BUCKET_ID not set — run: nebius storage bucket get-by-name ..." && exit 1)
	@echo "PARENT_ID=$(PARENT_ID)  BUCKET_ID=$(BUCKET_ID)  S3_BUCKET=$(S3_BUCKET)"

configure-s3-transfer:
	aws configure set default.s3.multipart_threshold 64MB
	aws configure set default.s3.multipart_chunksize 64MB
	aws configure set default.s3.max_concurrent_requests 50

# Upload code and configs to S3 (run before every job)
upload-job-files: configure-s3-transfer
	aws s3 sync . s3://$(S3_BUCKET)/ \
	  --endpoint-url $(S3_ENDPOINT) \
	  --exclude ".git/*" \
	  --exclude ".env" \
	  --exclude "output/*" \
	  --exclude "logs/*" \
	  --exclude "__pycache__/*" \
	  --exclude "*.pyc" \
	  --exclude "HCP_S1200_Atlas_Z4_pkXDZ/*" \
	  --exclude "data/fmri/*" \
	  --exclude "data/raw_data.npy" \
	  --exclude "data/raw_labels.npy" \
	  --exclude "data/random_permutation.npy"
	@echo "Code synced to s3://$(S3_BUCKET)/"

# Upload large data files once (run once after bucket creation)
upload-data: configure-s3-transfer
	@echo "Uploading ECG data..."
	aws s3 sync data/ s3://$(S3_BUCKET)/data/ \
	  --endpoint-url $(S3_ENDPOINT)
	@echo "Uploading saved models..."
	aws s3 sync saved_model/ s3://$(S3_BUCKET)/saved_model/ \
	  --endpoint-url $(S3_ENDPOINT)
	@echo "Uploading HCP atlas..."
	aws s3 sync HCP_S1200_Atlas_Z4_pkXDZ/ s3://$(S3_BUCKET)/HCP_S1200_Atlas_Z4_pkXDZ/ \
	  --endpoint-url $(S3_ENDPOINT)
	@echo "Data upload complete."

# RESUME_RUN_ID: set to a previous run-id to resume from its partial results
# e.g.: make deploy-attack RESUME_RUN_ID=20260628_023716
RESUME_RUN_ID ?=

_RESUME_FLAG = $(if $(RESUME_RUN_ID),--run-id $(RESUME_RUN_ID),--run-id $(RUN_ID))

# Main job: adversarial attack evaluation on H200
# Output goes directly to /workspace/data/output (S3-mounted) so partial results
# survive job failures — no cp needed at end.
deploy-attack: check-env upload-job-files
	$(NEBIUS) ai job create \
	  --parent-id $(PARENT_ID) \
	  --name $(JOB_NAME) \
	  --image pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime \
	  --platform gpu-h200-sxm \
	  --preset 1gpu-16vcpu-200gb \
	  --disk-size 200Gi \
	  --shm-size 32Gi \
	  --volume $(BUCKET_ID):/workspace/data \
	  --container-command bash \
	  --args '-c "apt-get update -qq && apt-get install -y git -q && cd /workspace/data && pip install --no-cache-dir -r requirements.txt && python test_fmri_model.py --config configs/config.yaml --output-dir /workspace/data/output $(_RESUME_FLAG)"'
	@echo "Job submitted. Monitor with: make logs"

# Optional: re-training job
deploy-train: check-env upload-job-files
	$(NEBIUS) ai job create \
	  --parent-id $(PARENT_ID) \
	  --name $(TRAIN_NAME) \
	  --image pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime \
	  --platform gpu-h200-sxm \
	  --preset 1gpu-16vcpu-200gb \
	  --disk-size 200Gi \
	  --shm-size 32Gi \
	  --volume $(BUCKET_ID):/workspace/data \
	  --container-command bash \
	  --args '-c "cd /workspace/data && pip install --no-cache-dir -r requirements.txt && python train_fmri.py --out-dir /tmp/model && cp -r /tmp/model /workspace/data/saved_model_$(RUN_ID)"'

logs:
	@JOB_ID=$$($(NEBIUS) ai job get-by-name --name $(JOB_NAME) \
	  --parent-id $(PARENT_ID) --format jsonpath='{.metadata.id}'); \
	echo "Job ID: $$JOB_ID"; \
	$(NEBIUS) ai job logs $$JOB_ID --follow

logs-train:
	@JOB_ID=$$($(NEBIUS) ai job get-by-name --name $(TRAIN_NAME) \
	  --parent-id $(PARENT_ID) --format jsonpath='{.metadata.id}'); \
	$(NEBIUS) ai job logs $$JOB_ID --follow

download-results:
	mkdir -p output
	aws s3 sync s3://$(S3_BUCKET)/output/ ./output/ \
	  --endpoint-url $(S3_ENDPOINT)
	@echo "Results in ./output/"

# Remove the oldest run directory from S3
delete-earliest-run:
	@EARLIEST=$$(aws s3 ls s3://$(S3_BUCKET)/output/ --endpoint-url $(S3_ENDPOINT) | \
	  awk '{print $$NF}' | sort | head -1); \
	echo "Deleting s3://$(S3_BUCKET)/output/$$EARLIEST"; \
	aws s3 rm s3://$(S3_BUCKET)/output/$$EARLIEST --recursive \
	  --endpoint-url $(S3_ENDPOINT)
