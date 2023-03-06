# Run on local dev

INSTANCE_NAME="tpup-1"
GCLOUD_ZONE="us-central1-f"

TPU_TYPE="v2-8"
TPU_VERSION="tpu-vm-pt-1.13"
GCLOUD_SNAPSHOT="imagenet-1k"

gcloud compute tpus tpu-vm create "$INSTANCE_NAME" \
    "--zone=$GCLOUD_ZONE" \
    "--accelerator-type=$TPU_TYPE" \
    --preemptible \
    "--version=$TPU_VERSION"

# Imagenet

gcloud compute disks create "imagenet--$INSTANCE_NAME" \
    "--zone=$GCLOUD_ZONE" \
    "--size=150GB" \
    "--source-snapshot=$GCLOUD_SNAPSHOT"

gcloud alpha compute tpus tpu-vm attach-disk "$INSTANCE_NAME" \
    "--zone=$GCLOUD_ZONE" \
    "--disk=imagenet--$INSTANCE_NAME" \
    --mode=read-only

# Model checkpoints

gcloud compute disks create "models--$INSTANCE_NAME" \
    "--zone=$GCLOUD_ZONE" \
    "--size=200GB"

gcloud alpha compute tpus tpu-vm attach-disk "$INSTANCE_NAME" \
    "--zone=$GCLOUD_ZONE" \
    "--disk=models--$INSTANCE_NAME"

# SSH

gcloud compute tpus tpu-vm ssh "$INSTANCE_NAME" \
    "--zone=$GCLOUD_ZONE"

# gcloud compute tpus tpu-vm ssh "$INSTANCE_NAME" \
#     "--zone=$GCLOUD_ZONE" \
#     --ssh-flag="-4 -L 9001:localhost:9001"