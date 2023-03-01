# Run on local dev

INSTANCE_NAME="tpu-eu-2"

GCLOUD_ZONE="europe-west4-a"
TPU_TYPE="v3-8"
TPU_VERSION="tpu-vm-pt-1.13"
GCLOUD_DISK="imagenet--$INSTANCE_NAME"
GCLOUD_DISK_SIZE="150GB"
GCLOUD_SNAPSHOT="imagenet-1k--eu"

gcloud compute tpus tpu-vm create "$INSTANCE_NAME" \
    "--zone=$GCLOUD_ZONE" \
    "--accelerator-type=$TPU_TYPE" \
    "--version=$TPU_VERSION"

gcloud compute disks create "$GCLOUD_DISK" \
    "--zone=$GCLOUD_ZONE" \
    "--size=$GCLOUD_DISK_SIZE" \
    "--source-snapshot=$GCLOUD_SNAPSHOT"

gcloud alpha compute tpus tpu-vm attach-disk "$INSTANCE_NAME" \
    "--zone=$GCLOUD_ZONE" \
    "--disk=$GCLOUD_DISK" \
    --mode=read-only

gcloud compute tpus tpu-vm ssh "$INSTANCE_NAME" \
    "--zone=$GCLOUD_ZONE"

# gcloud compute tpus tpu-vm ssh "$INSTANCE_NAME" \
#     "--zone=$GCLOUD_ZONE" \
#     --ssh-flag="-4 -L 9001:localhost:9001"