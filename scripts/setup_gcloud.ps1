# Run on local dev

$INSTANCE_NAME = "tpu-1"

$GCLOUD_ZONE = "us-central1-f"
$TPU_TYPE = "v2-8"
$TPU_VERSION = "tpu-vm-pt-1.13"
$GCLOUD_DISK = "imagenet--$GCLOUD_ZONE"

gcloud compute tpus tpu-vm create "$INSTANCE_NAME" "--zone=$GCLOUD_ZONE" "--accelerator-type=$TPU_TYPE" "--version=$TPU_VERSION"
gcloud alpha compute tpus tpu-vm attach-disk "$INSTANCE_NAME" "--zone=$GCLOUD_ZONE"  "--disk=$GCLOUD_DISK" --mode=read-only