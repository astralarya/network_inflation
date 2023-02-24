# Run on local dev

NAME=tpu-1

ZONE=us-central1-f
TPU_TYPE=v2-8
TPU_VERSION=tpu-vm-pt-1.13
DISK=imagenet--$ZONE

gcloud compute tpus tpu-vm create "$NAME" "--zone=$ZONE" "--accelerator-type=$TPU_TYPE" "--version=$TPU_VERSION"
gcloud alpha compute tpus tpu-vm attach-disk "$NAME" "--zone=$ZONE"  "--disk=$DISK" --mode=read-only