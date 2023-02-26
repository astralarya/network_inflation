SOURCE_DIR="$1"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
RSYNC_USER="$RSYNC_USER"

GCLOUD_ZONE="us-central1-f"
INSTANCE_NAME="tpu-4"

DEST_DIR="network_inflation/$1"


address="$( \
    gcloud compute tpus tpu-vm describe "$INSTANCE_NAME" \
    "--zone=$GCLOUD_ZONE" \
    "--format=get(networkEndpoints[0].accessConfig.externalIp)" \
)"
echo "Rsync push $INSTANCE_NAME ($address)..."

ssh "$RSYNC_USER@$address" "mkdir -p '$DEST_DIR'" &&
rsync -aP "$SOURCE_DIR/" "$RSYNC_USER@$address:$DEST_DIR/"
