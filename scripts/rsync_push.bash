INSTANCE_ADDR="$1"
SOURCE_DIR="$2"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
RSYNC_USER="$RSYNC_USER"

GCLOUD_ZONE=$( dirname "$INSTANCE_ADDR" )
INSTANCE_NAME=$( basename "$INSTANCE_ADDR" )

DEST_DIR="network_inflation/$SOURCE_DIR"


address="$( \
    gcloud compute tpus tpu-vm describe "$INSTANCE_NAME" \
    "--zone=$GCLOUD_ZONE" \
    "--format=get(networkEndpoints[0].accessConfig.externalIp)" \
)"
echo "Rsync push $INSTANCE_NAME ($address)..."

ssh "$RSYNC_USER@$address" "mkdir -p '$DEST_DIR'" &&
rsync -aP "$SOURCE_DIR/" "$RSYNC_USER@$address:$DEST_DIR/"
