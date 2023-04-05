SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
RSYNC_USER="$RSYNC_USER"


INSTANCES=(
    "us-central1-f/tpu-1"
    "us-central1-f/tpu-2"
    "us-central1-f/tpu-3"
    # "us-central1-f/tpu-4"
    # "us-central1-f/tpu-5"
    # "europe-west4-a/tpu-eu-1"
    # "europe-west4-a/tpu-eu-2"
    # "europe-west4-a/tpu-eu-3"
    # "europe-west4-a/tpu-eu-4"
    # "europe-west4-a/tpu-eu-5"
)

SOURCE_DIR="/mnt/models/data"
OUTPUT_DIR="$SCRIPT_DIR/../remotes"

mkdir -p "$OUTPUT_DIR"

for item in "${INSTANCES[@]}"
do
    zone="$(dirname "$item")"
    instance="$(basename "$item")"
    address="$( \
        gcloud compute tpus tpu-vm describe "$instance" \
        "--zone=$zone" \
        "--format=get(networkEndpoints[0].accessConfig.externalIp)" \
    )"
    echo "Rsync $item ($address)..."
    mkdir -p "$OUTPUT_DIR/$zone/$instance"
    rsync -aP --copy-links \
        "$RSYNC_USER@$address:$SOURCE_DIR/" \
        "$OUTPUT_DIR/$zone/$instance/" \
        --exclude='*.pkl' \
        --exclude=".[!.]*"
done
