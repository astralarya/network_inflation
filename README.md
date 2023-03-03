# Scaling up via Network Inflation

## Install environment and kernel

Install [Anaconda](https://www.anaconda.com/products/distribution).

```bash
conda env create -f environment.yml
conda activate network_inflation
python -m ipykernel install --user --name=network_inflation
```

OR

```bash
pip install -r requirements.txt
```

## Fetch Imagenet

This repo expects Imagenet data structured for
[ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)

The [`setup_imagenet.py`](./setup_imagenet.py)
script converts the assets in
Hugging Face [Imagenet](https://huggingface.co/datasets/imagenet-1k).
Access via this method requires a Hugging Face account
and Git LFS.

Clone the dataset after accepting the terms with:

```bash
git clone https://huggingface.co/datasets/imagenet-1k
cd imagenet-1k
git lfs pull
```

Then unpack the data:

```bash
python setup_imagenet.py <REPO_PATH> --imagenet-path=<OUTPUT_DIR>
```

`<REPO_PATH>` is the path to the `imagenet-1k` repo and
`<OUTPUT_DIR>` is the path to create the ImageFolder directories


## Train Networks

The `train_network.py` script starts a new training run,
optionally with inflation.