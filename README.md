# Scaling up via Network Inflation

## Install environment and kernel

Install [Anaconda](https://www.anaconda.com/products/distribution).

```bash
conda env create -f environment.yml
conda activate network_inflation
python -m ipykernel install --user --name=network_inflation
```

## Fetch Imagenet

This repo expects Imagenet data structured for
[ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)

The [`./setup_imagenet.py`](./setup_imagenet.py)
script converts the assets in
Hugging Face [Imagenet](https://huggingface.co/datasets/imagenet-1k).
Access via this method requires a Hugging Face account
and Git LFS.

Clone the dataset after accepting the terms with:

```bash
huggingface-cli login
git clone https://huggingface.co/datasets/imagenet-1k
```

Then unpack the data
(replace `<REPO_PATH>` with the path to the `imagenet-1k` repo):

`python setup_imagenet.py <REPO_PATH>`