# Snippets run to setup the compute environment


sudo apt-get update
sudo apt-get install -y screen wget bzip2 git libxml2-dev


wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
bash Anaconda3-2022.10-Linux-x86_64.sh -b -p $HOME/anaconda3
rm Anaconda3-2022.10-Linux-x86_64.sh
eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
conda update -y conda


wget https://github.com/git-lfs/git-lfs/releases/download/v3.3.0/git-lfs-linux-amd64-v3.3.0.tar.gz
tar -xvf git-lfs-linux-amd64-v3.3.0.tar.gz
pushd git-lfs-3.3.0/
sudo ./install.sh
popd
rm -r git-lfs-*


git clone git@github.com:astralarya/network_inflation.git
cd network_inflation
conda env create -f environment.yml
conda activate network_inflation
pip install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.13-cp38-cp38-linux_x86_64.whl

# https://cloud.google.com/compute/docs/disks/add-persistent-disk