sudo apt-get update
sudo apt-get install wget bzip2 git libxml2-dev


wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
bash Anaconda3-2022.10-Linux-x86_64.sh -b -p $HOME/anaconda3
rm Anaconda3-2022.10-Linux-x86_64.sh
source ~/.bashrc


wget https://github.com/git-lfs/git-lfs/releases/download/v3.3.0/git-lfs-linux-amd64-v3.3.0.tar.gz
tar -xvf git-lfs-linux-amd64-v3.3.0.tar.gz
pushd git-lfs-3.3.0/
sudo ./install.sh
popd
rm -r git-lfs-*

# https://cloud.google.com/compute/docs/disks/add-persistent-disk