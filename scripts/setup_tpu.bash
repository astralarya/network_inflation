# Snippets run to setup the TPU VM


sudo apt-get update
sudo apt-get install -y screen git

echo 'export XRT_TPU_CONFIG="localservice;0;localhost:51011"' >> ~/.bashrc
echo 'export PATH="$PATH:$HOME/.local/bin/"' >> ~/.bashrc
source ~/.bashrc

# https://cloud.google.com/compute/docs/disks/add-persistent-disk
sudo mkdir /mnt/imagenet
sudo mount -o discard,defaults /dev/sdb /mnt/imagenet


ssh-keygen
cat ~/.ssh/id_rsa.pub

git clone git@github.com:astralarya/network_inflation.git
cd network_inflation
pip install -r requirements.txt

 