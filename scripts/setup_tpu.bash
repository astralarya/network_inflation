# Snippets run to setup the TPU VM


sudo apt-get update
sudo apt-get install -y screen git

echo 'export XRT_TPU_CONFIG="localservice;-1;localhost:51011"' >> ~/.bashrc
source ~/.bashrc

# git clone git@github.com:astralarya/network_inflation.git
# pip install -r requirements.txt

# https://cloud.google.com/compute/docs/disks/add-persistent-disk
# sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
# sudo mkdir /mnt/imagenet
# sudo mount -o discard,defaults /dev/sdb /mnt/imagenet
 