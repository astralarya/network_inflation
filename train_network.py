from local import imagenet
from local import resnet

train_data = imagenet.train_data("/mnt/imagenet/imagenet-1k/train/")


resnet152 = resnet.resnet152()
imagenet.train(resnet152, "resnet152", train_data)

