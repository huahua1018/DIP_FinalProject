import torchvision
from torchvision import transforms
import os
import errno
import shutil
from pathlib import Path
from PIL import Image

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass

# 下載並加載驗證集
trainset = torchvision.datasets.Places365(
    root="./data",                  # 數據集存放路徑
    split="val",                    # 指定為驗證集
    small=True,                     # 使用小尺寸圖像 (256x256)
    download=True,                  # 下載數據集
)
root_train1 = './root_place365_val_train1/'
root_train2 = './root_place365_val_train2/'
del_folder(root_train1)
create_folder(root_train1)

del_folder(root_train2)
create_folder(root_train2)

exts = ['jpg', 'jpeg', 'png']
folder = './data/val_256'
paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

for idx in range(len(paths)):
    img = Image.open(paths[idx])
    print(idx)
    if idx < 0.5*len(paths):
        img.save(root_train1 + str(idx) + '.png')
    else:
        img.save(root_train2 + str(idx) + '.png')

############################################# MNIST ###############################################
# trainset = torchvision.datasets.MNIST(
#             root='./data', train=True, download=True)
# root = './root_mnist/'
# del_folder(root)
# create_folder(root)

# for i in range(10):
#     lable_root = root + str(i) + '/'
#     create_folder(lable_root)

# for idx in range(len(trainset)):
#     img, label = trainset[idx]
#     # print(idx)
#     img.save(root + str(label) + '/' + str(idx) + '.png')


# trainset = torchvision.datasets.MNIST(
#             root='./data', train=False, download=True)
# root = './root_mnist_test/'
# del_folder(root)
# create_folder(root)

# for i in range(10):
#     lable_root = root + str(i) + '/'
#     create_folder(lable_root)

# for idx in range(len(trainset)):
#     img, label = trainset[idx]
#     # print(idx)
#     img.save(root + str(label) + '/' + str(idx) + '.png')


############################################# Cifar10 ###############################################
# trainset = torchvision.datasets.CIFAR10(
#             root='./data', train=True, download=True)
# root = './root_cifar10/'
# del_folder(root)
# create_folder(root)

# for i in range(10):
#     lable_root = root + str(i) + '/'
#     create_folder(lable_root)

# for idx in range(len(trainset)):
#     img, label = trainset[idx]
#     # print(idx)
#     img.save(root + str(label) + '/' + str(idx) + '.png')


# trainset = torchvision.datasets.CIFAR10(
#             root='./data', train=False, download=True)
# root = './root_cifar10_test/'
# del_folder(root)
# create_folder(root)

# for i in range(10):
#     lable_root = root + str(i) + '/'
#     create_folder(lable_root)

# for idx in range(len(trainset)):
#     img, label = trainset[idx]
#     # print(idx)
#     img.save(root + str(label) + '/' + str(idx) + '.png')

# CelebA_folder = './data/CelebA-img' # change this to folder which has CelebA data
############################################# CelebA ###############################################
# root_train = './root_celebA_128_train_new/'
# root_test = './root_celebA_128_test_new/'
# del_folder(root_train)
# create_folder(root_train)

# del_folder(root_test)
# create_folder(root_test)

# exts = ['jpg', 'jpeg', 'png']
# folder = CelebA_folder
# paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

# for idx in range(len(paths)):
#     img = Image.open(paths[idx])
#     print(idx)
#     if idx < 0.9*len(paths):
#         img.save(root_train + str(idx) + '.png')
#     else:
#         img.save(root_test + str(idx) + '.png')