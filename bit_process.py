import os
import random
import shutil
from tqdm import tqdm

import scipy.io as sio

base_dir = "./data/BITVehicle_Dataset/"  # C:/users/username/downloads/Matlab to txt via python/BITVehicle_Dataset


def mymovefile(srcfile, dstfile):
    if not os.path.isfile(srcfile):

        print("%s not exist!" % (srcfile))

    else:

        fpath, fname = os.path.split(dstfile)

        if not os.path.exists(fpath):
            os.makedirs(fpath)

        shutil.move(srcfile, dstfile)

        print("move %s -> %s" % (srcfile, dstfile))


load_fn = base_dir + 'VehicleInfo.mat'
load_data = sio.loadmat(load_fn)
data = load_data['VehicleInfo']

# Line below will create a 1500 images for test set if you want more or less change the "1500" value below to your desire
random.seed(42)
test_index = random.sample(range(data.size), 7500)

image_dir = './data/BITVehicle/images/'
label_dir = './data/BITVehicle/labels/'
if os.path.exists(image_dir):
    shutil.rmtree(image_dir)
os.makedirs(image_dir)
if os.path.exists(label_dir):
    shutil.rmtree(label_dir)
os.makedirs(label_dir)

image_test_dir = os.path.join(image_dir, 'test')
label_test_dir = os.path.join(label_dir, 'test')
os.mkdir(image_test_dir)
os.mkdir(label_test_dir)

image_train_dir = os.path.join(image_dir, 'train')
label_train_dir = os.path.join(label_dir, 'train')
os.mkdir(image_train_dir)
os.mkdir(label_train_dir)

with tqdm(total=data.size) as pbar:
    for i, item in enumerate(data):
        # item = data[i]
        str = ""
        for j in range(item['vehicles'][0][0].size):
            name = item['name'][0][0]

            # Bus, Microbus, Minivan, Sedan, SUV, and Truck
            vehicles = item['vehicles'][0][0][j]
            height = item['height'][0][0][0]
            width = item['width'][0][0][0]

            left = vehicles[0][0][0]
            top = vehicles[1][0][0]
            right = vehicles[2][0][0]
            bottom = vehicles[3][0][0]
            vehicles_type = vehicles[4][0]

            if vehicles_type == 'Bus':
                vehicles_type = 0
            elif vehicles_type == 'Microbus':
                vehicles_type = 1
            elif vehicles_type == 'Minivan':
                vehicles_type = 2
            elif vehicles_type == 'Sedan':
                vehicles_type = 3
            elif vehicles_type == 'SUV':
                vehicles_type = 4
            elif vehicles_type == 'Truck':
                vehicles_type = 5

            str += f'{vehicles_type} {(left + (right - left)/2)/width:6f} {(top + (bottom - top)/2)/height:6f} {(right - left) / width:6f} {((bottom - top) / height):6f}\n'

        str = str[:str.rfind('\n')]
        # print(str, name[:-4])
        if i in test_index:
            shutil.copyfile(base_dir + name, os.path.join(image_test_dir, name))
            with open(os.path.join(label_test_dir, f'{name[:-4]}.txt'), 'a+') as f:
                f.write(str + '\n')

        else:
            shutil.copyfile(base_dir + name, os.path.join(image_train_dir, name))
            with open(os.path.join(label_train_dir, f'{name[:-4]}.txt'), 'a+') as f:
                f.write(str + '\n')

        pbar.update(1)