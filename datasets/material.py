import os
import math
import torch
from torch.utils import data
from torchvision import transforms as Tvision
# import datasets.transforms2 as T2
# import datasets.transforms3 as T3
from PIL import Image
import random
import numpy as np
#from utils.im_util import get_alpha_channel
import cv2
from matplotlib import pyplot as plt
import pytorch_lightning as pl
from tqdm import tqdm

# go and read the dataset file. Output the list of files with their attributes (and potentially the material/illum/shape but currently commented to go faster)


def make_dataset(root, train_file, test_file, mode, selected_attrs):
    assert mode in ['train', 'val', 'test']

    lines_train = [line.rstrip() for line in open(os.path.join(
        root,  train_file), 'r')]
    lines_test = [line.rstrip() for line in open(os.path.join(
        root, test_file), "r")]
    all_attr_names = lines_train[0].split()
    print(mode, all_attr_names)

    attr2idx = {}
    idx2attr = {}
    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name

    np.random.seed(10)
    random.seed(18)

    if mode == 'train':
        lines = lines_train[1:]
        print('Train Samples:', len(lines))
    if mode == 'val':  # put in first half a batch of test images, half of training images
        lines = lines_test[1:17]
        print('Validation Samples:', len(lines))
    if mode == 'test':
        # np.random.shuffle(lines_test)
        #lines_test=[line for line in lines_test if ("@mat" in line)]
        # [:32*2]+random.sample(lines_train,32*4) #for full dataset
        lines = lines_test[17:]
        print('Test Samples', len(lines))
        # #only from one shape/one env
        # shape=""
        # env=""
        # lines_train=[line for line in lines_train if (shape in line and env in line)]
        # #take 100 random images
        # np.random.shuffle(lines)
        # lines_train=lines_train[:200]
    # print(len(lines))

    files = []
    mat_attrs = []
    material = []
    geometry = []
    illumination = []
    for i, line in enumerate(lines):
        split = line.split()
        filename = split[0]
        values = split[1:]
        mat_attr = []
        for attr_name in selected_attrs:
            idx = attr2idx[attr_name]
            mat_attr.append(float(values[idx]) * 2 - 1)

        files.append(filename)
        mat_attrs.append(mat_attr)

        filename_split = filename.split(
            '/')[-1].split('@')[-1].split('.')[0].split('-')
        # material.append(filename_split[1])
        # geometry.append(filename_split[0])
        # illumination.append(filename_split[2])

    return {'files': files,
            'mat_attrs': mat_attrs,
            'materials': material,
            'geometries': geometry,
            'illuminations': illumination}

# used for the disentangler sampler


def list2idxs(l):
    idx2obj = list(set(l))
    idx2obj.sort()
    obj2idx = {mat: i for i, mat in enumerate(idx2obj)}
    l = [obj2idx[obj] for obj in l]
    # print(idx2obj)
    # print(obj2idx)
    return l, idx2obj, obj2idx

# extract the highlights from an image of a glossy mat


def extract_highlights(image):
    mask = (image[:, :, 3:]/255.0)
    image_bw = np.sum(image[:, :, :3] * [0.299, 0.587,
                      0.114], axis=2, keepdims=True)*mask
    # print(image_bw.shape,image.shape)
    # compute median color
    image_ma = np.ma.masked_array(image_bw, 1-mask)
    med = np.ma.median(image_ma)
    image_high = np.clip(image_bw-med, 0, 255).astype(np.uint8)  # +med
    return image_high


class MaterialDataset(data.Dataset):
    def __init__(self, root, train_file, test_file, mode, selected_attrs, disentangled=False, transform=None, mask_input_bg=True, use_illum=False):

        items = make_dataset(root, train_file, test_file, mode, selected_attrs)
        self.files = items['files']
        self.mat_attrs = items['mat_attrs']
        # if using the disentangled sampler
        # mats, self.idx2mats, self.mats2idxs = list2idxs(items['materials'])
        # geoms, self.idx2geoms, self.geoms2idx = list2idxs(items['geometries'])
        # illums, self.idx2illums, self.illums2idx = list2idxs(items['illuminations'])

        # self.mats = np.array(mats)
        # self.geoms = np.array(geoms)
        # self.illums = np.array(illums)

        self.root = root
        self.mode = mode
        self.disentangled = disentangled
        self.transform = transform
        self.mask_input_bg = mask_input_bg
        self.use_illum = use_illum

    def __getitem__(self, index_and_mode):
        if self.disentangled:
            index, sampling_mode = index_and_mode
        else:
            index = index_and_mode

        mat_attr = self.mat_attrs[index]
        # mat=self.mats[index]
        # geom=self.geoms[index]
        # illum=self.illums[index]

        # OpenCV version
        # read image

        image_rgb = cv2.cvtColor(cv2.imread(os.path.join(
            self.root, "renderings", self.files[index]), 1), cv2.COLOR_BGR2RGB)
        size = image_rgb.shape[0]
        # read normals
        normals_bgra = cv2.imread(os.path.join(
            self.root, "normals", self.files[index][:-3]+"png"), -1)
        if (type(normals_bgra) is np.ndarray):
            # if the normals exist, resize them and trasnform to RGB (is BGR when reading)
            if normals_bgra.shape[0] != size:
                normals_bgra = cv2.resize(normals_bgra, (size, size))
            normals = np.ndarray((size, size, 4), dtype=np.uint8)
            cv2.mixChannels([normals_bgra], [normals],
                            [0, 2, 1, 1, 2, 0, 3, 3])
        else:
            # otherwise, put the normals and a full mask
            mask = np.ones((size, size, 1), np.uint8)*255
            normals = np.ndarray((size, size, 4), dtype=np.uint8)
            cv2.mixChannels([image_rgb, mask], [normals],
                            [0, 0, 1, 1, 2, 2, 3, 3])
        if self.mode == "test":
            # slighlty erode mask so that the results are nicer
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            normals[:, :, 3] = cv2.dilate(normals[:, :, 3], element)
        # add mask to image
        image = np.ndarray(normals.shape, dtype=np.uint8)
        cv2.mixChannels([image_rgb, normals], [image],
                        [0, 0, 1, 1, 2, 2, 6, 3])

        if (self.use_illum):
            illum = cv2.imread(os.path.join(
                self.root, "illum", self.files[index]), -1)
            if (not type(illum) is np.ndarray):
                illum = extract_highlights(image)
                illum = np.concatenate(
                    [illum, illum, illum], axis=2)  # 3channels?
            else:
                if illum.ndim == 3:  # RGB image
                    # or cv2.COLOR_BGR2GRAY
                    illum = cv2.cvtColor(illum, cv2.COLOR_BGR2RGB)
                else:
                    illum = illum[:, :, np.newaxis]  # image is already B&W
                    illum = np.concatenate([illum, illum, illum], axis=2)
        else:
            illum = torch.Tensor()

        # PIL version: faster but apply the alpha channel when resizing
        #image = Image.open(os.path.join(self.root, "renderings", self.files[index]))
        # try:
        #     normals = Image.open(os.path.join(self.root, "normals", self.files[index][:-3]+"png"))
        #     mask=get_alpha_channel(normals)
        # except FileNotFoundError:
        #     #put the original image in place of the normals + full mask
        #     normals=image
        #     mask = Image.new('L',normals.size,255)
        #     normals.putalpha(mask)
        # image.putalpha(mask)

        # apply the transforms
        if self.transform is not None:
            if self.use_illum:
                image, normals, illum = self.transform(image, normals, illum)
            else:
                image, normals = self.transform(image, normals)

        # mask the normals
        normals = normals*normals[3:]
        # mask the input image if asked
        if self.mask_input_bg:
            image = image*image[3:]
            if self.use_illum:
                illum = illum*image[3:]

        if self.disentangled:
            return image, normals, illum, torch.FloatTensor(mat_attr), sampling_mode
        else:

            return image, normals, self.files[index][:-4].split("/")[-1], torch.FloatTensor(mat_attr)

    def __len__(self):
        return len(self.files)


class DataModule(pl.LightningDataModule):

    def __init__(self, root, train_file, test_file, mode, selected_attrs, crop_size=None, image_size=128, batch_size=16, data_augmentation=False, mask_input_bg=True, use_illum=False):
        if mode not in ['train', 'test']:

            return

        self.root = root
        self.data_augmentation = data_augmentation
        self.image_size = image_size
        self.crop_size = crop_size

        # setup the dataloaders
        self.train_trf, self.val_trf = self.setup_transforms(use_illum)
        self.val_set = self.train_set = self.test_set = None
        self.selected_attrs = selected_attrs
        self.mask_input_bg = mask_input_bg
        self.use_illum = use_illum

        if mode == 'train':
            self.batch_size = batch_size
        else:
            self.batch_size = 32

        self.root = root
        self.train_file = train_file
        self.test_file = test_file
        self.mode = mode
        self.mask_input_bg = mask_input_bg
        self.use_illum = use_illum

    def val_dataloader(self):

        self.val_set = MaterialDataset(
            self.root, self.train_file, self.test_file, 'val', self.selected_attrs, transform=self.val_trf, mask_input_bg=self.mask_input_bg, use_illum=self.use_illum)
        self.val_iterations = int(
            math.ceil(len(self.val_set) / self.batch_size))
        return data.DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def train_dataloader(self):

        self.train_set = self.train_set = MaterialDataset(
            self.root, self.train_file, self.test_file, 'train', self.selected_attrs, transform=self.train_trf, mask_input_bg=self.mask_input_bg, use_illum=self.use_illum)
        self.train_iterations = int(
            math.ceil(len(self.train_set) / self.batch_size))
        return data.DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def test_dataloader(self):
        self.test_set = MaterialDataset(
            self.root, self.train_file, self.test_file, 'test', self.selected_attrs, transform=self.val_trf, mask_input_bg=self.mask_input_bg, use_illum=self.use_illum)
        return data.DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def setup_transforms(self, use_illum):

        global T

        if (use_illum):
            dset = __import__('datasets.transforms3', globals(), locals())
            T = dset.transforms3
        else:
            dset = __import__('datasets.transforms2', globals(), locals())
            T = dset.transforms2
        # basic transform to put at the right size
        val_trf = T.Compose([
            # T.CenterCrop(self.crop_size),
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5, 0), std=(0.5, 0.5, 0.5, 1))
        ])
        # training transform : data augmentation
        original_size = self.image_size*2
        if self.data_augmentation:
            train_trf = T.Compose([
                T.Resize(original_size),  # suppose the dataset is of size 256
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.5),
                T.Random180DegRot(0.5),
                T.Random90DegRotClockWise(0.5),
                T.Albumentations(50, 50, 0.5),  # change in color
                T.RandomCrop(size=self.crop_size),
                T.RandomResize(low=original_size,
                               high=int(original_size*1.1718)),
                T.CenterCrop(original_size),
                # T.RandomRotation(degrees=(-5, 5)), #TODO recode for normals
                val_trf,
            ])
        else:
            train_trf = T.Compose([
                T.Resize(original_size),
                val_trf,
            ])
        val_trf = T.Compose([
            T.Resize(original_size),
            val_trf,
        ])
        return train_trf, val_trf
