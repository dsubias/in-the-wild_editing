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


def make_dataset(root, train_file, test_file, mode, selected_attrs):
    assert mode in ['train', 'val', 'test']

    lines_train = [line.rstrip() for line in open(os.path.join(
        root,  train_file), 'r')]
    lines_test = [line.rstrip() for line in open(os.path.join(
        root, test_file), 'r')]
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
        lines = lines_test[1:33]
        print('Validation Samples:', len(lines))
    if mode == 'test':
        # np.random.shuffle(lines_test)
        #lines_test=[line for line in lines_test if ("@mat" in line)]
        # [:32*2]+random.sample(lines_train,32*4) #for full dataset
        lines = lines_test[1:]
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
            mat_attr.append(float(values[idx]))  # * 2 - 1)

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


# sampler that return batches with a control on the images (where only the shape/illum/mat is changing across the batch)
# not used in the end
class DisentangledSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.mats = self.data_source.mats
        self.geoms = self.data_source.geoms
        self.illums = self.data_source.illums

        self.batch_size = batch_size

    def __iter__(self):
        # get all possible idxs in the dataset shuffled
        rand_idxs = torch.randperm(len(self.data_source)).tolist()

        # here we keep track of the number of batches sampled
        while len(rand_idxs) > 0:
            # mode for the batch -> 0:material, 1:geometry, 2:illumination
            mode = random.randint(0, 2)

            # get current index
            curr_idx = rand_idxs.pop(0)

            # make structure where we will append the indexes
            yield curr_idx, mode

            # get current object properties
            material_idx = self.mats[curr_idx]
            geometry_idx = self.geoms[curr_idx]
            illumination_idx = self.illums[curr_idx]

            # see other objects in the dataset that we can sample according to the
            # given mode
            if mode == 0:  # only MATERIAL changes in the batch
                materials = self.mats != material_idx
                geometries = self.geoms == geometry_idx
                illumination = self.illums == illumination_idx
            if mode == 1:  # only GEOMETRY changes in the batch
                materials = self.mats == material_idx
                geometries = self.geoms != geometry_idx
                illumination = self.illums == illumination_idx
            if mode == 2:  # only ILLUMINATION changes in the batch
                materials = self.mats == material_idx
                geometries = self.geoms == geometry_idx
                illumination = self.illums != illumination_idx

            # get the intersection of the possible factors to sample
            possible_idx = materials * geometries * illumination

            # retrieve is position in the tensor
            possible_idx = possible_idx.nonzero()

            # randomly shuffle the idxs that are possible to sample
            possible_idx = possible_idx[torch.randperm(len(possible_idx))]

            # populate the batch with such idxs
            for i in range(self.batch_size - 1):
                yield possible_idx[i], mode

    def __len__(self):
        return len(self.data_source)

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
            normals = np.ndarray((size