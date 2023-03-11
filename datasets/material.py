import os
import math
import torch
from torch.utils import data
from torchvision import transforms as Tvision
from PIL import Image
import random
import numpy as np
import cv2
from matplotlib import pyplot as plt
from torchvision.utils import save_image

def make_dataset(root, train_file, val_file, test_file, mode, selected_attrs):

    assert mode in ['train', 'val', 'edit_images', 'edit_video', 'plot_metrics']

    if mode == 'train':
        file_lines = [line.rstrip() for line in open(os.path.join(
        root,  train_file), 'r')]
    
        lines = file_lines[1:]
        print('Train Samples:', len(lines))

    if mode == 'val':  # put in first half a batch of test images, half of training images
        
        file_lines = [line.rstrip() for line in open(os.path.join(
                      root,  val_file), 'r')] 
        lines = file_lines[1:]
        print('Validation Samples:', len(lines))

    else:
        file_lines = [line.rstrip() for line in open(os.path.join(
                      root, test_file), 'r')]
        lines = file_lines[1:]
        print('Test Samples', len(lines))

    all_attr_names = file_lines[0].split()
    print('Mode: ', mode, ', Attributes in the dataset: ', all_attr_names)
    attr2idx = {}
    idx2attr = {}

    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name

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
            mat_attr.append(float(values[idx]))
        
        files.append(filename)
        mat_attrs.append(mat_attr)

    return {'files': files,
            'mat_attrs': mat_attrs,
            'materials': material,
            'geometries': geometry,
            'illuminations': illumination}

def list2idxs(l):

    idx2obj = list(set(l))
    idx2obj.sort()
    obj2idx = {mat: i for i, mat in enumerate(idx2obj)}
    l = [obj2idx[obj] for obj in l]

    return l, idx2obj, obj2idx

# extract the highlights from an image of a glossy mat


def extract_highlights(image,image_rgb):
    dset = __import__('datasets.transforms2', globals(), locals())
    T = dset.transforms2
    image = T.ToTensor()(T.Resize(256)(image))
    image_rgb = T.ToTensor()(T.Resize(256)(image_rgb))
    mask = torch.where(image[3,:,:] <= 0.1, 1,0)
    save_image(mask.float(),'mask.png')
    save_image(image.float(),'img.png')
    save_image(image_rgb.float(),'img_rgb.png')
    back_ground = image[:3,:,:] * mask

    return back_ground,mask, image_rgb


class MaterialDataset(data.Dataset):
    def __init__(self, root, train_file, val_file, test_file, mode, selected_attrs, disentangled=False, transform=None, mask_input_bg=True, use_illum=False):
        items = make_dataset(root, train_file, val_file, test_file, mode, selected_attrs)

        self.files = items['files']
        self.mat_attrs = items['mat_attrs']
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

        #print(image)
        #print(image_rgb)
        if (self.use_illum):
            #illum = cv2.imread(os.path.join(self.root, "illum", self.files[index]), -1)
            #if True:
            illum , mask, rgb = extract_highlights(image,image_rgb)


        if self.transform is not None:
            image, normals = self.transform(image, normals)

        # mask the normals
        normals = normals*normals[3:]
        # mask the input image if asked
        if self.mask_input_bg:
            image = image*image[3:]

        if self.mode == 'edit_images' or self.mode == 'edit_video':

            filename = self.files[index].split('/')[1][:-4]
            return image, mask, rgb
        
        elif self.mode == 'plot_metrics':

            return image, torch.FloatTensor(mat_attr), filename, mask,rgb
        
        else:
            return image, torch.FloatTensor(mat_attr)

    def __len__(self):
        return len(self.files)


# the main dataloader
class MaterialDataLoader(object):
    def __init__(self, root, train_file, val_file, test_file, mode, selected_attrs, crop_size=None, image_size=128, batch_size=16, data_augmentation=True, mask_input_bg=True, use_illum=False):
        if mode not in ['train', 'val', 'edit_images', 'edit_video', 'plot_metrics']:
            return

        self.root = root
        self.data_augmentation = data_augmentation
        self.image_size = image_size
        self.crop_size = crop_size

        # setup the dataloaders
        if mode != 'train':
            use_illum=True
            self.data_augmentation = False
        train_trf, val_trf = self.setup_transforms()

        if mode == 'train':
            print("loading data")
            val_set = MaterialDataset(
                root, train_file, val_file, test_file, 'val', selected_attrs, transform=val_trf, mask_input_bg=mask_input_bg, use_illum=use_illum)
            self.val_loader = data.DataLoader(
                val_set, batch_size=batch_size, shuffle=False, num_workers=4)
            self.val_iterations = int(math.ceil(len(val_set) / batch_size))

            train_set = MaterialDataset(
                root, train_file, val_file, test_file, 'train', selected_attrs, transform=train_trf, mask_input_bg=mask_input_bg, use_illum=use_illum)
            self.train_loader = data.DataLoader(
                train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            self.train_iterations = int(math.ceil(len(train_set) / batch_size))
        else:
            batch_size = 1
            test_set = MaterialDataset(
                root, train_file, val_file, test_file,  mode, selected_attrs, transform=val_trf, mask_input_bg=mask_input_bg, use_illum=use_illum)
            self.test_loader = data.DataLoader(
                test_set, batch_size=batch_size, shuffle=False, num_workers=1)
            self.test_iterations = int(math.ceil(len(test_set) / batch_size))

    def setup_transforms(self):
        global T
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
        #if self.image_size != 512:
        high_resolution = 512
        if self.data_augmentation:
            
            train_trf = T.Compose([
                T.Resize(high_resolution),  # suppose the dataset is of size 256
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.5),
                T.Random180DegRot(0.5),
                T.Random90DegRotClockWise(0.5),
                T.Albumentations(100, 100, 0.5),  # change in color
                T.RandomCrop(size=self.crop_size),
                T.RandomResize(low=high_resolution,
                               high=int(high_resolution*1.1718)),
                T.CenterCrop(high_resolution),
                val_trf,
            ])
        else:
            train_trf = T.Compose([
                T.Resize(high_resolution),
                val_trf,
            ])
        val_trf = T.Compose([
            T.Resize(high_resolution),
            val_trf,
        ])
        return train_trf, val_trf
