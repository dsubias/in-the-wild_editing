import os
import math
import torch
from torch.utils import data
import numpy as np
import cv2
from datasets.transforms import *

def make_dataset(logger, root, train_file, val_file, test_folder, mode, selected_attrs):

    assert mode in ['train', 'val', 'edit_images', 'edit_video', 'plot_metrics']

    if mode == 'train':
        file_lines = [line.rstrip() for line in open(os.path.join(
        root,  train_file), 'r')]
    
        lines = file_lines[1:]
        logger.info('Train Samples: ' + str(len(lines)))

    elif mode in ['val', 'plot_metrics']: 
        
        file_lines = [line.rstrip() for line in open(os.path.join(
                      root,  val_file), 'r')] 
        lines = file_lines[1:]
        logger.info('Validation Samples: ' + str(len(lines)))

    else:
        
        lines = os.listdir(test_folder)
        lines.sort()
        logger.info('Test Samples: ' + str(len(lines)))

    if mode == 'train':

        all_attr_names = file_lines[0].split()
        logger.info('Attributes in the dataset: ' + str(all_attr_names))
        attr2idx = {}
        idx2attr = {}
        

        for i, attr_name in enumerate(all_attr_names):

            attr2idx[attr_name] = i
            idx2attr[i] = attr_name

    files = []
    mat_attrs = []

    for i, line in enumerate(lines):
        split = line.split()
        filename = split[0]
        
        if mode == 'train':
            values = split[1:]

        mat_attr = []

        for attr_name in selected_attrs:

            if mode in ['train', 'plot_metrics'] :

                idx = attr2idx[attr_name]
                mat_attr.append(float(values[idx]))

            else:

                # when editing in the wild we assume 0.5 as the original 
                # value of the attribute
                mat_attr.append(0.5)
        
        files.append(filename)
        mat_attrs.append(mat_attr)

    return {'files': files,
            'mat_attrs': mat_attrs}

def list2idxs(l):

    idx2obj = list(set(l))
    idx2obj.sort()
    obj2idx = {mat: i for i, mat in enumerate(idx2obj)}
    l = [obj2idx[obj] for obj in l]

    return l, idx2obj, obj2idx

def get_mask(image,image_rgb):

    image = ToTensor()(Resize(256)(image))
    image_rgb = ToTensor()(Resize(256)(image_rgb))
    mask = torch.where(image[3,:,:] <= 0.1, 1,0)

    return mask, image_rgb


class MaterialDataset(data.Dataset):

    def __init__(self, root, train_file, val_file, test_folder, mode, selected_attrs, logger, transform=None, mask_input_bg=True, use_illum=False):
        items = make_dataset(logger, root, train_file, val_file, test_folder, mode, selected_attrs)

        self.files = items['files']
        self.mat_attrs = items['mat_attrs']
        self.root = root
        self.mode = mode
        self.transform = transform
        self.mask_input_bg = mask_input_bg
        self.use_illum = use_illum
        self.test_folder = test_folder

    def __getitem__(self, index_and_mode):
        
        index = index_and_mode
        mat_attr = self.mat_attrs[index]

        # OpenCV version
        # read image
        if self.mode in ['train', 'plot_metrics', 'val' ]: 

            image_rgb = cv2.cvtColor(cv2.imread(os.path.join(self.root, "renderings", self.files[index]), 1), cv2.COLOR_BGR2RGB)
            size = image_rgb.shape[0]
            # read normals
            normals_bgra = cv2.imread(os.path.join(self.root, "normals", self.files[index][:-3]+"png"), -1)
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
            
            image = np.ndarray(normals.shape, dtype=np.uint8)
            cv2.mixChannels([image_rgb, normals], [image],
                            [0, 0, 1, 1, 2, 2, 6, 3])
        else:

            image = cv2.cvtColor(cv2.imread(os.path.join(self.test_folder, self.files[index]), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
            image_rgb = cv2.cvtColor(cv2.imread(os.path.join(self.test_folder, self.files[index]), 1), cv2.COLOR_BGR2RGB)
            mask, rgb = get_mask(image,image_rgb)
        
        if self.transform is not None:
            image = self.transform(image)

        # mask the input image if asked
        if self.mask_input_bg:
            image = image*image[3:]

        filename = self.files[index].split('/')[-1][:-4]
        if self.mode == 'edit_images' or self.mode == 'edit_video':

            return image, mask, rgb, filename
        
        elif self.mode == 'plot_metrics':
            
            
            return image, torch.FloatTensor(mat_attr), filename
        
        else:
            return image, torch.FloatTensor(mat_attr)

    def __len__(self):
        return len(self.files)


# the main dataloader
class MaterialDataLoader(object):
    def __init__(self, logger, root, train_file, val_file, test_folder, mode, selected_attrs, crop_size=None, image_size=128, batch_size=16, data_augmentation=True, mask_input_bg=True, use_illum=False):
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
            
            val_set = MaterialDataset(
                root, train_file, val_file, test_folder, 'val', selected_attrs, logger, transform=val_trf, mask_input_bg=mask_input_bg, use_illum=use_illum)
            self.val_loader = data.DataLoader(
                val_set, batch_size=batch_size, shuffle=False, num_workers=4)
            self.val_iterations = int(math.ceil(len(val_set) / batch_size))

            train_set = MaterialDataset(
                root, train_file, val_file, test_folder, 'train', selected_attrs, logger, transform=train_trf, mask_input_bg=mask_input_bg, use_illum=use_illum)
            self.train_loader = data.DataLoader(
                train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            self.train_iterations = int(math.ceil(len(train_set) / batch_size))
        else:
            batch_size = 1
            test_set = MaterialDataset(
                root, train_file, val_file, test_folder,  mode, selected_attrs, logger, transform=val_trf, mask_input_bg=mask_input_bg, use_illum=use_illum)
            self.test_loader = data.DataLoader(
                test_set, batch_size=batch_size, shuffle=False, num_workers=1)
            self.test_iterations = int(math.ceil(len(test_set) / batch_size))

    def setup_transforms(self):

        # basic transform to put at the right size
        val_trf = Compose([
            Resize(self.image_size),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5, 0), std=(0.5, 0.5, 0.5, 1))
        ])

        high_resolution = 512
        if self.data_augmentation:
            
            train_trf = T.Compose([
                Resize(high_resolution),  # suppose the dataset is of size 256
                RandomHorizontalFlip(0.5),
                RandomVerticalFlip(0.5),
                Random180DegRot(0.5),
                Random90DegRotClockWise(0.5),
                Albumentations(100, 100, 0.5),  # change in color
                RandomCrop(size=self.crop_size),
                RandomResize(low=high_resolution,
                               high=int(high_resolution*1.1718)),
                CenterCrop(high_resolution),
                val_trf,
            ])
        else:
            train_trf = Compose([
                val_trf,
            ])

        return train_trf, val_trf
