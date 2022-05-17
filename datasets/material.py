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
# from utils.im_util import get_alpha_channel
import cv2
from matplotlib import pyplot as plt
from torchvision.utils import make_grid, save_image

def make_dataset(root, train_file, test_file, mode, selected_attrs,att_neg=False,thres_edition=1.0):
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

    if mode == 'train':
        lines = lines_train[1:]
        print('Train Samples:', len(lines))
    if mode == 'val':  # put in first half a batch of test images, half of training images
        lines = lines_test[1:]
        print('Validation Samples:', len(lines))
    if mode == 'test':
        # np.random.shuffle(lines_test)
        # lines_test=[line for line in lines_test if ("@mat" in line)]
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

            if att_neg:
                mat_attr.append(float(values[idx]) * (2*thres_edition)- thres_edition)
                
            else:
                mat_attr.append(float(values[idx])* thres_edition)
        
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
    dset = __import__('datasets.transforms2', globals(), locals())
    T = dset.transforms2
    image = T.ToTensor()(T.Resize(256)(image))
    mask = torch.where(image[3,:,:] <= 0.1, 1,0)
    save_image(mask.float(),'mask.png')
    back_ground = image[:3,:,:] * mask

    return back_ground,mask


class MaterialDataset(data.Dataset):
    def __init__(self, root, train_file, test_file, mode, selected_attrs, disentangled=False, transform=None, mask_input_bg=True, use_illum=False,att_neg=False,thres_edition=1.0):
        items = make_dataset(root, train_file, test_file, mode, selected_attrs,att_neg,thres_edition)

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

        if (self.use_illum):
            #illum = cv2.imread(os.path.join(self.root, "illum", self.files[index]), -1)
            #if True:
            illum,mask = extract_highlights(image)


        if self.transform is not None:
            image, normals = self.transform(image, normals)

        # mask the normals
        normals = normals*normals[3:]
        # mask the input image if asked
        if self.mask_input_bg:
            image = image*image[3:]

        if self.mode == 'test':
            filename = self.files[index].split('/')[1][:-4]
            return image, torch.FloatTensor(mat_attr) , filename, illum, mask
        else:
            return image, torch.FloatTensor(mat_attr)

    def __len__(self):
        return len(self.files)


# the main dataloader
class MaterialDataLoader(object):
    def __init__(self, root, train_file, test_file, mode, selected_attrs, crop_size=None, image_size=128, batch_size=16, data_augmentation=True, mask_input_bg=True, use_illum=False,att_neg=False,thres_edition=1.0):
        if mode not in ['train', 'test','latent']:
            return

        self.root = root
        self.data_augmentation = data_augmentation
        self.image_size = image_size
        self.crop_size = crop_size

        # setup the dataloaders
        if mode=='test':
            use_illum=True
            self.data_augmentation = False
        train_trf, val_trf = self.setup_transforms(use_illum)

        if mode == 'train' or mode == 'latent':
            print("loading data")
            val_set = MaterialDataset(
                root, train_file, test_file, 'val', selected_attrs, transform=val_trf, mask_input_bg=mask_input_bg, use_illum=use_illum,att_neg=att_neg,thres_edition=thres_edition)
            self.val_loader = data.DataLoader(
                val_set, batch_size=batch_size, shuffle=False, num_workers=4)
            self.val_iterations = int(math.ceil(len(val_set) / batch_size))

            train_set = MaterialDataset(
                root, train_file, test_file, 'train', selected_attrs, transform=train_trf, mask_input_bg=mask_input_bg, use_illum=use_illum,att_neg=att_neg,thres_edition=thres_edition)
            self.train_loader = data.DataLoader(
                train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            self.train_iterations = int(math.ceil(len(train_set) / batch_size))
        else:
            batch_size = 1
            test_set = MaterialDataset(
                root, train_file, test_file,  'test', selected_attrs, transform=val_trf, mask_input_bg=mask_input_bg, use_illum=use_illum,att_neg=att_neg,thres_edition=thres_edition)
            self.test_loader = data.DataLoader(
                test_set, batch_size=batch_size, shuffle=False, num_workers=1)
            self.test_iterations = int(math.ceil(len(test_set) / batch_size))

    def setup_transforms(self, use_illum):
        global T
        if (False):
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
        if self.image_size != 512:
            print('new size')
            original_size = 512
        if self.data_augmentation:
            
            train_trf = T.Compose([
                T.Resize(original_size),  # suppose the dataset is of size 256
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.5),
                T.Random180DegRot(0.5),
                T.Random90DegRotClockWise(0.5),
                T.Albumentations(100, 100, 0.5),  # change in color
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


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    use_illum = True
    global T
    if (use_illum):
        dset = __import__('transforms3', globals(), locals())
        T = dset
    else:
        dset = __import__('transforms2', globals(), locals())
        T = dset
    val_trf = T.Compose([
        T.CenterCrop(240),
        T.Resize(128),
        T.ToTensor()
        # T.Normalize(mean=(0.5, 0.5, 0.5,0), std=(0.5, 0.5, 0.5,1))
    ])
    trf = T.Compose([
        T.Resize(256),  # suppose the dataset is of size 256
        # T.RandomHorizontalFlip(0.5), #TODO recode for normals
        # T.RandomVerticalFlip(0.5), #TODO recode for normals
        T.RandomCrop(size=240),
        T.RandomResize(low=256, high=300),
        # T.RandomRotation(degrees=(-5, 5)), #TODO recode for normals
        val_trf,
    ])

    data_root = '/Users/delanoy/Documents/postdoc/project1_material_networks/dataset/renders_by_geom_ldr/network_dataset/'
    data = MaterialDataset(root=data_root,
                           mode='test',
                           selected_attrs=['glossy'],
                           transform=trf,
                           mask_input_bg=True, use_illum=use_illum)
    # sampler = DisentangledSampler(data, batch_size=8)
    loader = DataLoader(data,  batch_size=1, shuffle=True)
    iter(loader)
    for imgs, normal, illum, attr in loader:
        # from matplotlib import pyplot as plt

        # # for i in range(len(imgs)):
        # #     print (infos[i])
        for i in range(len(imgs)):
            # print(illum[i][:,0,0])
            plt.subplot(1, 3, 1)
            plt.imshow(imgs[i].permute(1, 2, 0).detach().cpu(), cmap='gray')
            plt.subplot(1, 3, 2)
            plt.imshow(normal[i].permute(1, 2, 0).detach().cpu(), cmap='gray')
            plt.subplot(1, 3, 3)
            plt.imshow(illum[i].permute(1, 2, 0).detach().cpu(), cmap='gray')
            plt.show()
        print("done")
        # input('press key to continue plotting')
