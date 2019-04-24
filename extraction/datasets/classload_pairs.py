import os
import torch
import torch.utils.data as data
import datasets.loaders as loaders
import numpy as np

IMG_EXTENTIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENTIONS)
def is_feat_file(filename):
    return filename.endswith('.npy')

def make_dataset(datafolder):
    datadict = {}
    classes = os.listdir(datafolder)
    classes.sort()

    for i in range(len(classes)):
        class_folder = os.path.join(datafolder, classes[i])
        if os.path.isdir(class_folder):
            datadict[classes[i]] = []
            fnames = os.listdir(class_folder)
            nrof_images = 0
            for fname in fnames:
                if is_feat_file(fname):
                    nrof_images += 1
                    fpath = os.path.join(class_folder, fname)
                    datadict[classes[i]].append(fpath)
    classes = list(datadict)
    return datadict, classes

class Iterator(object):
    def __init__(self, imagelist):
        self.length = len(imagelist)
        self.temp = torch.randperm(self.length)
        self.current = 0

    def __iter__(self):
        return self

    def next(self):
        value = self.temp[self.current]
        self.current += 1
        if self.current == self.length:
            self.current = 0
            self.temp = torch.randperm(self.length)
        return value

class ClassPairDataLoader(data.Dataset):
    def __init__(self, datafolder, if_norm, batch_size_image):
        self.if_norm = if_norm
        self.batch_size_image = batch_size_image
        datadict, classes = make_dataset(datafolder)
        if len(datadict) == 0:
            raise(RuntimeError('No images found'))
        else:
            self.classes = classes
            self.datadict = datadict

        self.num_classes = len(self.classes)
        self.iterdict_pos = {}
        self.iterdict_neg = {}
        for i in range(self.num_classes):
            self.iterdict_pos[i] = Iterator(datadict[self.classes[i]])
            self.iterdict_neg[i] = Iterator(datadict[self.classes[i]])

    def __len__(self):
        return self.num_classes

    def __getitem__(self, index):
        images = []
        num_images = self.batch_size_image;
        for i in range(num_images):
            if i == 0:
                ind = self.iterdict_neg[index].next()
            else:
                ind = self.iterdict_pos[index].next()
            imgpath = self.datadict[self.classes[index]][ind]
            image = np.load(imgpath)
            # normalization
            if self.if_norm:
                norm = float(np.linalg.norm(image))
                image = image/norm

            image = torch.Tensor(image)
            images.append(image)
        return images