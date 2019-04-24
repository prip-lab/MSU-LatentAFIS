# triplet.py

import os
import utils as utils

import torch
import torch.utils.data as data
import datasets.loaders as loaders


def make_dataset(ifile):
    tmpdata = utils.readcsvfile(ifile, "\t")
    classes = []
    for i in range(len(tmpdata)):
        classes.append(tmpdata[i][1])
    classes = list(set(classes))
    classes.sort()

    datalist = {}
    for i in range(len(classes)):
        datalist[i] = []

    for i in range(len(tmpdata)):
        row = tmpdata[i]
        datalist[classes.index(row[1])].append(row)

    return datalist, classes


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


class TripletDataLoader(data.Dataset):
    def __init__(
        self, root, ifile, num_images, transform=None,
        loader_input='loader_image', loader_label=None,
        prefetch=False
    ):

        self.root = root
        self.prefetch = prefetch
        self.num_images = num_images
        datalist, classes = make_dataset(ifile)
        if len(datalist) == 0:
            raise(RuntimeError("No images found"))

        if loader_input is not None:
            self.loader_input = getattr(loaders, loader_input)
        if loader_label is not None:
            self.loader_label = getattr(loaders, loader_label)

        self.transform = transform
        if len(datalist) > 0:
            self.classes = classes
            self.datalist = datalist

        self.num_classes = len(self.classes)
        self.class_iter = {}
        for i in range(self.num_classes):
            self.class_iter[i] = Iterator(self.datalist[i])

        if self.prefetch is True:
            print('Prefetching data, feel free to stretch your legs !!')
            self.objects = {}
            for index in range(self.num_classes):
                self.objects[i] = []
                for ind in range(len(self.datalist[index])):
                    name = self.datalist[index][ind][0]
                    name = os.path.join(self.root, name)
                    self.objects[i].append(self.loader_input(name))
                print('Loading %d out of %d' % (index, self.num_classes))

    def __len__(self):
        return self.num_classes

    def __getitem__(self, index):
        images = []
        fmetas = []
        labels = []
        for i in range(self.num_images):
            ind = self.class_iter[index].next()
            if self.prefetch is False:
                name = self.datalist[index][ind][0]
                name = os.path.join(self.root, name)
                image = self.loader_input(name)
            elif self.prefetch is True:
                image = self.objects[index][ind]
            images.append(self.transform(image))
            fmetas.append(self.datalist[index][ind][0])
            label = self.classes.index(self.datalist[index][ind][1])
            labels.append(label)
        
        return images, labels
