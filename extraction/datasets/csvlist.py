# filelist.py

import os
import csv
import math
import torch
from random import shuffle
import torch.utils.data as data
import datasets.loaders as loaders
import numpy as np

class CSVListLoader(data.Dataset):
    def __init__(self, ifile, root=None, split=1.0,
                 transform=None, loader='loader_image',
                 prefetch=False):

        self.root = root
        self.ifile = ifile
        self.split = split
        self.prefetch = prefetch
        self.transform = transform
        if loader is not None:
            self.loader = getattr(loaders, loader)

        self.nattributes = 0
        datalist = []
        classname = []
        if ifile is not None:
            with open(ifile, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter='\t')
                for row in reader:
                    if self.nattributes <= len(row):
                        self.nattributes = len(row)
                    datalist.append(row)
                    classname.append(row[1])
            csvfile.close()
        else:
            datalist = []

        if (self.split < 1.0) & (self.split > 0.0):
            if len(datalist) > 0:
                datalist = shuffle(datalist, classname)
                num = math.floor(self.split * len(datalist))
                self.data = datalist[0:num]
            else:
                self.data = []

        elif self.split == 1.0:
            if len(datalist) > 0:
                self.data = datalist
            else:
                self.data = []

        self.classname = list(set(classname))
        self.classname.sort()

        if prefetch is True:
            print('Prefetching data, feel free to stretch your legs !!')
            self.objects = []
            for index in range(len(self.data)):
                if self.root is not None:
                    path = os.path.join(self.root, self.data[index][0])
                else:
                    path = self.data[index][0]
                self.objects.append(self.loader(path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if len(self.data) > 0:
            if self.prefetch is False:
                if self.root is not None:
                    path = os.path.join(self.root, self.data[index][0])
                else:
                    path = self.data[index][0]
                image = self.loader(path)

            elif self.prefetch is True:
                image = self.objects[index]

            attributes = self.data[index]
            fmetas = attributes[0]
            attributes = attributes[1:]
            attributes[0] = self.classname.index(attributes[0])
            if len(attributes) > 1:
                attributes[1:] = [int(x) for x in attributes[1:]]
            else:
                attributes[1:] = [-1 for x in range(self.nattributes - 2)]

        if self.transform is not None:
            image = self.transform(image)

        attributes = torch.Tensor(attributes)
        return image, attributes, fmetas
