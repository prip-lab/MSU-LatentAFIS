# filelist.py

import os
import math
import utils as utils
import torch.utils.data as data
import datasets.loaders as loaders

import torch


class FileListLoader(data.Dataset):
    def __init__(self, ifile, root=None, split=1.0,
        transform=None, loader='loader_image'):

        self.ifile = ifile
        self.root = root
        self.split = split
        self.transform = transform
        if loader is not None:
            self.loader = getattr(loaders, loader)

        if ifile is not None:
            lines = utils.readtextfile(ifile)
            imagelist = []
            for x in lines:
                x = x.rstrip('\n')
                temp = [x]
                temp.append(os.path.basename(os.path.dirname(x)))
                imagelist.append(temp)

            labellist = [x[1] for x in imagelist]

        else:
            imagelist = []

        if (self.split < 1.0) & (self.split > 0.0):
            if len(imagelist) > 0:
                imagelist = shuffle(imagelist, labellist)
                num = math.floor(self.split * len(imagelist))
                self.images = imagelist[0:num]
            else:
                self.images = []

        elif self.split == 1.0:
            if len(imagelist) > 0:
                self.images = imagelist
            else:
                self.images = []
            if len(labellist) > 0:
                self.classname = labellist
            else:
                self.classname = []

        self.classname.sort()


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if len(self.images) > 0:
            if self.root is not None:
                path = os.path.join(self.root,self.images[index][0])
            else:
                path = self.images[index][0]
            image = self.loader(path)

            label = self.classname.index(self.images[index][1])
            fmeta = path

            if self.transform is not None:
                image = self.transform(image)

        else:
            image = []
            label = None
            fmeta = None        

        return image, label, fmeta
