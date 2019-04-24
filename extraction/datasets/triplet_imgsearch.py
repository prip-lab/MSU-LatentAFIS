# triplet.py

import os
import utils as utils

import torch
import torch.utils.data as data
import datasets.loaders as loaders


def make_dataset(ifile_query, ifile_gallery):
    tmpdata = utils.readcsvfile(ifile_query, "\t")
    tmpdata2 = utils.readcsvfile(ifile_gallery, "\t")
    classes = []
    for i in range(len(tmpdata)):
        classes.append(tmpdata[i][1])
    classes = list(set(classes))
    classes.sort()

    datalist_query = {}
    datalist_gallery = {}
    for i in range(len(classes)):
        datalist_query[i] = []
        datalist_gallery[i] = []

    for i in range(len(tmpdata)):
        row = tmpdata[i]
        datalist_query[classes.index(row[1])].append(row)

    for i in range(len(tmpdata2)):
        row = tmpdata2[i]
        datalist_gallery[classes.index(row[1])].append(row)

    return datalist_query, datalist_gallery, classes


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


class TripletDataLoader_ImageRetrieval(data.Dataset):
    def __init__(
        self, root, ifile_query, ifile_gallery, num_images, transform=None,
        loader_input='loader_image', loader_label=None,
        prefetch=False
    ):

        self.root = root
        self.prefetch = prefetch
        self.num_images = num_images
        datalist_query, datalist_gallery, classes = make_dataset(
            ifile_query, ifile_gallery)
        if len(datalist_query) == 0 or len(datalist_gallery) == 0:
            raise(RuntimeError("No images found"))

        if loader_input is not None:
            self.loader_input = getattr(loaders, loader_input)
        if loader_label is not None:
            self.loader_label = getattr(loaders, loader_label)

        self.transform = transform
        if len(datalist_query) > 0 and len(datalist_gallery) > 0:
            self.classes = classes
            self.datalist_query = datalist_query
            self.datalist_gallery = datalist_gallery

        self.num_classes = len(self.classes)
        self.class_iter_query = {}
        self.class_iter_gallery = {}
        for i in range(self.num_classes):
            self.class_iter_query[i] = Iterator(self.datalist_query[i])
            self.class_iter_gallery[i] = Iterator(self.datalist_gallery[i])

        if self.prefetch is True:
            print('Prefetching data, feel free to stretch your legs !!')
            self.objects_query = {}
            for index in range(self.num_classes):
                self.objects_query[i] = []
                for ind in range(len(self.datalist_query[index])):
                    name = self.datalist_query[index][ind][0]
                    name = os.path.join(self.root, name)
                    self.objects_query[i].append(self.loader_input(name))
                print('Loading %d out of %d' % (index, self.num_classes))

            self.objects_gallery = {}
            for index in range(self.num_classes):
                self.objects_gallery[i] = []
                for ind in range(len(self.datalist_gallery[index])):
                    name = self.datalist_gallery[index][ind][0]
                    name = os.path.join(self.root, name)
                    self.objects_gallery[i].append(self.loader_input(name))
                print('Loading %d out of %d' % (index, self.num_classes))

    def __len__(self):
        return self.num_classes

    def __getitem__(self, index):
        images_query = []
        images_gallery = []
        fmetas_query = []
        fmetas_gallery = []
        labels = torch.zeros(self.num_images)
        assert(self.datalist_query[index][0][1] == self.datalist_gallery[index][0][1])

        for i in range(self.num_images):
            ind1 = self.class_iter_query[index].next()
            ind2 = self.class_iter_gallery[index].next()
            if self.prefetch is False:
                name = self.datalist_query[index][ind1][0]
                name = os.path.join(self.root, name)
                image_query = self.loader_input(name)
                name = self.datalist_gallery[index][ind2][0]
                name = os.path.join(self.root, name)
                image_gallery = self.loader_input(name)
            elif self.prefetch is True:
                image_query = self.objects_query[index][ind1]
                image_gallery = self.objects_gallery[index][ind2]
            images_query.append(self.transform(image_query))
            images_gallery.append(self.transform(image_gallery))
            fmetas_query.append(self.datalist_query[index][ind1][0])
            fmetas_gallery.append(self.datalist_gallery[index][ind2][0])
            label = self.classes.index(self.datalist_query[index][ind1][1])
            labels[i] = label

        images_query = torch.stack(images_query)
        images_gallery = torch.stack(images_gallery)

        return images_query, images_gallery, labels, fmetas_query, fmetas_gallery
