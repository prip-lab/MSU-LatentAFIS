# dataloader.py

import os
import torch
import datasets
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


class Dataloader:
    def __init__(self, args):
        self.args = args

        self.loader_input = args.loader_input
        self.loader_label = args.loader_label

        self.dataset_options = args.dataset_options

        self.split = args.split
        self.dataset_test_name = args.dataset_test
        self.dataset_train_name = args.dataset_train

        self.train_dev_percent = args.train_dev_percent
        self.test_dev_percent = args.test_dev_percent

        self.resolution = (args.resolution_wide, args.resolution_high)
        self.input_size = (args.input_wide, args.input_high)

        self.input_filename_test = args.input_filename_test
        self.label_filename_test = args.label_filename_test
        self.input_filename_train = args.input_filename_train
        self.label_filename_train = args.label_filename_train

        if self.dataset_train_name == 'LSUN':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(
                db_path=args.dataset_root_train,
                classes=['bedroom_train'],
                transform=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            )

        elif self.dataset_train_name == 'CIFAR10' or self.dataset_train_name == 'CIFAR100':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(
                root=self.args.dataset_root_train, train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(self.resolution, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            )

        elif self.dataset_train_name == 'CocoCaption' or self.dataset_train_name == 'CocoDetection':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(
                root=self.args.dataset_root_train, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            )

        elif self.dataset_train_name == 'STL10' or self.dataset_train_name == 'SVHN':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(
                root=self.args.dataset_root_train, split='train', download=True,
                transform=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            )

        elif self.dataset_train_name == 'MNIST':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(
                root=self.args.dataset_root_train, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            )

        elif self.dataset_train_name == 'ImageNet':
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            self.dataset_train = datasets.ImageFolder(
                root=self.args.dataset_root_train + self.args.input_filename_train,
                transform=transforms.Compose([
                    transforms.RandomSizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            )

        elif self.dataset_train_name == 'FRGC':
            self.dataset_train = datasets.ImageFolder(
                root=self.args.dataset_root_train + self.args.input_filename_train,
                transform=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            )

        elif self.dataset_train_name == 'Folder':
            self.dataset_train = datasets.ImageFolder(
                root=os.path.join(self.args.dataset_root_train, self.args.input_filename_train),
                transform=transforms.Compose(
                    self.preprocess(self.args.preprocess_train),
                )
            )

        elif self.dataset_train_name == 'FileListLoader':
            self.dataset_train = datasets.FileListLoader(
                self.input_filename_train, self.args.dataset_root_train,
                self.split,
                transform=transforms.Compose(
                    self.preprocess(self.args.preprocess_train)
                ),
                loader=self.loader_input,
            )

        elif self.dataset_train_name == 'FolderListLoader':
            self.dataset_train = datasets.FileListLoader(
                self.input_filename_train, self.label_filename_train,
                self.split_train, self.split_test, train=True,
                transform_train=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                transform_test=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                loader_input=self.loader_input,
                loader_label=self.loader_label,
            )

        elif self.dataset_train_name == 'CSVListLoader':
            self.dataset_train = datasets.CSVListLoader(
                self.input_filename_train, self.args.dataset_root_train, 
                self.split,
                transform=transforms.Compose(
                    self.preprocess(self.args.preprocess_train)
                ),
                loader=self.loader_input,
            )

        elif self.dataset_train_name == 'TripletDataLoader':
            self.dataset_train = datasets.TripletDataLoader(
                self.args.dataset_root_train, self.input_filename_train, 
                self.args.batch_size_image,
                transform=transforms.Compose(
                    self.preprocess(self.args.preprocess_train)
                ),
            )

        elif self.dataset_train_name == 'TripletDataLoader_ImageRetrieval':
            if "\n" in self.input_filename_train:
                self.input_filename_train = self.input_filename_train.split('\n')
            else:
                raise(Exception("Format of input filename is wrong!"))

            assert(len(self.input_filename_train) == 2)
            self.dataset_train = datasets.TripletDataLoader_ImageRetrieval(
                self.args.dataset_root_train, 
                self.input_filename_train[0], self.input_filename_train[1], 
                self.args.batch_size_image,
                transform=transforms.Compose(
                    self.preprocess(self.args.preprocess_train)
                ),
            )

        elif self.dataset_train_name == 'Featpair':
            self.dataset_train = datasets.Featpair(
                self.args.input_filename_train,
                self.args.pair_index_filename,
                self.args.if_norm,
                self.args.num_images, self.args.in_dims,
                self.args.template_filename,
                )

        elif self.dataset_train_name == 'ClassPairDataLoader':
            self.dataset_train = datasets.ClassPairDataLoader(
                self.args.input_filename_train,
                self.args.if_norm,
                self.args.batch_size_image,
                )

        elif self.dataset_train_name == 'Featarray':
            self.dataset_train = datasets.Featarray(
                self.args.input_filename_train,
                self.args.if_norm,
                self.args.num_images, self.args.in_dims,
                )

        elif self.dataset_train_name == None:
            print("No training data assigned!")
        else:
            raise(Exception("Unknown Dataset"))

        if self.dataset_test_name == 'LSUN':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(
                db_path=args.dataset_root_test, classes=['bedroom_val'],
                transform=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            )

        elif self.dataset_test_name == 'CIFAR10' or self.dataset_test_name == 'CIFAR100':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(
                root=self.args.dataset_root_test, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            )

        elif self.dataset_test_name == 'CocoCaption' or self.dataset_test_name == 'CocoDetection':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(
                root=self.args.dataset_root_test, train=False, download=True,
                transform=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            )

        elif self.dataset_test_name == 'STL10' or self.dataset_test_name == 'SVHN':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(
                root=self.args.dataset_root_test, split='test', download=True,
                transform=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            )

        elif self.dataset_test_name == 'MNIST':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(
                root=self.args.dataset_root_test, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            )

        elif self.dataset_test_name == 'ImageNet':
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            self.dataset_test = datasets.ImageFolder(
                root=self.args.dataset_root_test + self.args.input_filename_test,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])
            )

        elif self.dataset_test_name == 'FRGC':
            self.dataset_test = datasets.ImageFolder(
                root=self.args.dataset_root_test + self.args.input_filename_test,
                transform=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            )

        elif self.dataset_test_name == 'Folder':
            self.dataset_test = datasets.ImageFolder(
                root=os.path.join(self.args.dataset_root_test, self.args.input_filename_test),
                transform=transforms.Compose(
                    self.preprocess(self.args.preprocess_test),
                )
            )

        elif self.dataset_test_name == 'FileListLoader':
            self.dataset_test = datasets.FileListLoader(
                self.input_filename_test, self.args.dataset_root_test,
                self.split,
                transform=transforms.Compose(
                    self.preprocess(self.args.preprocess_test)
                ),
                loader=self.loader_input,
            )

        elif self.dataset_test_name == 'FolderListLoader':
            self.dataset_test = datasets.FileListLoader(
                self.input_filename_test, self.label_filename_test,
                self.split_train, self.split_test, train=False,
                transform_test=transforms.Compose(
                    self.preprocess(self.args.preprocess_test)
                ),
                loader_input=self.loader_input,
                loader_label=self.loader_label,
            )
        
        elif self.dataset_test_name == 'CSVListLoader':
            self.dataset_test = datasets.CSVListLoader(
                self.input_filename_test, self.args.dataset_root_test,
                self.split, 
                transform=transforms.Compose(
                    self.preprocess(self.args.preprocess_test)
                ),
                loader=self.loader_input,
            )

        elif self.dataset_test_name == 'TripletDataLoader':
            self.dataset_test = datasets.TripletDataLoader(
                self.args.dataset_root_test, self.input_filename_test, 
                self.args.batch_size_image,
                transform=transforms.Compose(
                    self.preprocess(self.args.preprocess_test)
                ),
            )

        elif self.dataset_test_name == 'TripletDataLoader_ImageRetrieval':
            if "\n" in self.input_filename_test:
                self.input_filename_test = self.input_filename_test.split('\n')
            else:
                raise(Exception("Format of input filename is wrong!"))

            assert(len(self.input_filename_test) == 2)
            self.dataset_test = datasets.TripletDataLoader_ImageRetrieval(
                self.args.dataset_root_test, 
                self.input_filename_test[0], self.input_filename_test[1],
                self.args.batch_size_image,
                transform=transforms.Compose(
                    self.preprocess(self.args.preprocess_test)
                ),
            )

        elif self.dataset_test_name == 'Featarray':
            self.dataset_test = datasets.Featarray(
                self.args.input_filename_test,
                self.args.if_norm,
                self.args.num_images, self.args.in_dims,
                )

        elif self.dataset_test_name == None:
            print("No testing data assigned!")
        else:
            raise(Exception("Unknown Dataset"))

    def preprocess(self, preprocess):
        process_list = []
        keys = [
            'Resize',
            'CenterCrop',
            'RandomCrop',
            'RandomHorizontalFlip',
            'RandomVerticalFlip',
            'RandomRotation',
            'ToTensor',
            'Normalize',
        ]
        for key in keys:
            if key in preprocess.keys():
                if key == keys[0]:
                    process_list.append(getattr(transforms, key)(self.input_size))
                elif key == keys[1] or key == keys[2]:
                    process_list.append(getattr(transforms, key)(self.resolution))
                elif key == keys[3] or key == keys[4]:
                    process_list.append(getattr(transforms, key)())
                elif key == keys[5]:
                    process_list.append(getattr(transforms, key)(preprocess[key]))
                elif key == keys[6]:
                    process_list.append(getattr(transforms, key)())
                else:
                    process_list.append(transforms.Normalize(preprocess[key][0], preprocess[key][1]))
        return process_list

    def create(self, flag=None):
        dataloader = {}
        if flag == "Train":
            dataloader['train'] = torch.utils.data.DataLoader(
                self.dataset_train,
                batch_size=self.args.batch_size,
                num_workers=int(self.args.nthreads),
                shuffle=True, pin_memory=True
            )
            return dataloader

        elif flag == "Test":
            dataloader['test'] = torch.utils.data.DataLoader(
                self.dataset_test,
                batch_size=self.args.batch_size,
                num_workers=int(self.args.nthreads),
                shuffle=False, pin_memory=True
            )
            return dataloader

        elif flag is None:
            dataloader['train'] = torch.utils.data.DataLoader(
                self.dataset_train,
                batch_size=self.args.batch_size,
                num_workers=int(self.args.nthreads),
                shuffle=True, pin_memory=True
            )

            dataloader['test'] = torch.utils.data.DataLoader(
                self.dataset_test,
                batch_size=self.args.batch_size,
                num_workers=int(self.args.nthreads),
                shuffle=False, pin_memory=True)
            return dataloader

    def create_devsets(self, flag=None):
        dataloader = {}
        if flag is None or flag == "train":
            train_len = len(self.dataset_train)
            train_cut_index = int(train_len * self.train_dev_percent)
            train_random_indices = list(torch.randperm(train_len))
            train_indices = train_random_indices[train_cut_index:]
            train_dev_indices = train_random_indices[:train_cut_index]

            train_sampler = SubsetRandomSampler(train_indices)
            train_dev_sampler = SubsetRandomSampler(train_dev_indices)

            dataloader['train'] = torch.utils.data.DataLoader(
                self.dataset_train, batch_size=self.args.batch_size,
                sampler=train_sampler, num_workers=self.args.nthreads,
                pin_memory=True
            )

            dataloader['train_dev'] = torch.utils.data.DataLoader(
                self.dataset_train, batch_size=self.args.batch_size,
                sampler=train_dev_sampler, num_workers=self.args.nthreads,
                pin_memory=True
            )

        if flag is None or flag == "test":
            test_len = len(self.dataset_test)
            test_cut_index = int(test_len * self.test_dev_percent)
            test_random_indices = list(torch.randperm(test_len))
            test_indices = test_random_indices[test_cut_index:]
            test_dev_indices = test_random_indices[:test_cut_index]

            test_sampler = SubsetRandomSampler(test_indices)
            test_dev_sampler = SubsetRandomSampler(test_dev_indices)

            dataloader['test_dev'] = torch.utils.data.DataLoader(
                self.dataset_test, batch_size=self.args.batch_size,
                sampler=test_dev_sampler, num_workers=self.args.nthreads,
                pin_memory=True
            )

            dataloader['test'] = torch.utils.data.DataLoader(
                self.dataset_test, batch_size=self.args.batch_size,
                sampler=test_sampler, num_workers=self.args.nthreads,
                pin_memory=True
            )

        return dataloader
