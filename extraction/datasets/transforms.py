# transforms.py

from __future__ import division

import math
import types
import torch
import random
import numbers
import numpy as np
import scipy as sp
import torchvision
from PIL import Image, ImageOps
from datasets import loaders
import utils


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        for t in self.transforms:
            input = t(input)

        return input


class ToTensor(object):
    """Convert a dictionary of type ``PIL.Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self):
        self.toTensor = torchvision.transforms.ToTensor()

    def __call__(self, input):
        """
        Args:
            input (a dictionary containing PIL.Image or numpy.ndarray elements): Dict to be converted to tensor.

        Returns:
            Dict: Tensorized/Converted dictionay.
        """
        for key in input.keys():
            value = input[key]
            if isinstance(value, Image.Image):
                input[key] = self.toTensor(value)
            elif isinstance(value, np.ndarray):
                input[key] = torch.from_numpy(value).float()
            elif type(input[key].__module__ == 'torch'):
                # assumed to be a Tensor
                pass
            else:
                raise ("Unsupported input type, please update the ToTensor "
                "class")
        return input


class ToPILImage(object):
    """Converts a torch.*Tensor of range [0, 1] and shape C x H x W
    or numpy ndarray of dtype=uint8, range[0, 255] and shape H x W x C
    to a PIL.Image of range [0, 255]
    """
    def __call__(self, input):
        if isinstance(input['img'], np.ndarray):
            # handle numpy array
            input['img'] = Image.fromarray(input['img'])
        else:
            npimg = input['img'].mul(255).byte().numpy()
            npimg = np.transpose(npimg, (1, 2, 0))
            input['img'] = Image.fromarray(npimg)
        return input


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, input):
        # TODO: make efficient
        for t, m, s in zip(input['img'], self.mean, self.std):
            t.sub_(m).div_(s)
        return input


class Scale(object):
    """Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, input):
        w, h = input['img'].size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return input
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            input['img'] = input['img'].resize((ow, oh), self.interpolation)
            return input
        else:
            oh = self.size
            ow = int(self.size * w / h)
            input['img'] = input['img'].resize((ow, oh), self.interpolation)
            return input


class CenterCrop(object):
    """Crops the given PIL.Image at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, input):
        w, h = input['img'].size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        input['img'] = input['img'].crop((x1, y1, x1 + tw, y1 + th))
        return input


class Pad(object):
    """Pads the given PIL.Image on all sides with the given "pad" value"""
    def __init__(self, padding, fill=0):
        assert isinstance(padding, numbers.Number)
        assert isinstance(fill, numbers.Number)
        self.padding = padding
        self.fill = fill

    def __call__(self, input):
        input['img'] = ImageOps.expand(input['img'], border=self.padding, fill=self.fill)
        return input


class Lambda(object):
    """Applies a lambda as a transform."""
    def __init__(self, lambd):
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def __call__(self, input):
        input['img'] = self.lambd(input['img'])
        return input


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, input):
        if self.padding > 0:
            input['img'] = ImageOps.expand(input['img'], border=self.padding, fill=0)

        w, h = input['img'].size
        th, tw = self.size
        if w == tw and h == th:
            return input

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        input['img'] = input['img'].crop((x1, y1, x1 + tw, y1 + th))
        return input


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, input):
        if random.random() < 0.5:
            input['img'] = input['img'].transpose(Image.FLIP_LEFT_RIGHT)
            input['tgt'] = input['tgt'].transpose(Image.FLIP_LEFT_RIGHT)
            input['loc'][0] = input['loc'][0] - math.ceil(input['img'].size[0] / 2)
        return input


class RandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, input):
        for attempt in range(10):
            area = input['img'].size[0] * input['img'].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= input['img'].size[0] and h <= input['img'].size[1]:
                x1 = random.randint(0, input['img'].size[0] - w)
                y1 = random.randint(0, input['img'].size[1] - h)

                input['img'] = input['img'].crop((x1, y1, x1 + w, y1 + h))
                assert(input['img'].size == (w, h))
                input['img'] = input['img'].resize((self.size, self.size), self.interpolation)
                return input

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(input))


class NormalizeLandmarks(object):
    """ max-min normalization of landmarks to range [-1,1]"""
    def __init__(self, xsize, ysize):
        self.xsize = xsize
        self.ysize = ysize

    def __call__(self, input):
        valid_points = [v for v in input['loc'] if v[0] != 0 and v[1] != 0]
        mean = np.mean(valid_points,axis = 0)
        for i in range(input['loc'].shape[0]):
            input['loc'][i][0] = -1 + (input['loc'][i][0] * 2. )/(inputx_res)
            input['loc'][i][1] = -1 + (input['loc'][i][1] * 2. )/(inputy_res)
        
        return input


class AffineCrop(object):
    def __init__(self,nlandmark,ix,iy,ox,oy,rangle=0,rscale=0,rtrans=0,gauss=1):
        self.rangle=rangle
        self.rscale=rscale
        self.rtrans=rtrans
        self.nlandmark=nlandmark
        self.ix = ix
        self.iy = iy
        self.ox = ox
        self.oy = oy
        self.utils = utils
        self.gauss = gauss

    def __call__(self, input):

        angle = self.rangle*(2*torch.rand(1)[0] - 1)
        grad_angle  = angle * math.pi / 180
        scale = 1+self.rscale*(2*torch.rand(1)[0] - 1)
        transx = self.rtrans*(2*torch.rand(1)[0] - 1)
        transy = self.rtrans*(2*torch.rand(1)[0] - 1)
        
        img = input['img']
        size = img.size
        h, w = size[0], size[1]
        centerX, centerY = int(w/2), int(h/2)

        # perform rotation
        img = img.rotate(angle, Image.BICUBIC)
        # perform translation
        img = img.transform(img.size, Image.AFFINE, (1, 0, transx, 0, 1, transy))
        # perform scaling
        img = img.resize((int(math.ceil(scale*h)) , int(math.ceil(scale*w))) , Image.ANTIALIAS)

        w, h = img.size
        x1 = int(round((w - self.ix) / 2.))
        y1 = int(round((h - self.ix) / 2.))
        input['img'] = img.crop((x1, y1, x1 + self.ix, y1 + self.iy))
        
        if (np.sum(input['loc']) != 0):
            
            occ = input['occ']
            loc = input['loc']
            newloc = np.ones((3,loc.shape[1]+1))
            newloc[0:2,0:loc.shape[1]] = loc
            newloc[0,loc.shape[1]] = centerY
            newloc[1,loc.shape[1]] = centerX
            
            trans_matrix = np.array([[1,0,-1*transx],[0,1,-1*transy],[0,0,1]])
            scale_matrix = np.array([[scale,0,0],[0,scale,0],[0,0,1]])
            angle_matrix = np.array([[math.cos(grad_angle),math.sin(grad_angle),0],[-math.sin(grad_angle),math.cos(grad_angle),0],[0,0,1]])

            # perform rotation
            newloc[0,:] = newloc[0,:] - centerY
            newloc[1,:] = newloc[1,:] - centerX
            newloc = np.dot(angle_matrix, newloc)
            newloc[0,:] = newloc[0,:] + centerY
            newloc[1,:] = newloc[1,:] + centerX
            # perform translation
            newloc = np.dot(trans_matrix, newloc)
            # perform scaling
            newloc = np.dot(scale_matrix, newloc)

            newloc[0,:] = newloc[0,:] - y1
            newloc[1,:] = newloc[1,:] - x1
            input['loc'] = newloc[0:2,:]
            
            for i in range(input['loc'].shape[1]):
                if ~((input['loc'][0, i] == np.nan) & (input['loc'][1,i] == np.nan)):
                    if ((input['loc'][0, i] < 0) | (input['loc'][0, i] > self.iy) | (input['loc'][1, i] < 0) | (input['loc'][1, i] > self.ix)):
                        input['loc'][:, i] = np.nan
                        input['occ'][i] = 0

        # generate heatmaps
        input['tgt'] = np.zeros((self.nlandmark+1, self.ox, self.oy))
        for i in range(self.nlandmark):
            if  (not np.isnan(input['loc'][:,i][0]) and not np.isnan(input['loc'][:,i][1])):
                tmp = self.utils.gaussian(np.array([self.ix,self.iy]),input['loc'][:,i],self.gauss)
                scaled_tmp = sp.misc.imresize(tmp, [self.ox, self.oy])
                scaled_tmp = (scaled_tmp - min(scaled_tmp.flatten()) ) / ( max(scaled_tmp.flatten()) - min(scaled_tmp.flatten()))
            else:
                scaled_tmp = np.zeros([self.ox,self.oy])
            input['tgt'][i] = scaled_tmp

        tmp = self.utils.gaussian(np.array([self.iy, self.ix]), input['loc'][:, -1], 4 * self.gauss)
        scaled_tmp = sp.misc.imresize(tmp, [self.ox, self.oy])
        scaled_tmp = (scaled_tmp - min(scaled_tmp.flatten())) / (max(scaled_tmp.flatten()) - min(scaled_tmp.flatten()))
        input['tgt'][self.nlandmark] = scaled_tmp

        return input


class AffineCropNGenerateHeatmap(object):
    def __init__(self, image_resolution, heatmap_resolution,
                 rangle=0, rscale=0, rtrans=0, gauss=1, keep_landmarks_visible=False):

        self.rangle = rangle
        self.rscale = rscale
        self.rtrans = rtrans
        self.image_resolution = image_resolution
        self.ix = image_resolution[0]
        self.iy = image_resolution[1]
        self.gauss = gauss
        self.keep_landmarks_visible = keep_landmarks_visible
        self.toHeatmaps = ToHeatmaps(heatmap_resolution, gauss)

    def __call__(self, input):

        def _just_resize():
            img = input['img']
            w, h = img.size

            # perform scaling
            input['img'] = img.resize((self.ix, self.iy), Image.ANTIALIAS)

            if np.sum(input['loc']) != 0:
                loc = input['loc']
                loc[0, :] = loc[0, :] * self.ix / w
                loc[1, :] = loc[1, :] * self.iy / h
                input['loc'] = loc

        def _transform():
            angle = self.rangle * (2 * torch.rand(1)[0] - 1)
            grad_angle = angle * math.pi / 180
            scale = 1 + self.rscale * (2 * torch.rand(1)[0] - 1)
            transx = self.rtrans * (2 * torch.rand(1)[0] - 1)
            transy = self.rtrans * (2 * torch.rand(1)[0] - 1)

            img = input['img']
            w, h = img.size
            centerX, centerY = w // 2, h // 2

            # perform rotation
            img = img.rotate(angle, Image.BICUBIC)
            # perform translation
            img = img.transform(img.size, Image.AFFINE,
                                (1, 0, transx, 0, 1, transy))
            # perform scaling
            img = img.resize((int(math.ceil(scale * h)),
                              int(math.ceil(scale * w))),
                             Image.ANTIALIAS)

            w, h = img.size
            x1 = round((w - self.ix) // 2)
            y1 = round((h - self.iy) // 2)
            input['img'] = img.crop((x1, y1, x1 + self.ix, y1 + self.iy))

            if np.sum(input['loc']) != 0:
                loc = input['loc']

                newloc = np.ones((3, loc.shape[1]))
                newloc[0:2, :] = loc

                trans_matrix = np.array([[1,0,-1*transx], [0,1,-1*transy], [0,0,1]])
                scale_matrix = np.array([[scale,0,0], [0,scale,0], [0,0,1]])
                angle_matrix = np.array([
                    [math.cos(grad_angle),math.sin(grad_angle),0],
                    [-math.sin(grad_angle),math.cos(grad_angle),0],
                    [0,0,1]])

                # perform rotation
                newloc[0,:] = newloc[0,:] - centerY
                newloc[1,:] = newloc[1,:] - centerX
                newloc = np.dot(angle_matrix, newloc)
                newloc[0,:] = newloc[0,:] + centerY
                newloc[1,:] = newloc[1,:] + centerX
                # perform translation
                newloc = np.dot(trans_matrix, newloc)
                # perform scaling
                newloc = np.dot(scale_matrix, newloc)

                newloc[0,:] = newloc[0,:] - y1
                newloc[1,:] = newloc[1,:] - x1
                input['loc'] = newloc[0:2,:]

                for i in range(input['loc'].shape[1]):
                    if not np.isnan(input['loc'][:, i]).any():
                        if np.any(input['loc'][:, i] < 0) or \
                                        input['loc'][0,i] > self.iy or \
                                        input['loc'][1,i] > self.ix:
                            input['loc'][:, i] = np.nan
                            # TODO: fill the surrounding with normal noise
                            input['occ'][0, i] = 0

        # FIXME: create multiple images for the same sample with different occluded blocks for testing purposes
        # input['im'][:, 10:40, 22:50] = 0

        # adding one more at the end for the center landmark
        # add the center of image as the last landmark
        h, w = input['img'].size
        input['loc'] = np.hstack((input['loc'], np.array([[w // 2], [h // 2]])))

        input['occ'] = torch.cat((input['occ'], torch.ByteTensor([[1]])), 1)
        input['mask'] = torch.cat((input['mask'], torch.ByteTensor([[1]])), 1)
        orig_img = input['img']
        orig_loc = input['loc']
        orig_occ = input['occ'].clone()
        orig_mask = input['mask'].clone()

        _transform()

        if self.keep_landmarks_visible:
            # train: making sure all landmarks are still visible, if not perform
            #        another transformation
            mask = input['mask']
            mask2D = torch.cat((mask, mask), dim=0)
            landmarks = torch.from_numpy(input['loc'])
            limit = 100
            while not (mask == mask * input['occ']).all() or utils.isnan(landmarks[mask2D]).any():
                input['img'] = orig_img
                input['loc'] = orig_loc
                input['occ'] = orig_occ.clone()
                input['mask'] = orig_mask.clone()

                _transform()

                mask = input['mask']
                mask2D = torch.cat((mask, mask), dim=0)
                landmarks = torch.from_numpy(input['loc'])

                limit -= 1
                if limit == 0:
                    input['img'] = orig_img
                    input['loc'] = orig_loc
                    input['occ'] = orig_occ.clone()
                    input['mask'] = orig_mask.clone()
                    _just_resize()
                    print('using the orignal data because even after 100 transformation, there are still occluded landmarks!!!')
                    break

        input['tgt'] = self.toHeatmaps(input['loc'], self.image_resolution)

        return input



class MakePartialBlockage(object):
    def __init__(self, filler_images, block_sizes=[25, 35, 45, 55, 65, 75, 85, 95, 105, 115]):

        self.block_sizes = block_sizes
        self.toTensor = torchvision.transforms.ToTensor()
        self.filler_image_list = [x.rstrip('\n') for x in utils.readtextfile(filler_images)]
        assert (len(self.filler_image_list) > 0)

    def _make_copy(self, sample):
        newSample = {}
        for key in sample.keys():
            newSample[key] = sample[key].clone()
        return newSample

    def __call__(self, sample):
        assert (torch.is_tensor(sample['img']))

        def _transform(newSample):
            occ = newSample['occ']
            loc = newSample['loc']
            img_size_h = newSample['img'].shape[1]
            img_size_w = newSample['img'].shape[2]

            x_max = loc[0][occ].max()
            x_min = loc[0][occ].min()
            y_max = loc[1][occ].max()
            y_min = loc[1][occ].min()

            # pick a random filler image
            filler_image = self.toTensor(loaders.loader_image(
                self.filler_image_list[
                    int(len(self.filler_image_list) * torch.rand(1)[0])]))
            filler_image_slice_y = int(
                (filler_image.shape[1] - block_size[0]) * torch.rand(1)[0])
            filler_image_slice_x = int(
                (filler_image.shape[2] - block_size[1]) * torch.rand(1)[0])

            filler_slices = [slice(filler_image_slice_y,
                                   filler_image_slice_y + block_size[0]),
                             slice(filler_image_slice_x,
                                   filler_image_slice_x + block_size[1])]

            loc_idx = 2
            block_pos_x = int(min(max(0, loc[0][occ][loc_idx]-block_size[1]/2), (img_size_w - block_size[1])))
            block_pos_y = int(min(max(0, loc[1][occ][loc_idx]-block_size[0]/2), (img_size_h - block_size[0])))
            slices = [slice(block_pos_y, block_pos_y + block_size[0]),
                      slice(block_pos_x, block_pos_x + block_size[1])]

            newSample['img'][:, slices[0], slices[1]] = filler_image[:,
                                                        filler_slices[0],
                                                        filler_slices[1]]

            for i in range(newSample['loc'].shape[1]):
                if block_pos_x <= newSample['loc'][0, i] <= block_pos_x + \
                        block_size[1] and \
                                        block_pos_y <= newSample['loc'][
                                    1, i] <= block_pos_y + block_size[0]:
                    newSample['loc'][:, i] = np.nan
                    newSample['occ'][0, i] = 0
                    newSample['tgt'][i, :, :] = 0
            return newSample

        new_sample_list = [sample]
        for i in range(len(self.block_sizes)):
            block_size = [self.block_sizes[i], self.block_sizes[i]]  # (h, w)
            new_sample_list.append(_transform(self._make_copy(sample)))

        new_sample = {}
        for key in sample.keys():
            values = []
            for i in range(len(new_sample_list)):
                values.append(new_sample_list[i][key])
            new_sample[key] = torch.stack(values)

        return new_sample


class ToHeatmaps:
    r"""Generates heatmaps for given landmarks.

        Your landmarks should be given as ( 2 x N ) where N is the number of
        landmarks in a 2D plane. The generated heatmaps will be a Tensor of
        size (N x H x W).
        """


    def __init__(self, resolution, gauss=1):
        """
            Args:
                resolution: The resoultion ( H x W ) of generated heatmap.
        """
        self.resolution = resolution
        self.gauss = gauss

    def __call__(self, landmarks, input_resolution):
        """
            Returns a Tensor which contains the generated heatmaps
            of all elements in the :attr:`landmarks` tensor.

        Args:
            landmarks (ndarray): ndarray ( 2 x N ) contains N two dimensional
            landmarks.
            input_resolution: resolution ( H x W ) is the resoultion/dimension
            in which the landmarks are given.

        Returns:
            Tensor: The generated heatmaps ( N x outputH x outputW ).
        """
        self.inputH = input_resolution[0]
        self.inputW = input_resolution[1]
        self.outputH = self.resolution[0]
        self.outputW = self.resolution[1]
        heatmaps = np.zeros((landmarks.shape[1], self.outputH, self.outputW))
        for i in range(landmarks.shape[1]-1):
            if not np.isnan(landmarks[:, i]).any():
                tmp = utils.gaussian(np.array([self.inputH, self.inputW]),
                                          landmarks[:, i], self.gauss)
                scaled_tmp = sp.misc.imresize(tmp, [self.outputH, self.outputW])
                scaled_tmp = (scaled_tmp - min(scaled_tmp.flatten())) / (
                    max(scaled_tmp.flatten()) - min(scaled_tmp.flatten()))
            else:
                scaled_tmp = np.zeros([self.outputH, self.outputW])
            heatmaps[i] = scaled_tmp

        tmp = utils.gaussian(np.array([self.inputH, self.inputW]),
                                  landmarks[:, -1], 4 * self.gauss)
        scaled_tmp = sp.misc.imresize(tmp, [self.outputH, self.outputW])
        scaled_tmp = (scaled_tmp - min(scaled_tmp.flatten())) / (
            max(scaled_tmp.flatten()) - min(scaled_tmp.flatten()))
        heatmaps[landmarks.shape[1]-1] = scaled_tmp

        return torch.from_numpy(heatmaps)

class ToColorHeatmap:
    """Converts a one-channel grayscale image Tensor ( H x W ) to a
    color heatmap image Tensor ( 3 x H x W )."""

    def __init__(self):
        self.toPILImage = torchvision.transforms.ToPILImage()
        self.toTensor = torchvision.transforms.ToTensor()

    @staticmethod
    def gauss(x, a, b, c):
        return torch.exp(-torch.pow(torch.add(x, -b), 2).div(2*c*c)).mul(a)

    def __call__(self, input, resolution=None):
        """
            Returns a Tensor which contains landmarks for every elements in the
            :attr:`heatmaps` tensor.

        Args:
            input (Tensor): input one-channel grayscale heatmap Tensor ( H x W )
            resolution ( H_new , W_new ): desired output size of colored heatmap
                                          Tensor ( 3 x H_new x W_new )

        Returns:
            Tensor: The color heatmap image Tensor ( 3 x H x W ).
        """
        colored_heatmap = torch.zeros(3, input.size(0), input.size(1))
        colored_heatmap[1] = self.gauss(input, 0.7, 1, 0.4)
        colored_heatmap[0] = self.gauss(input, 1, 1, 0.001) + self.gauss(input, 0.7, 1, 0.4)
        colored_heatmap[2] = self.gauss(input, 0.3, 1, 0.4)
        colored_heatmap[1][input.gt(0.99)] = 0
        colored_heatmap[2][input.gt(0.99)] = 0
        colored_heatmap[colored_heatmap.gt(1)] = 1
        if resolution:
            colored_heatmap = self.toTensor(
                self.toPILImage(colored_heatmap).resize(resolution))
        return colored_heatmap


    @staticmethod
    def to_numpy(tensor):
        if torch.is_tensor(tensor):
            return tensor.cpu().numpy()
        elif type(tensor).__module__ != 'numpy':
            raise ValueError("Cannot convert {} to numpy array"
                             .format(type(tensor)))
        return tensor

    @staticmethod
    def to_torch(ndarray):
        if type(ndarray).__module__ == 'numpy':
            return torch.from_numpy(ndarray)
        elif not torch.is_tensor(ndarray):
            raise ValueError("Cannot convert {} to torch tensor"
                             .format(type(ndarray)))
        return ndarray

    @staticmethod
    def gauss(x, a, b, c, d=0):
        return a * np.exp(-(x - b) ** 2 / (2 * c ** 2)) + d

    @staticmethod
    def color_heatmap(x):
        x = ToColorHeatmap.to_numpy(x)
        color = np.zeros((x.shape[0], x.shape[1], 3))
        color[:, :, 0] = ToColorHeatmap.gauss(x, .5, .6, .2) + ToColorHeatmap.gauss(x, 1, .8, .3)
        color[:, :, 1] = ToColorHeatmap.gauss(x, 1, .5, .3)
        color[:, :, 2] = ToColorHeatmap.gauss(x, 1, .2, .3)
        color[color > 1] = 1
        color = (color * 255).astype(np.uint8)
        return color

    @staticmethod
    def sample_with_heatmap(inp, out, num_rows=2, parts_to_show=None):
        inp = ToColorHeatmap.to_numpy(inp * 255)
        out = ToColorHeatmap.to_numpy(out)

        img = np.zeros((inp.shape[1], inp.shape[2], inp.shape[0]))
        for i in range(3):
            img[:, :, i] = inp[i, :, :]

        if parts_to_show is None:
            parts_to_show = np.arange(out.shape[0])

        # Generate a single image to display input/output pair
        num_cols = int(np.ceil(float(len(parts_to_show)) / num_rows))
        size = img.shape[0] // num_rows

        full_img = np.zeros((img.shape[0], size * (num_cols + num_rows), 3),
                            np.uint8)
        full_img[:img.shape[0], :img.shape[1]] = img

        inp_small = scipy.misc.imresize(img, [size, size])

        # Set up heatmap display for each part
        for i, part in enumerate(parts_to_show):
            part_idx = part
            out_resized = scipy.misc.imresize(out[part_idx], [size, size])
            out_resized = out_resized.astype(float) / 255
            out_img = inp_small.copy() * .3
            color_hm = ToColorHeatmap.color_heatmap(out_resized)
            out_img += color_hm * .7

            col_offset = (i % num_cols + num_rows) * size
            row_offset = (i // num_cols) * size
            full_img[row_offset:row_offset + size,
            col_offset:col_offset + size] = out_img

        return full_img

    @staticmethod
    def sample_with_heatmap_and_blockage(inputs, targets, predictions, num_rows=1, parts_to_show=None):
        inputs = ToColorHeatmap.to_numpy(inputs * 255)
        targets = ToColorHeatmap.to_numpy(targets)
        predictions = ToColorHeatmap.to_numpy(predictions)

        number_of_blocks = inputs.shape[0]
        for block_idx in range(number_of_blocks):
            inp = inputs[block_idx, ]
            out = predictions[block_idx, ]

            img = np.zeros((inp.shape[1], inp.shape[2], inp.shape[0]))
            for i in range(3):
                img[:, :, i] = inp[i, :, :]

            if parts_to_show is None:
                parts_to_show = np.arange(out.shape[0])

            # Generate a single image to display input/output pair
            num_cols = int(np.ceil(float(len(parts_to_show)) / num_rows))
            size = img.shape[0] // num_rows

            full_img = np.zeros((img.shape[0], size * (num_cols + num_rows), 3),
                                np.uint8)
            full_img[:img.shape[0], :img.shape[1]] = img

            inp_small = scipy.misc.imresize(img, [size, size])

            # Set up heatmap display for each part
            for i, part in enumerate(parts_to_show):
                part_idx = part
                out_resized = scipy.misc.imresize(out[part_idx], [size, size])
                out_resized = out_resized.astype(float) / 255
                out_img = inp_small.copy() * .3
                color_hm = ToColorHeatmap.color_heatmap(out_resized)
                out_img += color_hm * .7

                col_offset = (i % num_cols + num_rows) * size
                row_offset = (i // num_cols) * size
                full_img[row_offset:row_offset + size,
                col_offset:col_offset + size] = out_img

        return full_img


class ToLandmarks:
    r"""Generates landmarks for given heatmaps.

        Your heatmaps should be given as ( N x H x W ) where N is the number of
        (H x W) heatmaps. The landmarks will be a Tensor of size ( 3 x N ) where
        the first two elements are x and y position and last element is the
        confidence.
        """
        # REVIEW: is confidence a good term? or maybe probability, score?!!

    def __init__(self, resolution=None, threshold=0.1, gauss=1):
        """
            Args:
                resolution: landmarks will be provided in this resoultion ( H x W ). If None, the
                            heatmap resolution will be used.
                threshold: threshold for selecting a peak
                gauss: the width of gaussian
        """

    def __call__(self, heatmaps):
        """
            Returns a Tensor which contains landmarks for every elements in the
            :attr:`heatmaps` tensor.

        Args:
            heatmaps (Tensor): Tensor ( N x H x W ) contains N heatmaps of size
            ( H x W ).

        Returns:
            Tensor: The N landmarks ( 3 x N ) where where
            the first two elements are x and y position and last element is the
            confidence.
        """
        return self._get_preds(heatmaps)

    # retrieved from https://github.com/bearpaw/pytorch-pose/raw/master/pose/utils/evaluation.py
    @staticmethod
    def _get_preds(scores):
        ''' get predictions from score maps in torch Tensor
            return type: torch.LongTensor
        '''
        assert scores.dim() == 4, 'Score maps should be 4-dim'
        maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

        maxval = maxval.view(scores.size(0), scores.size(1), 1)
        idx = idx.view(scores.size(0), scores.size(1), 1) + 1

        preds = idx.repeat(1, 1, 2).float()

        preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
        preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

        pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
        preds *= pred_mask
        return preds
