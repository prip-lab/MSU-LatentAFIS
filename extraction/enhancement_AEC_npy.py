#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DCGAN.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import sys
import argparse
from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils.globvars import globalns as opt
import tensorflow as tf
import scipy.misc
from tensorpack import Trainer
from tensorpack import *
from tensorpack.utils.globvars import globalns as opt
from tensorpack.tfutils.common import get_tensors_by_names
import glob, os
import cv2
from tensorpack import (Trainer, QueueInput,
                        ModelDescBase, DataFlow, StagingInputWrapper,
                        MultiGPUTrainerBase,
                        TowerContext)
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.symbolic_functions import *

import matplotlib.pyplot as plt
from tensorpack.utils.argtools import memoized
from PIL import Image, ImageEnhance
import preprocessing
import math
from scipy.ndimage.interpolation import shift
sys.path.append('../Minutiae_UNet/')
"""
1. Download the 'aligned&cropped' version of CelebA dataset
   from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

2. Start training:
    ./DCGAN-CelebA.py --data /path/to/img_align_celeba/ --crop-size 140
    Generated samples will be available through tensorboard

3. Visualize samples with an existing model:
    ./DCGAN-CelebA.py --load path/to/model --sample

You can also train on other images (just use any directory of jpg files in
`--data`). But you may need to change the preprocessing.

A pretrained model on CelebA is at https://drive.google.com/open?id=0B9IPQTvr2BBkLUF2M0RXU1NYSkE
"""

# global vars
opt.SHAPE = 128
opt.BATCH = 128

class ImportGraph():
    def __init__(self, model_dir):
        # create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.weight = get_weights(opt.SHAPE, opt.SHAPE,1, sigma=None)
        with self.graph.as_default():
            meta_file, ckpt_file = get_model_filenames(os.path.expanduser(model_dir))
            model_dir_exp = os.path.expanduser(model_dir)
            saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
            saver.restore(self.sess, os.path.join(model_dir_exp, ckpt_file))

            self.images_placeholder = tf.get_default_graph().get_tensor_by_name('QueueInput/input_deque:0')
            output_name = 'reconstruction/gen:0'
            self.minutiae_cylinder_placeholder = tf.get_default_graph().get_tensor_by_name(output_name)
            self.shape = self.minutiae_cylinder_placeholder.get_shape()

    def run(self, img,minu_thr=0.2):
        h,w = img.shape
        nrof_samples = len(range(0, h, opt.SHAPE // 2)) * len(range(0, w, opt.SHAPE // 2))
        patches = np.zeros((nrof_samples, opt.SHAPE, opt.SHAPE, 1))
        n = 0
        x =[]
        y = []
        for i in range(0,h-opt.SHAPE+1,opt.SHAPE//2):
            for j in range(0, w-opt.SHAPE+1, opt.SHAPE // 2):
                patch = img[i:i+opt.SHAPE,j:j+opt.SHAPE,np.newaxis]
                x.append(j)
                y.append(i)
                patches[n,:,:,:] = patch
                n = n + 1
        feed_dict = {self.images_placeholder: patches}
        minutiae_cylinder_array = self.sess.run(self.minutiae_cylinder_placeholder, feed_dict=feed_dict)

        minutiae_cylinder = np.zeros((h, w, 1))
        for i in range(n):
            minutiae_cylinder[y[i]:y[i]+opt.SHAPE,x[i]:x[i]+opt.SHAPE,:] =minutiae_cylinder[y[i]:y[i]+opt.SHAPE,x[i]:x[i]+opt.SHAPE,:] + minutiae_cylinder_array[i]*self.weight

        minutiae_cylinder = minutiae_cylinder[:,:,0]
        minV = np.min(minutiae_cylinder)
        maxV = np.max(minutiae_cylinder)
        minutiae_cylinder = (minutiae_cylinder-minV)/(maxV-minV)*255

        return minutiae_cylinder

    def run_whole_image(self, img, minu_thr=0.2):
        
        img = np.asarray(img)
        h,w = img.shape
        img = np.expand_dims(img, axis=2)
        img = np.expand_dims(img, axis=0)
        feed_dict = {self.images_placeholder: img}
        minutiae_cylinder = self.sess.run(self.minutiae_cylinder_placeholder, feed_dict=feed_dict)

        minutiae_cylinder = np.squeeze(minutiae_cylinder,axis=0)
        minutiae_cylinder = np.squeeze(minutiae_cylinder, axis=2)
        minutiae_cylinder = minutiae_cylinder[:h,:w]
        minV = np.min(minutiae_cylinder)
        maxV = np.max(minutiae_cylinder)
        minutiae_cylinder = (minutiae_cylinder-minV)/(maxV-minV)*255


        return minutiae_cylinder
def PIL2array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)

def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)

def upsample(net, nf=32, upsample_factor=2):
    upsample_filter_np = upsampling.bilinear_upsample_weights(upsample_factor, nf)
    upsample_filter_tensor = tf.constant(upsample_filter_np)
    downsampled_logits_shape = net.get_shape()

    # Calculate the ouput size of the upsampled tensor
    upsampled_logits_shape = [
        downsampled_logits_shape[0],
        downsampled_logits_shape[1] * upsample_factor,
        downsampled_logits_shape[2] * upsample_factor,
        downsampled_logits_shape[3]
    ]
    print upsampled_logits_shape
    # Perform the upsampling
    net = tf.nn.conv2d_transpose(net, upsample_filter_tensor,
                                 output_shape=upsampled_logits_shape, strides=[1, upsample_factor, upsample_factor, 1])

    return net


class ImageFromFile_AutoEcoder(RNGDataFlow):
    """ Produce images read from a list of files. """
    def __init__(self, files, channel=3, resize=None, shuffle=False):
        """
        Args:
            files (list): list of file paths.
            channel (int): 1 or 3. Will convert grayscale to RGB images if channel==3.
            resize (tuple): int or (h, w) tuple. If given, resize the image.
        """
        assert len(files), "No image files given to ImageFromFile!"
        self.files = files
        self.channel = int(channel)
        self.imread_mode = cv2.IMREAD_GRAYSCALE if self.channel == 1 else cv2.IMREAD_COLOR
        if resize is not None:
            resize = shape2d(resize)
        self.resize = resize
        self.shuffle = shuffle

    def size(self):
        return len(self.files)

    def get_data(self):
        if self.shuffle:
            self.rng.shuffle(self.files)
        for f in self.files:
            matrix = np.load(f)

            matrix = np.float32(matrix)
            h,w,c = matrix.shape
            mx = np.random.randint(w-opt.SHAPE)
            my = np.random.randint(h-opt.SHAPE)
            im_label = matrix[my:my+opt.SHAPE,mx:mx+opt.SHAPE,1:2].copy()
            im_input = matrix[my:my+opt.SHAPE,mx:mx+opt.SHAPE,:1].copy()

            # random brightness
            delta = (np.random.rand(1) - 0.5) * 50
            im_input += delta

            # random contrastness
            scale = np.random.rand(1) + 0.5
            im_input *= scale

            im_input = im_input / 128.0 - 1
            mean = 0
            sigma = 1
            gauss = np.random.normal(mean, sigma, (opt.SHAPE, opt.SHAPE))
            im_input[:, :, 0] += gauss/2

            sigma_int = np.random.randint(0, 4)*2 + 1
            blur = cv2.GaussianBlur(im_input[:, :, 0], (sigma_int, sigma_int), 0)
            im_input[:, :, 0] = blur

            im_label = im_label / 128.0 - 1

            yield [im_input,im_label]


class Model(ModelDesc):

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, None, None,1), 'input'),
         InputDesc(tf.float32, (None, None, None, 1), 'label')]


    def collect_variables(self, scope='reconstruction'):
        """
        Assign self.g_vars to the parameters under scope `g_scope`,
        and same with self.d_vars.
        """
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        assert self.vars

    @auto_reuse_variable_scope
    def reconstruction_depth_5(self, imgs):
        nf = 16
        with argscope(Conv2D, nl=BNReLU, kernel_shape=4, stride=2), \
             argscope(LeakyReLU, alpha=0.2):
            l = (LinearWrap(imgs)
                 .Conv2D('conv0', nf)  # 64
                 .LeakyReLU()
                 .Conv2D('conv1', nf * 2)  # 32
                 .LeakyReLU()
                 .Conv2D('conv2', nf * 4)  # 16
                 .LeakyReLU()
                 .Conv2D('conv3', nf * 8)  # 8
                 .LeakyReLU()
                 .Conv2D('conv4', nf * 8)  # 4
                 .LeakyReLU()
                 .Conv2D('conv5', nf,kernel_shape=1, stride=1)  # 4
                 .LeakyReLU()())
            l = tf.tanh(l, name='feature')
        with argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
             l = Deconv2D('deconv3', l, nf * 8)
             l = Deconv2D('deconv4', l, nf * 4)
             l = Deconv2D('deconv5', l, nf * 2)
             l = Deconv2D('deconv6', l, nf * 1)
             l = Deconv2D('deconv7', l, nf)
             l = Conv2D('conv8', l, 1, kernel_shape=3, stride=1,nl=tf.identity)
             l = tf.tanh(l, name='gen')
        return l

    @auto_reuse_variable_scope
    def reconstruction_depth_4(self, imgs):
        nf = 16
        with argscope(Conv2D, nl=BNReLU, kernel_shape=4, stride=2), \
             argscope(LeakyReLU, alpha=0.2):
            l = (LinearWrap(imgs)
                 .Conv2D('conv0', nf)  # 64
                 .Conv2D('conv1', nf * 2)  # 32
                 .LeakyReLU()
                 .Conv2D('conv2', nf * 4)  # 16
                 .LeakyReLU()
                 .Conv2D('conv3', nf * 8)  # 8
                 .LeakyReLU()
                 .Conv2D('conv4', nf * 4, kernel_shape=3, stride=1)  # 8
                 .LeakyReLU()
                 .Conv2D('conv5', nf * 2, kernel_shape=3, stride=1)  # 8
                 .LeakyReLU()
                 .Conv2D('conv6', nf, kernel_shape=3, stride=1)  # 8
                 .LeakyReLU()())
        with argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
            l = Deconv2D('deconv4', l, nf * 4)
            l = Deconv2D('deconv5', l, nf * 2)
            l = Deconv2D('deconv6', l, nf * 1)
            l = Deconv2D('deconv7', l, nf)
            l = Conv2D('conv8', l, 1, kernel_shape=3, stride=1, nl=tf.identity)
            l = tf.tanh(l, name='gen')
        return l
    @auto_reuse_variable_scope
    def reconstruction_FCN(self, imgs):
        """ return a (b, 1) logits"""
 
        nf = 32
        net = imgs


        net = tf.layers.conv2d(net, nf, (3, 3), activation=tf.nn.relu, padding='same', name="conv_{}".format(1))
        net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(1))
        net = tf.layers.conv2d(net, nf*2, (3, 3), activation=tf.nn.relu, padding='same', name="conv_{}".format(2))
        net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(2))

        net = tf.layers.conv2d(net, nf*4, (3, 3), activation=tf.nn.relu, padding='same', name="conv_{}".format(3))
        net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(3))

        net = tf.layers.conv2d(net, nf*8, (3, 3), activation=tf.nn.relu, padding='same', name="conv_{}".format(4))
        net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(4))


        upsample_factor = 2
        
        net = upsample(net, nf*8, upsample_factor=2)
        net = tf.layers.conv2d(net, nf*4, (3, 3), activation=tf.nn.relu, padding='same', name="conv_{}".format(5))

        net = upsample(net,nf*4,upsample_factor = 2)
        net = tf.layers.conv2d(net, nf*2, (3, 3), activation=tf.nn.relu, padding='same', name="conv_{}".format(6))
     
        net = upsample(net,nf*2,upsample_factor = 2)
        net = tf.layers.conv2d(net, nf*1, (3, 3), activation=tf.nn.relu, padding='same', name="conv_{}".format(7))

        net = upsample(net,nf, upsample_factor = 2)
        net = tf.layers.conv2d(net, 12, (3, 3), activation=tf.identity, padding='same', name="conv_{}".format(8))
        net = tf.tanh(net, name='gen')
        
        return net


    def UNet(self, imgs):
        NF = 64
        with argscope(Conv2D, kernel_shape=4, stride=2), \
             argscope(LeakyReLU, alpha=0.2):
            # encoder
            e1 = Conv2D('conv1', imgs, NF, nl=LeakyReLU)
            e2 = Conv2D('conv2', e1, NF * 2)
            e3 = Conv2D('conv3', e2, NF * 4)
            e4 = Conv2D('conv4', e3, NF * 8)
            e5 = Conv2D('conv5', e4, NF * 8)
            e6 = Conv2D('conv6', e5, NF * 8)
            e7 = Conv2D('conv7', e6, NF * 8)
        with argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
            # decoder
            e7 = Deconv2D('deconv2', e7, NF * 8)
            e7 = Dropout(e7)
            e7 = ConcatWith(e7, e6,3)

            e6 = Deconv2D('deconv3', e7, NF * 8)
            e6 = Dropout(e6)
            e6 = ConcatWith(e6,  e5,3)

            e5 = Deconv2D('deconv4', e6, NF * 8)
            e5 = Dropout(e5)

            e4 = Deconv2D('deconv5', e5, NF * 4)
            e4 = Dropout(e4)

            e3 = Deconv2D('deconv6', e4, NF * 2)
            e3 = Dropout(e3)

            e2 = Deconv2D('deconv7', e3, NF * 1)
            e2 = Dropout(e2)

            e1 = Deconv2D('prediction', e2, 1, nl=tf.identity)
            prediction = tf.tanh(e1, name='gen')
        return prediction

    def _build_graph(self, inputs):
        image_pos = inputs[0]
        target = inputs[1] # / 128.0 - 1

        with argscope([Conv2D, Deconv2D],
                      W_init=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('reconstruction'):
                prediction = self.reconstruction_depth_4(image_pos)

        self.cost = tf.nn.l2_loss(prediction - target, name="L2loss")
        add_moving_summary(self.cost)
        tf.summary.image('original', image_pos, max_outputs=30)
        tf.summary.image('prediction', prediction, max_outputs=30)
        tf.summary.image('target', target, max_outputs=30)

        self.build_losses()
        self.collect_variables()

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 1e-4, summary=True)
        opt = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
        return opt

    def build_losses(self):
        """D and G play two-player minimax game with value function V(G,D)

          min_G max _D V(D, G) = IE_{x ~ p_data} [log D(x)] + IE_{z ~ p_fake} [log (1 - D(G(z)))]

        Args:
            logits_real (tf.Tensor): discrim logits from real samples
            logits_fake (tf.Tensor): discrim logits from fake samples produced by generator
        """
        with tf.name_scope("L2_loss"):
            self.loss = self.cost

    #@memoized
    def get_optimizer(self):
        return self._get_optimizer()



class AutoEncoderTrainer(Trainer):
    def __init__(self, config):
        """
        GANTrainer expects a ModelDesc in config which sets the following attribute
        after :meth:`_build_graph`: g_loss, d_loss, g_vars, d_vars.
        """
        input = QueueInput(config.dataflow)
        model = config.model

        cbs = input.setup(model.get_inputs_desc())
        config.callbacks.extend(cbs)

        with TowerContext('', is_training=True):
            model.build_graph(input)
        opt = model.get_optimizer()

        # by default, run one d_min after one g_min
        with tf.name_scope('optimize'):
            rec_min = opt.minimize(model.loss, var_list=model.vars, name='g_op')
        self.train_op = rec_min

        super(AutoEncoderTrainer, self).__init__(config)

def get_augmentors():
    augs = []
    augs.append(imgaug.Resize(opt.SHAPE))

    return augs

def get_data(datadir):
    imgs = glob.glob(datadir + '/*.jpeg')
    ds = ImageFromFile_AutoEcoder(imgs, channel=1, shuffle=True)
    ds = BatchData(ds, opt.BATCH)
    ds = PrefetchDataZMQ(ds, 1)
    return ds

def generate_training_npy(datadir, output_path):
    finger_paths = glob.glob(datadir + '/MI*')
    imgs = []
    for finger_path in finger_paths:
        img_files = glob.glob(finger_path + '/high*.bmp')
        for img_file in img_files:
            img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            img_STFT =preprocessing.STFT(img)

            matrix = np.concatenate((np.expand_dims(img_STFT, axis=2), np.expand_dims(img, axis=2)),2)
            output_file = output_path + img_file.split('/')[-1][:-3] + 'npy'
            matrix = np.uint8(matrix)
            np.save(output_file,matrix)
def get_data_low_and_high(datadir):
    imgs = glob.glob(datadir + '/*.npy')

    ds = ImageFromFile_AutoEcoder(imgs, channel=1, shuffle=True)
    ds = AugmentImageComponent(ds, get_augmentors())
    ds = BatchData(ds, opt.BATCH)
    ds = PrefetchDataZMQ(ds, 1)
    return ds

def Enhancement(model, model_path,sample_path, imgs, output_name='reconstruction/gen:0'):
    config = PredictConfig(
        session_init=get_model_loader(model_path),
        model=model,
        input_names=['sub:0'],#['z'],#
        output_names=[output_name, 'sub:0','reconstruction/feature:0'])
    imgs = glob.glob('/Data/Rolled/NSITSD14/Image2_jpeg' + '/*.jpeg')

    ds = ImageFromFile_AutoEcoder_Prediction(imgs, channel=1, shuffle=False)
    ds = PrefetchDataZMQ(ds, 1)
    ds = BatchData(ds, 1, remainder=True)
    pred = SimpleDatasetPredictor(config, ds)
    for o in pred.get_result():
        batch_size = o[0].shape[0]
        cv2.imwrite('test.jpeg', (o[0][0] + 1) * 128)
        cv2.imwrite('test2.jpeg', (o[1][0] + 1) * 128)


def get_weights(h,w,c,sigma=None):
    Y, X = np.mgrid[0:h, 0:w]
    x0 = w//2
    y0 = h//2
    if sigma is None:
        sigma = (np.max([h,w])*1./3)**2
    weight = np.exp(-((X - x0) * (X - x0) + (Y - y0) * (Y - y0)) / sigma)
    weight = np.stack((weight,) * c,axis=2)
    return weight

def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), ckpt_file)


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt_file = tf.train.latest_checkpoint(model_dir)
    return meta_file, ckpt_file


def enhancement2(model_path,sample_path, imgs, output_name='reconstruction/gen:0'):
    imgs = glob.glob('/Data/Rolled/NISTSD4/Image_Aligned'+'/*.jpeg')
    imgs = glob.glob('/Data/Latent/NISTSD27/image/*.bmp')
    imgs.sort()
    sample_path = '/AutomatedLatentRecognition/enhanced_latents_3/'
    weight = get_weights(opt.SHAPE, opt.SHAPE, 1)
    with tf.Graph().as_default():

        with TowerContext('', is_training=False):
            with tf.Session() as sess:
                is_training= get_current_tower_context().is_training
                load_model(model_path)
                images_placeholder = tf.get_default_graph().get_tensor_by_name('sub:0')
                #is_training
                minutiae_cylinder_placeholder = tf.get_default_graph().get_tensor_by_name(output_name)
                for k, file in enumerate(imgs):
                    img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
                    u, texture = LP.FastCartoonTexture(img)
                    img = texture / 128.0 - 1

                    #img = img/128.0-1
                    h,w = img.shape
                    x = []
                    y = []
                    nrof_samples = len(range(0,h,opt.SHAPE//2)) * len(range(0,w,opt.SHAPE//2))
                    patches = np.zeros((nrof_samples, opt.SHAPE, opt.SHAPE, 1))
                    n = 0
                    for i in range(0,h-opt.SHAPE+1,opt.SHAPE//2):

                        for j in range(0, w-opt.SHAPE+1, opt.SHAPE // 2):
                            print j
                            patch = img[i:i+opt.SHAPE,j:j+opt.SHAPE,np.newaxis]
                            x.append(j)
                            y.append(i)
                            patches[n,:,:,:] = patch
                            n = n + 1
                    feed_dict = {images_placeholder: patches}
                    minutiae_cylinder_array = sess.run(minutiae_cylinder_placeholder, feed_dict=feed_dict)

                    minutiae_cylinder = np.zeros((h, w, 1))
                    for i in range(n):
                        minutiae_cylinder[y[i]:y[i]+opt.SHAPE,x[i]:x[i]+opt.SHAPE,:] =minutiae_cylinder[y[i]:y[i]+opt.SHAPE,x[i]:x[i]+opt.SHAPE,:] + minutiae_cylinder_array[i]*weight

                    minV = np.min(minutiae_cylinder)
                    maxV = np.max(minutiae_cylinder)
                    minutiae_cylinder = (minutiae_cylinder-minV)/(maxV-minV)*255
                    cv2.imwrite( (sample_path+'test_%03d.jpeg'%(k+1)), minutiae_cylinder)
                    print n

def enhancement_whole_image(model_path,sample_path, imgs, output_name='reconstruction/gen:0'):
    imgs = glob.glob('/Data/Rolled/NISTSD4/Image_Aligned'+'/*.jpeg')
    imgs = glob.glob('/Data/Latent/NISTSD27/image/*.bmp')
    imgs.sort()
    sample_path = '/AutomatedLatentRecognition/enhanced_latents_2/'
    weight = get_weights(opt.SHAPE, opt.SHAPE, 1)
    with tf.Graph().as_default():

        with TowerContext('', is_training=False):
            with tf.Session() as sess:
                is_training= get_current_tower_context().is_training
                load_model(model_path)
                images_placeholder = tf.get_default_graph().get_tensor_by_name('sub:0')
                #is_training
                minutiae_cylinder_placeholder = tf.get_default_graph().get_tensor_by_name(output_name)
                for k, file in enumerate(imgs):
                    img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
                    img = Image.fromarray(img)
                    img = ImageEnhance.Contrast(img)
                    img = np.asarray(img)
                    img = img/128.0-1
                    h,w = img.shape
                    img = np.expand_dims(img, axis=2)
                    img = np.expand_dims(img, axis=0)
                    feed_dict = {images_placeholder: img}
                    minutiae_cylinder = sess.run(minutiae_cylinder_placeholder, feed_dict=feed_dict)

                    minutiae_cylinder = np.squeeze(minutiae_cylinder,axis=0)
                    minutiae_cylinder = np.squeeze(minutiae_cylinder, axis=2)
                    minV = np.min(minutiae_cylinder)
                    maxV = np.max(minutiae_cylinder)
                    minutiae_cylinder = (minutiae_cylinder-minV)/(maxV-minV)*255
                    cv2.imwrite( (sample_path+'test_%03d.jpeg'%(k+1)), minutiae_cylinder)
                    print h,w

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.',default = '1')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--enhance', action='store_true', help='enhance examples')
    parser.add_argument('--test_data', help='a jpeg directory',default='/Data/Rolled/selected_rolled_prints/MI0479144T_07/')
    parser.add_argument('--sample_dir', help='directory for generated examples', type=str,
                        default='/AutomatedLatentRecognition/Enhancement_test')

    parser.add_argument('--data', help='a jpeg directory',default='/AutomatedLatentRecognition/Data/enhancement_training/')
    parser.add_argument('--load-size', help='size to load the original images', type=int)
    parser.add_argument('--batch_size', help='batch size', type=int)
    parser.add_argument('--crop-size', help='crop the original images', type=int)
    parser.add_argument('--log_dir', help='directory to save checkout point', type=str,
                        default='/AutomatedLatentRecognition/models/Enhancement/AEC_net/Enhancement_AEC_128_depth_4_STFT_2/')
    args = parser.parse_args()

    opt.use_argument(args)
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.batch_size:
        opt.BATCH = args.batch_size
    return args

def get_config(log_dir,datadir):
    logger.set_logger_dir(log_dir)
    return TrainConfig(
        dataflow=dataset,
        callbacks=[ModelSaver(keep_recent=True)],
        model=Model(),
        steps_per_epoch=1000, #dataset.size()
        max_epoch=3000,
        session_init=SaverRestore(args.load) if args.load else None
    )

def test_noise():

    training_path = '/AutomatedLatentRecognition/Data/enhancement_training/'
    imgs = glob.glob(training_path + '/*.npy')
    f = imgs[1]
    matrix = np.load(f)
    matrix = np.float32(matrix)
    h, w, c = matrix.shape
    mx = np.random.randint(w - opt.SHAPE)
    my = np.random.randint(h - opt.SHAPE)
    im_label = matrix[my:my + opt.SHAPE, mx:mx + opt.SHAPE, 1:2].copy()
    im_input = matrix[my:my + opt.SHAPE, mx:mx + opt.SHAPE, :1].copy()

    # random brightness
    delta = (np.random.rand(1) - 0.5) * 50
    im_input += delta

    # random contrastness
    scale = np.random.rand(1) + 0.5
    im_input *= scale

    im_input = im_input / 128.0 - 1
    mean = 0
    sigma = 1
    gauss = np.random.normal(mean, sigma, (opt.SHAPE, opt.SHAPE))
    im_input[:, :, 0] += gauss / 2

    sigma = np.random.rand(1) * 5 + 5
    theta = (np.random.rand(1) - 0.5) * math.pi * 2
    scale = np.random.rand(1) + 0.5
    lambd = np.random.rand(1) * 10
    gamma = np.random.rand(1)*0.5
    noise = scale * cv2.getGaborKernel((opt.SHAPE, opt.SHAPE), sigma, theta, lambd, gamma)
    noise = shift(noise, np.random.randint(64),cval=0.)
    im_input[:, :, 0] += noise[:opt.SHAPE, :opt.SHAPE]  # , sigma, theta,lambd,gamma)
    plt.imshow(noise,cmap='gray')
    plt.show(block=True)

if __name__ == '__main__':
    test_noise()
    args = get_args()
    print(args)
    if args.enhance and args.load:
        imgs = ['/Data/Rolled/selected_rolled_prints/MI0479144T_07/low_02_A103585608W_07.bmp']
        enhancement2(args.load, args.sample_dir, imgs)
    else:
        config = get_config(args.log_dir, args.data)
        AutoEncoderTrainer(config).train()


