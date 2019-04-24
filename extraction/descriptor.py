import template
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import load


class ImportGraph():
    # input:
    #       model_dir: path for descriptor model
    # 	    input_name: input name of the tensor
    #       output_name: output name of the tensor
    def __init__(self, model_dir, input_name="batch_join:0", output_name="Add:0"):
        # create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # load minutiae model
            meta_file, ckpt_file = load.get_model_filenames(os.path.expanduser(model_dir))
            model_dir_exp = os.path.expanduser(model_dir)
            saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
            saver.restore(self.sess, os.path.join(model_dir_exp, ckpt_file))

            self.images_placeholder = tf.get_default_graph().get_tensor_by_name(input_name)
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            self.embeddings = tf.get_default_graph().get_tensor_by_name(output_name)
            self.embedding_size = self.embeddings.get_shape()[1]

    def run(self, imgs):
        feed_dict = {self.images_placeholder: imgs, self.phase_train_placeholder: False}
        return self.sess.run(self.embeddings, feed_dict=feed_dict)


def get_patch_location(patch_type=1):
    if patch_type == 1:
        x = np.array(xrange(40, 120))
        y = list(xrange(40, 120))
    elif patch_type == 2:
        x = np.array(xrange(32, 128))
        y = np.array(xrange(32, 128))
    elif patch_type == 3:
        x = np.array(xrange(24, 136))
        y = np.array(xrange(24, 136))
    elif patch_type == 4:
        x = np.array(xrange(16, 144))
        y = np.array(xrange(16, 144))
    elif patch_type == 5:
        x = np.array(xrange(8, 152))
        y = np.array(xrange(8, 152))
    elif patch_type == 6:
        x = np.array(xrange(0, 160))
        y = np.array(xrange(0, 160))
    elif patch_type == 7:
        x = np.array(xrange(0, 96))
        y = np.array(xrange(0, 96))
    elif patch_type == 8:
        x = np.array(xrange(32, 128))
        y = np.array(xrange(0, 96))
    elif patch_type == 9:
        x = np.array(xrange(64, 160))
        y = np.array(xrange(0, 96))
    elif patch_type == 10:
        x = np.array(xrange(64, 160))
        y = np.array(xrange(32, 128))
    elif patch_type == 11:
        x = np.array(xrange(64, 160))
        y = np.array(xrange(64, 160))
    elif patch_type == 12:
        x = np.array(xrange(32, 128))
        y = np.array(xrange(64, 160))
    elif patch_type == 13:
        x = np.array(xrange(1, 96))
        y = np.array(xrange(64, 160))
    elif patch_type == 14:
        x = np.array(xrange(1, 96))
        y = np.array(xrange(32, 128))

    xv, yv = np.meshgrid(x, y)
    return yv, xv


def get_patch_index(patchSize_L, patchSize_H, oriNum, isMinu=1):
    if isMinu == 1:
        PI2 = 2 * math.pi
    else:
        PI2 = math.pi
    x = list(xrange(-patchSize_L / 2 + 1, patchSize_L / 2 + 1))
    x = np.array(x)
    y = list(xrange(-patchSize_H / 2 + 1, patchSize_H / 2 + 1))
    y = np.array(y)
    xv, yv = np.meshgrid(x, y)
    patchIndexV = {}
    patchIndexV['x'] = []
    patchIndexV['y'] = []
    for i in range(oriNum):

        th = i * PI2 / oriNum
        u = xv * np.cos(th) - yv * np.sin(th)
        v = xv * np.sin(th) + yv * np.cos(th)
        u = np.around(u)
        v = np.around(v)
        patchIndexV['x'].append(u)
        patchIndexV['y'].append(v)
    return patchIndexV


# extract fingerprint patches aligned by minutiae location and angles
# input:
#        img: input fingerprint image
#        minutiae:  input minutiae points, including x, y locations and orientations
#        patch_type: there are totally 14 patches types, for more details, please refer Kai Cao's 2018 PAMI paper
#        patch_size: patch size to feed into neural network
# output:
#        patches:  fingerprint patches aligned by minutiae. The number of patches is the same as the number of minutiae
def extract_patches(minutiae, img, patchIndexV, patch_type=1, patch_size=96):

    assert(minutiae.shape[1] > 0)
    channels = 1
    if len(img.shape) == 2:
        h, w = img.shape
        ret = np.empty((h, w, channels), dtype=np.float)
        ret[:, :, :] = img[:, :, np.newaxis]
        img = ret

    num_minu = minutiae.shape[0]
    oriNum = len(patchIndexV['x'])

    ly, lx = get_patch_location(patch_type=patch_type)
    h, w, c = img.shape
    patches = np.zeros((num_minu, patch_size, patch_size, c), dtype=np.float32)
    for i in xrange(num_minu):
        x = minutiae[i, 0]
        y = minutiae[i, 1]
        ori = -minutiae[i, 2]
        ori = ori % (math.pi * 2)
        if ori < 0:
            ori += math.pi * 2
        oriInd = round(ori / (math.pi * 2) * oriNum)
        if oriInd >= oriNum:
            oriInd -= oriNum
        oriInd = np.int(oriInd)
        xv = patchIndexV['x'][oriInd] + x
        yv = patchIndexV['y'][oriInd] + y
        xv[xv < 0] = 0
        xv[xv >= w] = w - 1
        yv[yv < 0] = 0
        yv[yv >= h] = h - 1
        xv = xv.astype(int)
        yv = yv.astype(int)
        patch = img[yv, xv, :]
        patch = patch[ly, lx, :]
        if patch.shape[0] < patch_size:
            patch = cv2.resize(patch, (patch_size, patch_size))

        patches[i, :, :, :] = patch

    return patches


def minutiae_descriptor_extraction(img, minutiae, patch_types, models, patchIndexV, batch_size=128, patch_size=96):
    des = []
    if len(minutiae) == 0:
        return des
    for k, patch_type in enumerate(patch_types):
        embedding_size = models[k].embedding_size
        patches = extract_patches(minutiae, img, patchIndexV, patch_type=patch_type, patch_size=patch_size)

        nrof_patches = len(patches)
        emb_array = np.zeros((nrof_patches, embedding_size))
        nrof_batches = int(math.ceil(1.0 * nrof_patches / batch_size))
        for i in range(nrof_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, nrof_patches)
            patches_batch = patches[start_index:end_index, :, :]
            emb_array[start_index:end_index, :] = models[k].run(patches_batch)
        for i in range(nrof_patches):
            norm = np.linalg.norm(emb_array[i, :]) + 0.0000001
            emb_array[i, :] = emb_array[i, :] / norm
        des.append(emb_array)
    return des


if __name__ == '__main__':
    # a demo for mnutiae desriptor extraction. But the template formart may be different from the latest one now.
    patchSize = 160
    oriNum = 64
    patchIndexV = get_patch_index(patchSize, patchSize, oriNum, isMinu=1)

    fname = 'Data/Latent/001.dat'

    latent_template = template.Bin2Template_Byte(fname, isLatent=1)
    imgfile = '/latent/001.bmp'

    img = cv2.imread(imgfile)
    h, w, c = img.shape

    print img

    num_minu = len(template.minu_template[0].minutiae)
    patches = extract_patches(template.minu_template[0].minutiae, img, patchIndexV, patch_type=1)
    for i in range(len(patches)):
        patch = patches[i, :, :, 0]
        plt.imshow(patch, cmap='gray')
        plt.show()

    num_minu = len(template.texture_template[0].minutiae)
    patches = extract_patches(template.texture_template[0].minutiae, img, patchIndexV, patch_type=1)
    for i in range(len(patches)):
        patch = patches[i, :, :, 0]
        plt.imshow(patch, cmap='gray')
        plt.show()
