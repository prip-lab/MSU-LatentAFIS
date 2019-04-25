import struct
import numpy as np
from numpy import linalg as LA
from io import BytesIO
from cStringIO import StringIO


class MinuTemplate():
    def __init__(self, h=0, w=0, block_size=16, blkH=0, blkW=0, minutiae=None, des=None, oimg=None, mask=None):
        self.h = h
        self.w = w
        self.blkH = blkH
        self.blkW = blkW
        self.minutiae = minutiae
        self.des = des

        self.mask = mask
        self.oimg = oimg


class TextureTemplate():
    def __init__(self, h=0, w=0, minutiae=None, des=None, mask=None):
        self.h = h
        self.w = w
        self.minutiae = minutiae
        self.des = des
        self.mask = mask


class Template():
    def __init__(self, minu_template=None, texture_template=None):
        self.minu_template = [] if minu_template is None else minu_template
        self.texture_template = [] if texture_template is None else texture_template

    def add_minu_template(self, minu_template):
        self.minu_template.append(minu_template)

    def nrof_minu_template(self):
        self.len(self.minu_template)

    def add_texture_template(self, texture_template):
        self.texture_template.append(texture_template)


def run_length_decoding(run_mask, h, w):
    mask = np.zeros((h * w,), dtype=int)
    run_mask = np.cumsum(run_mask)
    for i in range(1, len(run_mask), 2):
        mask[run_mask[i - 1]:run_mask[i]] = 1
    mask = np.reshape(mask, (w, h))
    mask = mask.transpose()
    return mask


def run_length_encoding(mask):
    mask = mask.transpose()
    h, w = mask.shape
    mask = np.reshape(mask, (h * w,))
    mask[0] = 0
    diff = [0]
    num = h * w
    for i in range(1, num):
        if mask[i] != mask[i - 1]:
            diff.append(i)
    diff.append(num)

    run_mask = []
    for i in range(1, len(diff)):
        run_mask.append(diff[i] - diff[i - 1])
    return run_mask


def Bin2Template_Byte(fname, isLatent=True):
    template = Template()
    with open(fname, 'rb') as file:
        tmp = struct.unpack('H' * 2, file.read(4))
        h = tmp[0]
        w = tmp[1]
        if w <= 0 or h <= 0:
            return None
        tmp = struct.unpack('H' * 2, file.read(4))
        blkH = tmp[0]
        blkW = tmp[1]
        tmp = struct.unpack('B', file.read(1))
        # number of minutiae templates
        num_template = tmp[0]
        for i in range(num_template):
            tmp = struct.unpack('H', file.read(2))
            minu_num = tmp[0]
            if minu_num <= 0:
                minutiae = None
                des = None
                continue
            minutiae = np.zeros((minu_num, 3), dtype=float)

            # x location
            tmp = struct.unpack('H' * minu_num, file.read(2 * minu_num))
            minutiae[:, 0] = np.array(list(tmp))
            # y location
            tmp = struct.unpack('H' * minu_num, file.read(2 * minu_num))
            minutiae[:, 1] = np.array(list(tmp))

            # minutiae orientation
            tmp = struct.unpack('f' * minu_num, file.read(4 * minu_num))
            minutiae[:, 2] = np.array(list(tmp))

            tmp = struct.unpack('H', file.read(2))
            des_num = tmp[0]
            tmp = struct.unpack('H', file.read(2))
            des_len = tmp[0]
            des = []
            for j in range(des_num):
                tmp = struct.unpack('H' * des_len * minu_num, file.read(2 * des_len * minu_num))
                single_des = np.array(list(tmp))
                single_des = np.reshape(single_des, (minu_num, des_len))
                single_des = np.float32(single_des)
                for k in range(minu_num):
                    single_des[k] = single_des[k] * 1.0 / (LA.norm(single_des[k]) + 0.000001)
                des.append(single_des)

            # get the orientation field
            tmp = struct.unpack('f' * blkH * blkW, file.read(4 * blkH * blkW))
            oimg = np.array(list(tmp))
            oimg = np.reshape(oimg, (blkW, blkH))
            oimg = oimg.transpose()

            tmp = struct.unpack('H', file.read(2))
            run_mask_num = tmp[0]
            tmp = struct.unpack('I' * run_mask_num, file.read(4 * run_mask_num))
            tmp_list = list(tmp)
            mask = run_length_decoding(tmp_list, h, w)
            minu_template = MinuTemplate(h=h, w=w, blkH=blkH, blkW=blkW,
                                         minutiae=minutiae, des=des, oimg=oimg, mask=mask)
            template.add_minu_template(minu_template)

        # load texture template
        tmp = struct.unpack('H', file.read(2))
        minu_num = tmp[0]
        if isLatent == 1:
            minutiae = np.zeros((minu_num, 4), dtype=float)
        else:
            minutiae = np.zeros((minu_num, 3), dtype=float)

        if minu_num <= 0:
            minutiae = None
            des = None
            texture_template = TextureTemplate(h=h, w=w, minutiae=minutiae, des=des, mask=mask)
            template.add_texture_template(texture_template)
            return

        # x location
        tmp = struct.unpack('H' * minu_num, file.read(2 * minu_num))
        minutiae[:, 0] = np.array(list(tmp))
        # y location
        tmp = struct.unpack('H' * minu_num, file.read(2 * minu_num))
        minutiae[:, 1] = np.array(list(tmp))

        # minutiae orientation
        tmp = struct.unpack('f' * minu_num, file.read(4 * minu_num))
        minutiae[:, 2] = np.array(list(tmp))

        # distance to the border
        if isLatent:
            tmp = struct.unpack('f' * minu_num, file.read(4 * minu_num))
            minutiae[:, 3] = np.array(list(tmp))

        tmp = struct.unpack('H', file.read(2))
        des_num = tmp[0]
        tmp = struct.unpack('H', file.read(2))
        des_len = tmp[0]
        des = []
        for j in range(des_num):
            tmp = struct.unpack('H' * des_len * minu_num, file.read(2 * des_len * minu_num))
            single_des = np.array(list(tmp))
            single_des = np.reshape(single_des, (minu_num, des_len))
            # Each row of single_des is a descriptor for a minutia
            single_des = np.float32(single_des)
            for k in range(minu_num):
                single_des[k] = single_des[k] * 1.0 / (LA.norm(single_des[k]) + 0.000001)
            des.append(single_des)
        texture_template = TextureTemplate(h=h, w=w, minutiae=minutiae, des=des, mask=mask)
        template.add_texture_template(texture_template)
    return template


def Template2Bin_Byte(fname, template, isLatent=True):
    MAXV = 65535
    with open(fname, 'wb') as file:
        # save first two minutiae templates
        tmp = (template.minu_template[0].h, template.minu_template[0].w,
               template.minu_template[0].blkH, template.minu_template[0].blkW)
        file.write(struct.pack('H' * 4, *tmp))
        num_minu_template = len(template.minu_template)
        tmp = (num_minu_template,)
        file.write(struct.pack('B', *tmp))
        for i in range(num_minu_template):
            minu_num = len(template.minu_template[i].minutiae)
            tmp = (minu_num,)
            file.write(struct.pack('H', *tmp))
            if minu_num <= 0:
                continue
            x = template.minu_template[i].minutiae[:, 0]
            x = tuple(np.uint16(x))
            file.write(struct.pack('H' * minu_num, *x))
            y = template.minu_template[i].minutiae[:, 1]
            y = tuple(np.uint16(y))
            file.write(struct.pack('H' * minu_num, *y))
            # orientation
            ori = template.minu_template[i].minutiae[:, 2]
            ori = tuple(np.float32(ori))
            file.write(struct.pack('f' * minu_num, *ori))

            des_num = len(template.minu_template[i].des)
            des_len = template.minu_template[i].des[0].shape[1]
            tmp = (des_num, des_len)
            file.write(struct.pack('H' * 2, *tmp))
            for j in range(des_num):
                descriptor = template.minu_template[i].des[j]
                for k in range(minu_num):
                    maxV = np.max(descriptor[k])
                    descriptor[k, :] = descriptor[k, :] / (maxV + 0.00001) * MAXV
                descriptor = np.floor(descriptor)
                descriptor = np.reshape(descriptor, (des_len * minu_num,))
                descriptor_tuple = tuple(np.uint16(descriptor))
                file.write(struct.pack('H' * des_len * minu_num, *(descriptor_tuple)))
            # orientation field
            oimg = template.minu_template[i].oimg
            oimg = oimg.transpose()
            blkH = template.minu_template[i].blkH
            blkW = template.minu_template[i].blkW
            oimg = tuple(np.reshape(oimg, (blkH * blkW,)))
            file.write(struct.pack('f' * blkH * blkW, *(oimg)))

            run_mask = run_length_encoding(template.minu_template[i].mask)
            tmp = (len(run_mask),)
            file.write(struct.pack('H', *(tmp)))
            run_mask = np.uint32(run_mask)
            run_mask = tuple(run_mask)
            file.write(struct.pack('I' * len(run_mask), *(run_mask)))

        # save first one texture template
        num_texture_template = len(template.texture_template)
        tmp = (num_texture_template,)
        file.write(struct.pack('H', *tmp))
        if num_texture_template == 0:
            return
        minu_num = len(template.texture_template[0].minutiae)
        tmp = (minu_num,)
        file.write(struct.pack('H', *tmp))
        if minu_num > 0:
            x = template.texture_template[0].minutiae[:, 0]
            x = tuple(np.uint16(x))
            file.write(struct.pack('H' * minu_num, *x))
            y = template.texture_template[0].minutiae[:, 1]
            y = tuple(np.uint16(y))
            file.write(struct.pack('H' * minu_num, *y))
            # orientation
            ori = template.texture_template[0].minutiae[:, 2]
            ori = tuple(np.float32(ori))
            file.write(struct.pack('f' * minu_num, *ori))
            if isLatent:
                D = template.texture_template[0].minutiae[:, 3]
                D = tuple(np.float32(D))
                file.write(struct.pack('f' * minu_num, *D))

            des_num = len(template.texture_template[0].des)
            des_len = template.texture_template[0].des[0].shape[1]
            tmp = (des_num, des_len)
            file.write(struct.pack('H' * 2, *tmp))
            for j in range(des_num):
                descriptor = template.texture_template[0].des[j]
                for k in range(minu_num):
                    maxV = np.max(descriptor[k])
                    descriptor[k, :] = descriptor[k, :] / (maxV + 0.00001) * MAXV
                descriptor = np.floor(descriptor)
                descriptor = np.reshape(descriptor, (des_len * minu_num,))
                descriptor_tuple = tuple(np.uint16(descriptor))
                file.write(struct.pack('H' * des_len * minu_num, *(descriptor_tuple)))


def Bin2Template_Byte_TF(fname, isLatent=True):
    template = Template()
    with open(fname, 'rb') as file:
        all = file.read()
        string = StringIO(all)
        tmp = struct.unpack('H' * 2, string.read(4))
        h = tmp[0]
        w = tmp[1]
        if w <= 0 or h <= 0:
            return None
        tmp = struct.unpack('H' * 2, string.read(4))
        blkH = tmp[0]
        blkW = tmp[1]
        tmp = struct.unpack('B', string.read(1))
        # number of minutiae templates
        num_template = tmp[0]
        for i in range(num_template):
            tmp = struct.unpack('H', string.read(2))
            minu_num = tmp[0]
            if minu_num <= 0:
                minutiae = None
                des = None
                continue
            minutiae = np.zeros((minu_num, 4), dtype=float)

            # x location
            tmp = struct.unpack('H' * minu_num, string.read(2 * minu_num))
            minutiae[:, 0] = np.array(list(tmp))
            # y location
            tmp = struct.unpack('H' * minu_num, string.read(2 * minu_num))
            minutiae[:, 1] = np.array(list(tmp))

            # minutiae orientation
            tmp = struct.unpack('f' * minu_num, string.read(4 * minu_num))
            minutiae[:, 2] = np.array(list(tmp))

            # minutiae reliablility
            tmp = struct.unpack('f' * minu_num, string.read(4 * minu_num))
            minutiae[:, 3] = np.array(list(tmp))

            tmp = struct.unpack('H', string.read(2))
            des_num = tmp[0]
            tmp = struct.unpack('H', string.read(2))
            des_len = tmp[0]
            des = []
            for j in range(des_num):
                tmp = struct.unpack('f' * des_len * minu_num, string.read(4 * des_len * minu_num))
                single_des = np.array(list(tmp))
                single_des = np.reshape(single_des, (minu_num, des_len))
                single_des = np.float32(single_des)
                for k in range(minu_num):
                    single_des[k] = single_des[k] * 1.0 / (LA.norm(single_des[k]) + 0.000001)
                des.append(single_des)

            # get the orientation field
            tmp = struct.unpack('f' * blkH * blkW, string.read(4 * blkH * blkW))
            oimg = np.array(list(tmp))
            oimg = np.reshape(oimg, (blkW, blkH))
            oimg = oimg.transpose()

            tmp = struct.unpack('H', string.read(2))
            run_mask_num = tmp[0]
            tmp = struct.unpack('I' * run_mask_num, string.read(4 * run_mask_num))
            tmp_list = list(tmp)
            mask = run_length_decoding(tmp_list, h, w)
            minu_template = MinuTemplate(h=h, w=w, blkH=blkH, blkW=blkW,
                                         minutiae=minutiae, des=des, oimg=oimg, mask=mask)
            template.add_minu_template(minu_template)
        # load texture template

        tmp = struct.unpack('H', string.read(2))
        num_texture_template = tmp[0]
        if num_texture_template == 0:
            return template
        tmp = struct.unpack('H', string.read(2))
        minu_num = tmp[0]
        if isLatent == 1:
            minutiae = np.zeros((minu_num, 4), dtype=float)
        else:
            minutiae = np.zeros((minu_num, 3), dtype=float)

        if minu_num <= 0:
            minutiae = None
            des = None
            texture_template = TextureTemplate(h=h, w=w, minutiae=minutiae, des=des, mask=mask)
            template.add_texture_template(texture_template)
            return

        # x location
        tmp = struct.unpack('H' * minu_num, string.read(2 * minu_num))
        minutiae[:, 0] = np.array(list(tmp))
        # y location
        tmp = struct.unpack('H' * minu_num, string.read(2 * minu_num))
        minutiae[:, 1] = np.array(list(tmp))

        # minutiae orientation
        tmp = struct.unpack('f' * minu_num, string.read(4 * minu_num))
        minutiae[:, 2] = np.array(list(tmp))

        # distance to the border
        if isLatent:
            tmp = struct.unpack('f' * minu_num, string.read(4 * minu_num))
            minutiae[:, 3] = np.array(list(tmp))

        tmp = struct.unpack('H', string.read(2))
        des_num = tmp[0]
        tmp = struct.unpack('H', string.read(2))
        des_len = tmp[0]
        des = []
        for j in range(des_num):
            tmp = struct.unpack('H' * des_len * minu_num, string.read(2 * des_len * minu_num))
            single_des = np.array(list(tmp))
            single_des = np.reshape(single_des, (minu_num, des_len))
            # Each row of single_des is a descriptor for a minutia
            single_des = np.float32(single_des)
            for k in range(minu_num):
                single_des[k] = single_des[k] * 1.0 / (LA.norm(single_des[k]) + 0.000001)
            des.append(single_des)
        texture_template = TextureTemplate(h=h, w=w, minutiae=minutiae, des=des, mask=mask)
        template.add_texture_template(texture_template)
    return template


def Template2Bin_Byte_TF(fname, template, isLatent=True, save_mask=False):
    with open(fname, 'wb') as file:
        # save first two minutiae templates
        tmp = (template.minu_template[0].h, template.minu_template[0].w,
               template.minu_template[0].blkH, template.minu_template[0].blkW)
        file.write(struct.pack('H' * 4, *tmp))
        num_minu_template = len(template.minu_template)
        tmp = (num_minu_template,)
        file.write(struct.pack('B', *tmp))
        for i in range(num_minu_template):
            minu_num = len(template.minu_template[i].minutiae)
            tmp = (minu_num,)
            file.write(struct.pack('H', *tmp))
            if minu_num <= 0:
                continue
            x = template.minu_template[i].minutiae[:, 0]
            x = tuple(np.uint16(x))
            file.write(struct.pack('H' * minu_num, *x))
            y = template.minu_template[i].minutiae[:, 1]
            y = tuple(np.uint16(y))
            file.write(struct.pack('H' * minu_num, *y))
            # orientation
            ori = template.minu_template[i].minutiae[:, 2]
            ori = tuple(np.float32(ori))
            file.write(struct.pack('f' * minu_num, *ori))

            reliability = template.minu_template[i].minutiae[:, 3]
            reliability = tuple(np.float32(reliability))
            file.write(struct.pack('f' * minu_num, *reliability))

            des_num = len(template.minu_template[i].des)
            des_len = template.minu_template[i].des[0].shape[1]
            tmp = (des_num, des_len)
            file.write(struct.pack('H' * 2, *tmp))
            for j in range(des_num):
                descriptor = template.minu_template[i].des[j]
                descriptor = np.reshape(descriptor, (des_len * minu_num,))
                descriptor.astype(np.float32)
                descriptor_tuple = tuple(descriptor)
                file.write(struct.pack('f' * des_len * minu_num, *(descriptor_tuple)))
            # orientation field
            oimg = template.minu_template[i].oimg
            oimg = oimg.transpose()
            blkH = template.minu_template[i].blkH
            blkW = template.minu_template[i].blkW
            oimg = tuple(np.reshape(oimg, (blkH * blkW,)))
            file.write(struct.pack('f' * blkH * blkW, *(oimg)))
            if save_mask:
                run_mask = run_length_encoding(template.minu_template[i].mask)
                tmp = (len(run_mask),)
                file.write(struct.pack('H', *(tmp)))
                run_mask = np.uint32(run_mask)
                run_mask = tuple(run_mask)
                file.write(struct.pack('I' * len(run_mask), *(run_mask)))

        # save first one texture template
        num_texture_template = len(template.texture_template)
        tmp = (num_texture_template,)
        file.write(struct.pack('H', *tmp))
        if num_texture_template == 0:
            return
        for i in range(num_texture_template):
            minu_num = len(template.texture_template[i].minutiae)
            tmp = (minu_num,)
            file.write(struct.pack('H', *tmp))
            if minu_num > 0:
                x = template.texture_template[i].minutiae[:, 0]
                x = tuple(np.uint16(x))
                file.write(struct.pack('H' * minu_num, *x))
                y = template.texture_template[i].minutiae[:, 1]
                y = tuple(np.uint16(y))
                file.write(struct.pack('H' * minu_num, *y))
                # orientation
                ori = template.texture_template[i].minutiae[:, 2]
                ori = tuple(np.float32(ori))
                file.write(struct.pack('f' * minu_num, *ori))

                des_num = len(template.texture_template[i].des)
                des_len = template.texture_template[i].des[0].shape[1]
                tmp = (des_num, des_len)
                file.write(struct.pack('H' * 2, *tmp))
                for j in range(des_num):
                    descriptor = template.texture_template[i].des[j]
                    descriptor = np.reshape(descriptor, (des_len * minu_num,))
                    descriptor.astype(np.float32)
                    descriptor_tuple = tuple(descriptor)
                    file.write(struct.pack('f' * des_len * minu_num, *(descriptor_tuple)))


def Bin2Template_Byte_TF_C_old(fname, isLatent=True):
    template = Template()
    with open(fname, 'rb') as file:
        all = file.read()
        string = StringIO(all)
        tmp = struct.unpack('H' * 2, string.read(4))
        h = tmp[0]
        w = tmp[1]
        if w <= 0 or h <= 0:
            return None
        tmp = struct.unpack('H' * 2, string.read(4))
        blkH = tmp[0]
        blkW = tmp[1]
        tmp = struct.unpack('B', string.read(1))
        # number of minutiae templates
        num_template = tmp[0]
        for i in range(num_template):
            tmp = struct.unpack('H', string.read(2))
            minu_num = tmp[0]
            if minu_num <= 0:
                minutiae = None
                des = None
                continue
            minutiae = np.zeros((minu_num, 4), dtype=float)

            # x location
            tmp = struct.unpack('H' * minu_num, string.read(2 * minu_num))
            minutiae[:, 0] = np.array(list(tmp))
            # y location
            tmp = struct.unpack('H' * minu_num, string.read(2 * minu_num))
            minutiae[:, 1] = np.array(list(tmp))

            # minutiae orientation
            tmp = struct.unpack('f' * minu_num, string.read(4 * minu_num))
            minutiae[:, 2] = np.array(list(tmp))

            # minutiae reliablility
            tmp = struct.unpack('f' * minu_num, string.read(4 * minu_num))
            minutiae[:, 3] = np.array(list(tmp))

            tmp = struct.unpack('H', string.read(2))
            des_num = tmp[0]
            tmp = struct.unpack('H', string.read(2))
            des_len = tmp[0]
            des = []
            for j in range(des_num):
                tmp = struct.unpack('f' * des_len * minu_num, string.read(4 * des_len * minu_num))
                single_des = np.array(list(tmp))
                single_des = np.reshape(single_des, (minu_num, des_len))
                single_des = np.float32(single_des)
                for k in range(minu_num):
                    single_des[k] = single_des[k] * 1.0 / (LA.norm(single_des[k]) + 0.000001)
                des.append(single_des)

            # get the orientation field
            tmp = struct.unpack('f' * blkH * blkW, string.read(4 * blkH * blkW))
            oimg = np.array(list(tmp))
            oimg = np.reshape(oimg, (blkW, blkH))
            oimg = oimg.transpose()
            minu_template = MinuTemplate(h=h, w=w, blkH=blkH, blkW=blkW,
                                         minutiae=minutiae, des=des, oimg=oimg, mask=None)
            template.add_minu_template(minu_template)

        # load texture template
        tmp = struct.unpack('H', string.read(2))
        num_texture_template = tmp[0]
        if num_texture_template == 0:
            return template

        tmp = struct.unpack('H', string.read(2))
        minu_num = tmp[0]
        if isLatent == 1:
            minutiae = np.zeros((minu_num, 4), dtype=float)
        else:
            minutiae = np.zeros((minu_num, 3), dtype=float)

        if minu_num <= 0:
            minutiae = None
            des = None
            texture_template = TextureTemplate(h=h, w=w, minutiae=minutiae, des=des, mask=None)
            template.add_texture_template(texture_template)
            return

        # x location
        tmp = struct.unpack('H' * minu_num, string.read(2 * minu_num))
        minutiae[:, 0] = np.array(list(tmp))
        # y location
        tmp = struct.unpack('H' * minu_num, string.read(2 * minu_num))
        minutiae[:, 1] = np.array(list(tmp))

        # minutiae orientation
        tmp = struct.unpack('f' * minu_num, string.read(4 * minu_num))
        minutiae[:, 2] = np.array(list(tmp))

        # distance to the border
        if isLatent:
            tmp = struct.unpack('f' * minu_num, string.read(4 * minu_num))
            minutiae[:, 3] = np.array(list(tmp))

        tmp = struct.unpack('H', string.read(2))
        des_num = tmp[0]
        tmp = struct.unpack('H', string.read(2))
        des_len = tmp[0]
        des = []
        for j in range(des_num):
            tmp = struct.unpack('f' * des_len * minu_num, string.read(4 * des_len * minu_num))
            single_des = np.array(list(tmp))
            single_des = np.reshape(single_des, (minu_num, des_len))
            single_des = np.float32(single_des)
            for k in range(minu_num):
                single_des[k] = single_des[k] * 1.0 / (LA.norm(single_des[k]) + 0.000001)
            des.append(single_des)
        texture_template = TextureTemplate(h=h, w=w, minutiae=minutiae, des=des, mask=None)
        template.add_texture_template(texture_template)
    return template


def Template2Bin_Byte_TF_C(fname, template, isLatent=True, save_mask=False):
    Max_BlkSize = 50
    Max_Nrof_minutiae = 1000
    if isLatent:
        Max_Nrof_minutiae = Max_Nrof_minutiae * 2

    with open(fname, 'wb') as file:
        if template is None or len(template.minu_template) == 0:
            tmp = (0, 0, 0, 0)
            file.write(struct.pack('H' * 4, *tmp))
            return
        # save first two minutiae templates
        blkH = template.minu_template[0].blkH
        if blkH > Max_BlkSize:
            blkH = Max_BlkSize
        blkW = template.minu_template[0].blkW
        if blkW > Max_BlkSize:
            blkW = Max_BlkSize

        tmp = (template.minu_template[0].h, template.minu_template[0].w, blkH, blkW)
        file.write(struct.pack('H' * 4, *tmp))
        num_minu_template = len(template.minu_template)
        tmp = (num_minu_template,)
        file.write(struct.pack('B', *tmp))
        for i in range(num_minu_template):
            if len(template.minu_template[i].minutiae) > Max_Nrof_minutiae:
                template.minu_template[i].minutiae = template.minu_template[i].minutiae[:Max_Nrof_minutiae]
            minu_num = len(template.minu_template[i].minutiae)
            tmp = (minu_num,)
            file.write(struct.pack('H', *tmp))
            if minu_num <= 0:
                continue
            x = template.minu_template[i].minutiae[:, 0]
            x = tuple(np.uint16(x))
            file.write(struct.pack('H' * minu_num, *x))
            y = template.minu_template[i].minutiae[:, 1]
            y = tuple(np.uint16(y))
            file.write(struct.pack('H' * minu_num, *y))
            # orientation
            ori = template.minu_template[i].minutiae[:, 2]
            ori = tuple(np.float32(ori))
            file.write(struct.pack('f' * minu_num, *ori))

            reliability = template.minu_template[i].minutiae[:, 3]
            reliability = tuple(np.float32(reliability))
            file.write(struct.pack('f' * minu_num, *reliability))

            if len(template.minu_template[i].des) == minu_num:
                des = template.minu_template[i].des
            elif len(template.minu_template[i].des) == 3:
                des = np.concatenate((template.minu_template[i].des[0], template.minu_template[i].des[1],
                                      template.minu_template[i].des[2]), axis=1)
            elif len(template.minu_template[i].des) == 1:
                des = template.minu_template[i].des[0]

            des_len = des.shape[1]

            tmp = (des_len,)
            file.write(struct.pack('H', *tmp))
            descriptor = np.reshape(des, (des_len * minu_num,))
            descriptor_tuple = tuple(np.float32(descriptor))
            file.write(struct.pack('f' * des_len * minu_num, *(descriptor_tuple)))

            oimg = template.minu_template[i].oimg
            oimg = oimg[:blkH, :blkW]
            oimg = oimg.transpose()
            oimg = tuple(np.reshape(oimg, (blkH * blkW,)))
            file.write(struct.pack('f' * blkH * blkW, *(oimg)))
            if save_mask:
                run_mask = run_length_encoding(template.minu_template[i].mask)
                tmp = (len(run_mask),)
                file.write(struct.pack('H', *(tmp)))
                run_mask = np.uint32(run_mask)
                run_mask = tuple(run_mask)
                file.write(struct.pack('I' * len(run_mask), *(run_mask)))

        # save first one texture template
        num_texture_template = len(template.texture_template)
        tmp = (num_texture_template,)
        file.write(struct.pack('B', *tmp))
        if num_texture_template == 0:
            return
        for i in range(num_texture_template):
            if len(template.texture_template[i].minutiae) > Max_Nrof_minutiae:
                template.texture_template[i].minutiae = template.texture_template[i].minutiae[:Max_Nrof_minutiae]
            minu_num = len(template.texture_template[i].minutiae)
            tmp = (minu_num,)
            file.write(struct.pack('H', *tmp))
            if minu_num > 0:
                x = template.texture_template[i].minutiae[:, 0]
                x = tuple(np.uint16(x))
                file.write(struct.pack('H' * minu_num, *x))
                y = template.texture_template[i].minutiae[:, 1]
                y = tuple(np.uint16(y))
                file.write(struct.pack('H' * minu_num, *y))
                # orientation
                ori = template.texture_template[i].minutiae[:, 2]
                ori = tuple(np.float32(ori))
                file.write(struct.pack('f' * minu_num, *ori))
                if len(template.texture_template[i].des) == minu_num or len(template.texture_template[i].des) == Max_Nrof_minutiae:
                    des = template.texture_template[i].des
                elif len(template.texture_template[i].des) == 3:
                    des = np.concatenate((template.texture_template[i].des[0], template.texture_template[i].des[1],
                                          template.texture_template[i].des[2]), axis=1)
                elif len(template.texture_template[i].des) == 1:
                    des = template.texture_template[i].des[0]

                if des.shape[0] > Max_Nrof_minutiae:
                    des = des[:Max_Nrof_minutiae]
                des_len = des.shape[1]
                tmp = (des_len,)
                file.write(struct.pack('H', *tmp))
                if des_len <= 0:
                    continue

                descriptor = np.reshape(des, (des_len * minu_num,))
                descriptor_tuple = tuple(np.float32(descriptor))
                file.write(struct.pack('f' * des_len * minu_num, *(descriptor_tuple)))


def Bin2Template_Byte_TF_C(fname, isLatent=True):
    template = Template()
    with open(fname, 'rb') as file:
        all = file.read()
        string = BytesIO(all)
        tmp = struct.unpack('H' * 2, string.read(4))
        h = tmp[0]
        w = tmp[1]
        if w <= 0 or h <= 0:
            return None
        tmp = struct.unpack('H' * 2, string.read(4))
        blkH = tmp[0]
        blkW = tmp[1]
        if blkH > 50:
            blkH = 50
        if blkW > 50:
            blkW = 50
        tmp = struct.unpack('B', string.read(1))
        # number of minutiae templates
        num_template = tmp[0]
        for i in range(num_template):
            tmp = struct.unpack('H', string.read(2))
            minu_num = tmp[0]
            if minu_num <= 0:
                minutiae = None
                des = None
                continue
            minutiae = np.zeros((minu_num, 4), dtype=float)

            # x location
            tmp = struct.unpack('H' * minu_num, string.read(2 * minu_num))
            minutiae[:, 0] = np.array(list(tmp))
            # y location
            tmp = struct.unpack('H' * minu_num, string.read(2 * minu_num))
            minutiae[:, 1] = np.array(list(tmp))

            # minutiae orientation
            tmp = struct.unpack('f' * minu_num, string.read(4 * minu_num))
            minutiae[:, 2] = np.array(list(tmp))

            # minutiae reliablility
            tmp = struct.unpack('f' * minu_num, string.read(4 * minu_num))
            minutiae[:, 3] = np.array(list(tmp))

            tmp = struct.unpack('H', string.read(2))
            des_len = tmp[0]
            des = []

            tmp = struct.unpack('f' * des_len * minu_num, string.read(4 * des_len * minu_num))
            single_des = np.array(list(tmp))
            single_des = np.reshape(single_des, (minu_num, des_len))
            single_des = np.float32(single_des)

            # get the orientation field
            tmp = struct.unpack('f' * blkH * blkW, string.read(4 * blkH * blkW))
            oimg = np.array(list(tmp))
            oimg = np.reshape(oimg, (blkW, blkH))
            oimg = oimg.transpose()

            minu_template = MinuTemplate(h=h, w=w, blkH=blkH, blkW=blkW, minutiae=minutiae,
                                         des=single_des, oimg=oimg, mask=None)
            template.add_minu_template(minu_template)

        # load texture template
        tmp = struct.unpack('B', string.read(1))
        num_texture_template = tmp[0]
        if num_texture_template == 0:
            return template
        for i in range(num_texture_template):
            tmp = struct.unpack('H', string.read(2))
            minu_num = tmp[0]
            if isLatent == 1:
                minutiae = np.zeros((minu_num, 4), dtype=float)
            else:
                minutiae = np.zeros((minu_num, 3), dtype=float)

            if minu_num <= 0:
                minutiae = None
                des = None
                texture_template = TextureTemplate(h=h, w=w, minutiae=minutiae, des=des, mask=None)
                template.add_texture_template(texture_template)
                return

            # x location
            tmp = struct.unpack('H' * minu_num, string.read(2 * minu_num))
            minutiae[:, 0] = np.array(list(tmp))
            # y location
            tmp = struct.unpack('H' * minu_num, string.read(2 * minu_num))
            minutiae[:, 1] = np.array(list(tmp))

            # minutiae orientation
            tmp = struct.unpack('f' * minu_num, string.read(4 * minu_num))
            minutiae[:, 2] = np.array(list(tmp))

            tmp = struct.unpack('H', string.read(2))
            des_len = tmp[0]
            des = []

            t = string.read(4 * des_len * minu_num)
            if len(t) < des_len * minu_num * 4:
                texture_template = TextureTemplate(h=h, w=w, minutiae=[], des=[], mask=None)
                template.add_texture_template(texture_template)
                break
            tmp = struct.unpack('f' * des_len * minu_num, t)
            single_des = np.array(list(tmp))
            single_des = np.reshape(single_des, (minu_num, des_len))
            single_des = np.float32(single_des)
            texture_template = TextureTemplate(h=h, w=w, minutiae=minutiae, des=single_des, mask=None)
            template.add_texture_template(texture_template)
    return template


def Template2Bin_Byte_TF_C_PQ(fname, template, isLatent=True, save_mask=False):
    Max_BlkSize = 50
    Max_Nrof_minutiae = 1000
    if isLatent:
        Max_Nrof_minutiae = Max_Nrof_minutiae * 2

    with open(fname, 'wb') as file:
        if template is None or len(template.minu_template) == 0:
            tmp = (0, 0, 0, 0)
            file.write(struct.pack('H' * 4, *tmp))
            return
        # save first two minutiae templates
        blkH = template.minu_template[0].blkH
        if blkH > Max_BlkSize:
            blkH = Max_BlkSize
        blkW = template.minu_template[0].blkW
        if blkW > Max_BlkSize:
            blkW = Max_BlkSize

        tmp = (template.minu_template[0].h, template.minu_template[0].w, blkH, blkW)
        file.write(struct.pack('H' * 4, *tmp))
        num_minu_template = len(template.minu_template)
        tmp = (num_minu_template,)
        file.write(struct.pack('B', *tmp))
        for i in range(num_minu_template):
            if len(template.minu_template[i].minutiae) > Max_Nrof_minutiae:
                template.minu_template[i].minutiae = template.minu_template[i].minutiae[:Max_Nrof_minutiae]
            minu_num = len(template.minu_template[i].minutiae)
            tmp = (minu_num,)
            file.write(struct.pack('H', *tmp))
            if minu_num <= 0:
                continue
            x = template.minu_template[i].minutiae[:, 0]
            x = tuple(np.uint16(x))
            file.write(struct.pack('H' * minu_num, *x))
            y = template.minu_template[i].minutiae[:, 1]
            y = tuple(np.uint16(y))
            file.write(struct.pack('H' * minu_num, *y))
            # orientation
            ori = template.minu_template[i].minutiae[:, 2]
            ori = tuple(np.float32(ori))
            file.write(struct.pack('f' * minu_num, *ori))

            reliability = template.minu_template[i].minutiae[:, 3]
            reliability = tuple(np.float32(reliability))
            file.write(struct.pack('f' * minu_num, *reliability))

            if len(template.minu_template[i].des) == minu_num:
                des = template.minu_template[i].des
            elif len(template.minu_template[i].des) == 3:
                des = np.concatenate((template.minu_template[i].des[0], template.minu_template[i].des[1],
                                      template.minu_template[i].des[2]), axis=1)
            elif len(template.minu_template[i].des) == 1:
                des = template.minu_template[i].des[0]

            des_len = des.shape[1]

            tmp = (des_len,)
            file.write(struct.pack('H', *tmp))
            descriptor = np.reshape(des, (des_len * minu_num,))
            descriptor_tuple = tuple(np.float32(descriptor))
            file.write(struct.pack('f' * des_len * minu_num, *(descriptor_tuple)))

            oimg = template.minu_template[i].oimg
            oimg = oimg[:blkH, :blkW]
            oimg = oimg.transpose()
            oimg = tuple(np.reshape(oimg, (blkH * blkW,)))
            file.write(struct.pack('f' * blkH * blkW, *(oimg)))
            if save_mask:
                run_mask = run_length_encoding(template.minu_template[i].mask)
                tmp = (len(run_mask),)
                file.write(struct.pack('H', *(tmp)))
                run_mask = np.uint32(run_mask)
                run_mask = tuple(run_mask)
                file.write(struct.pack('I' * len(run_mask), *(run_mask)))

        # save first one texture template
        num_texture_template = len(template.texture_template)
        tmp = (num_texture_template,)
        file.write(struct.pack('B', *tmp))
        if num_texture_template == 0:
            return
        for i in range(num_texture_template):
            if len(template.texture_template[i].minutiae) > Max_Nrof_minutiae:
                template.texture_template[i].minutiae = template.texture_template[i].minutiae[:Max_Nrof_minutiae]
            minu_num = len(template.texture_template[i].minutiae)
            tmp = (minu_num,)
            file.write(struct.pack('H', *tmp))
            if minu_num > 0:
                x = template.texture_template[i].minutiae[:, 0]
                x = tuple(np.uint16(x))
                file.write(struct.pack('H' * minu_num, *x))
                y = template.texture_template[i].minutiae[:, 1]
                y = tuple(np.uint16(y))
                file.write(struct.pack('H' * minu_num, *y))
                # orientation
                ori = template.texture_template[i].minutiae[:, 2]
                ori = tuple(np.float32(ori))
                file.write(struct.pack('f' * minu_num, *ori))

                if len(template.texture_template[i].des) == minu_num or len(template.texture_template[i].des) == Max_Nrof_minutiae:
                    des = template.texture_template[i].des
                elif len(template.texture_template[i].des) == 3:
                    des = np.concatenate((template.texture_template[i].des[0], template.texture_template[i].des[1],
                                          template.texture_template[i].des[2]), axis=1)
                elif len(template.texture_template[i].des) == 1:
                    des = template.texture_template[i].des[0]

                if des.shape[0] > Max_Nrof_minutiae:
                    des = des[:Max_Nrof_minutiae]
                des_len = des.shape[1]
                tmp = (des_len,)
                file.write(struct.pack('H', *tmp))
                if des_len <= 0:
                    continue

                descriptor = np.reshape(des, (des_len * minu_num,))
                descriptor_tuple = tuple(np.float32(descriptor))
                file.write(struct.pack('f' * des_len * minu_num, *(descriptor_tuple)))


def Bin2Template_Byte_TF_C_judge(fname, isLatent=True):
    with open(fname, 'rb') as file:
        all = file.read()
        string = StringIO(all)
        tmp = struct.unpack('H' * 2, string.read(4))
        h = tmp[0]
        w = tmp[1]
        if w <= 0 or h <= 0:
            return None
        tmp = struct.unpack('H' * 2, string.read(4))
        tmp = struct.unpack('B', string.read(1))
        # number of minutiae templates
        num_template = tmp[0]
        for i in range(num_template):
            tmp = struct.unpack('H', string.read(2))
            minu_num = tmp[0]
            if minu_num <= 0:
                minutiae = None
                continue
            minutiae = np.zeros((minu_num, 4), dtype=float)

            # x location
            tmp = struct.unpack('H' * minu_num, string.read(2 * minu_num))
            minutiae[:, 0] = np.array(list(tmp))
            # y location
            tmp = struct.unpack('H' * minu_num, string.read(2 * minu_num))
            minutiae[:, 1] = np.array(list(tmp))

            # minutiae orientation
            tmp = struct.unpack('f' * minu_num, string.read(4 * minu_num))
            minutiae[:, 2] = np.array(list(tmp))

            # minutiae reliablility
            tmp = struct.unpack('f' * minu_num, string.read(4 * minu_num))
            minutiae[:, 3] = np.array(list(tmp))

            tmp = struct.unpack('H', string.read(2))
            des_len = tmp[0]
            if des_len == 3:
                return False  # old
            else:
                return True  # new template
            # des = []


def compare_templates():
    import glob
    template_path_1 = '/AutomatedLatentRecognition/Results/template/evaluation_06082018_C++/MSP_0.1_nomask_stride_16/'
    template_path_2 = '/AutomatedLatentRecognition/evaluation_06082018_C++/MSP_0.15_nomask/'
    template_file_list_1 = glob.glob(template_path_1 + '*.dat')
    template_file_list_2 = glob.glob(template_path_2 + '*.dat')
    template_file_list_1.sort(key=lambda filename: int(''.join(filter(str.isdigit, filename))))
    template_file_list_2.sort(key=lambda filename: int(''.join(filter(str.isdigit, filename))))

    max_minu_diff = 0
    max_minu_ind = -1
    max_texture_diff = 0
    max_texture_ind = -1
    for i in range(258):
        template_file_1 = template_file_list_1[i]
        template_file_2 = template_file_list_2[i]

        template_1 = Bin2Template_Byte_TF_C(template_file_1, isLatent=False)
        template_2 = Bin2Template_Byte_TF_C(template_file_2, isLatent=False)
        diff = abs(len(template_1.minu_template[0].minutiae) - len(template_2.minu_template[0].minutiae))
        if diff > max_minu_diff:
            max_minu_diff = diff
            max_minu_ind = i
        diff = abs(len(template_1.texture_template[0].minutiae) - len(template_2.texture_template[0].minutiae))
        if diff > max_texture_diff:
            max_texture_diff = diff
            max_texture_ind = i
    template_file_1 = template_file_list_1[max_minu_ind]
    template_file_2 = template_file_list_2[max_minu_ind]

    template_1 = Bin2Template_Byte_TF_C(template_file_1, isLatent=False)
    template_2 = Bin2Template_Byte_TF_C(template_file_2, isLatent=False)
    print(len(template_1.minu_template[0].minutiae))
    print(len(template_2.minu_template[0].minutiae))
    img_path = '/Data/MSP_background/images/'
    from skimage import io
    img = io.imread(img_path + str(max_minu_ind + 1) + '.bmp', s_grey=True)

    template_file_1 = template_file_list_1[max_texture_ind]
    template_file_2 = template_file_list_2[max_texture_ind]

    template_1 = Bin2Template_Byte_TF_C(template_file_1, isLatent=False)
    template_2 = Bin2Template_Byte_TF_C(template_file_2, isLatent=False)
    print(len(template_1.texture_template[0].minutiae))
    print(len(template_2.texture_template[0].minutiae))

    img = io.imread(img_path + str(max_texture_ind + 1) + '.bmp', s_grey=True)


if __name__ == '__main__':
    compare_templates()
