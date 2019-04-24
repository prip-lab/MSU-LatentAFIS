import glob
from matplotlib.patches import Circle
import os.path
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np
import preprocessing as LP
from scipy.ndimage import rotate
matplotlib.interactive(False)


def show_minutiae(img, minutiae, ROI=None, fname=None, block=True):
    # for the latent or the low quality rolled print
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    R = 10
    arrow_len = 15
    if ROI is not None:
        h, w = ROI.shape
        for i in range(h):
            for j in range(w):
                if ROI[i, j] == 0:
                    img[i, j] = 255

    ax.imshow(img, cmap='gray')
    minu_num = len(minutiae)
    for i in range(0, minu_num):
        xx = minutiae[i][0]
        yy = minutiae[i][1]
        circ = Circle((xx, yy), R, color='r', fill=False)
        ax.add_patch(circ)

        ori = -minutiae[i][2]
        dx = math.cos(ori) * arrow_len
        dy = math.sin(ori) * arrow_len
        ax.arrow(xx, yy, dx, dy, head_width=0.05, head_length=0.1, fc='r', ec='r')
    plt.show(block=block)
    if fname is not None:
        fig.savefig(fname, dpi=600)
        plt.close()


def show_minutiae_sets(img, minutiae_sets, ROI=None, fname=None, block=True):
    # for the latent or the low quality rolled print
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')

    arrow_len = 15
    if ROI is not None:
        h, w = ROI.shape
        for i in range(h):
            for j in range(w):
                if ROI[i, j] == 0:
                    img[i, j] = 255

    ax.imshow(img, cmap='gray')
    color = ['r', 'b']
    R = [10, 8, 6]
    for k in range(len(minutiae_sets)):
        minutiae = minutiae_sets[k]
        minu_num = len(minutiae)
        for i in range(0, minu_num):
            xx = minutiae[i, 0]
            yy = minutiae[i, 1]
            circ = Circle((xx, yy), R[k], color=color[k], fill=False)
            ax.add_patch(circ)

            ori = -minutiae[i, 2]
            dx = math.cos(ori) * arrow_len
            dy = math.sin(ori) * arrow_len
            ax.arrow(xx, yy, dx, dy, head_width=0.05, head_length=0.1, fc=color[k], ec=color[k])

    plt.show(block=block)
    if fname is not None:
        fig.savefig(fname, dpi=600)
        plt.close()


def modify_minutiae_cylinder(input_file, output_file, angle=None, processing=None):

    cylinder = np.load(input_file)
    img = cylinder[:, :, 0]
    if processing == 'STFT':
        img = LP.STFT(img)
    cylinder[:, :, 0] = img
    np.save(output_file, cylinder)


def extract_minutiae_cylinder(img_input, minutiae_input, ROI=None, num_ori=12, angle=None, processing=None):
    # for the latent or the low quality rolled print
    minutiae = minutiae_input.copy()
    img = img_input.copy()
    if processing == 'STFT':
        img = LP.STFT(img)
    elif processing == 'contrast':
        img = LP.local_constrast_enhancement(img)
    elif processing == 'texture':
        img = LP.FastCartoonTexture(img)
    sigma = 5**2
    if ROI is not None:
        h, w = ROI.shape
        for i in range(h):
            for j in range(w):
                if ROI[i, j] == 0:
                    img[i, j] = 255

        h, w = ROI.shape
        col_sum = np.sum(ROI, axis=0)

        ind = [x for x in range(len(col_sum)) if col_sum[x] > 0]
        min_x = np.max([np.min(ind) - 32, 0])
        max_x = np.min([np.max(ind) + 32, w])

        row_sum = np.sum(ROI, axis=1)

        ind = [x for x in range(len(row_sum)) if row_sum[x] > 0]
        min_y = np.max([np.min(ind) - 32, 0])
        max_y = np.min([np.max(ind) + 32, h])

        ROI = ROI[min_y:max_y, min_x:max_x]
        img = img[min_y:max_y, min_x:max_x]
        minutiae[:, 0] = minutiae[:, 0] - min_x
        minutiae[:, 1] = minutiae[:, 1] - min_y
    else:
        h, w = img.shape[0:2]
        ROI = np.ones((h, w))

    # rotate the image and ROI, and also update minutiae points
    h0, w0 = ROI.shape
    if angle is not None:
        h02 = (h0 + 1) / 2
        w02 = (w0 + 1) / 2

        img = rotate(img, angle)
        ROI = rotate(ROI, angle)

        h, w = ROI.shape
        h2 = (h + 1) / 2
        w2 = (w + 1) / 2

        angle = -angle / 180.0 * math.pi
        cosTheta = math.cos(angle)
        sinTheta = math.sin(angle)
        xx = (minutiae[:, 0] - w02) * cosTheta - (minutiae[:, 1] - h02) * sinTheta + w2

        yy = (minutiae[:, 0] - w02) * sinTheta + (minutiae[:, 1] - h02) * cosTheta + h2
        ori = minutiae[:, 2] - angle
        #
        minutiae[:, 0] = xx
        minutiae[:, 1] = yy
        minutiae[:, 2] = ori
        show = 0
        if show:
            minu_num = minutiae.shape[0]
            fig, ax = plt.subplots(1)
            ax.set_aspect('equal')

            R = 10
            arrow_len = 15
            ax.imshow(img, cmap='gray')
            for i in range(0, minu_num):
                xx = minutiae[i, 0]
                yy = minutiae[i, 1]
                circ = Circle((xx, yy), R, color='r', fill=False)
                ax.add_patch(circ)

                ori = -minutiae[i, 2]
                dx = math.cos(ori) * arrow_len
                dy = math.sin(ori) * arrow_len
                ax.arrow(xx, yy, dx, dy, head_width=0.05, head_length=0.1, fc='r', ec='r')
            plt.show()
    h, w = ROI.shape
    minutiae_cylinder = np.zeros((h, w, num_ori), dtype=float)
    cylinder_ori = np.asarray(range(num_ori)) * math.pi * 2 / num_ori

    Y, X = np.mgrid[0:h, 0:w]
    minu_num = minutiae.shape[0]
    for i in range(0, minu_num):
        xx = minutiae[i, 0]
        yy = minutiae[i, 1]
        if yy < 0 or xx < 0:
            continue
            print xx, yy
            minu_num = minutiae.shape[0]
            fig, ax = plt.subplots(1)
            ax.set_aspect('equal')

            R = 10
            arrow_len = 15
            ax.imshow(img, cmap='gray')
            for i in range(0, minu_num):
                xx = minutiae[i, 0]
                yy = minutiae[i, 1]
                circ = Circle((xx, yy), R, color='r', fill=False)
                ax.add_patch(circ)

                ori = -minutiae[i, 2]
                dx = math.cos(ori) * arrow_len
                dy = math.sin(ori) * arrow_len
                ax.arrow(xx, yy, dx, dy, head_width=0.05, head_length=0.1, fc='r', ec='r')
            plt.show()
        weight = np.exp(-((X - xx) * (X - xx) + (Y - yy) * (Y - yy)) / sigma)

        ori = minutiae[i, 2]
        if ori < 0:
            ori += np.pi * 2
        if ori > np.pi * 2:
            ori -= np.pi * 2

        for j in range(num_ori):

            ori_diff = np.fabs(ori - cylinder_ori[j])

            if ori_diff > np.pi * 2:
                ori_diff = ori_diff - np.pi * 2

            ori_diff = np.min([ori_diff, np.pi * 2 - ori_diff])
            minutiae_cylinder[:, :, j] += weight * np.exp(-ori_diff / np.pi * 6)
    show = 0
    if show:
        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')

        R = 10
        arrow_len = 15
        ax.imshow(img, cmap='gray')
        for i in range(0, minu_num):
            xx = minutiae[i, 0]
            yy = minutiae[i, 1]
            circ = Circle((xx, yy), R, color='r', fill=False)
            ax.add_patch(circ)

            ori = -minutiae[i, 2]
            dx = math.cos(ori) * arrow_len
            dy = math.sin(ori) * arrow_len
            ax.arrow(xx, yy, dx, dy, head_width=0.05, head_length=0.1, fc='r', ec='r')
        plt.show()

    return img, ROI, minutiae_cylinder


def get_minutiae_from_cylinder(minutiae_cylinder, thr=0.5):
    h, w, c = minutiae_cylinder.shape

    max_arg = np.argmax(minutiae_cylinder, axis=2)
    max_val = np.max(minutiae_cylinder, axis=2)
    r = 2
    minutiae = []
    for i in range(r, h - r):
        for j in range(r, w - r):
            v = max_val[i, j]
            ind = max_arg[i, j]
            if v < thr:
                continue
            if ind == 0:
                local_value = np.concatenate((minutiae_cylinder[i - r:i + r + 1, j - r:j + r + 1, -1::],
                                              minutiae_cylinder[i - r:i + r + 1, j - r:j + r + 1, 0:2]), 2)
            elif ind == c - 1:
                local_value = np.concatenate((minutiae_cylinder[i - r:i + r + 1, j - r:j + r + 1, -2:-1],
                                              minutiae_cylinder[i - r:i + r + 1, j - r:j + r + 1, -1::],
                                              minutiae_cylinder[i - r:i + r + 1, j - r:j + r + 1, 0:1]), 2)
            else:
                local_value = minutiae_cylinder[i - r:i + r + 1, j - r:j + r + 1, ind - 1:ind + 2]
            local_value = local_value.copy()
            local_value[r, r, 1] = 0
            local_max_ind = np.argmax(local_value)
            local_max_ind = np.unravel_index(local_max_ind, local_value.shape)
            local_max_v = local_value[local_max_ind]

            if local_max_v > v:
                continue

            # refine the minutiae orientation
            ind_1 = ind - 1
            if ind_1 < 0:
                ind_1 = ind_1 + c

            ind_2 = ind + 1
            if ind_2 >= c:
                ind_2 = ind_2 - c

            y1 = minutiae_cylinder[i, j, ind_1]
            y2 = minutiae_cylinder[i, j, ind] - y1
            y3 = minutiae_cylinder[i, j, ind_2] - y1
            pred = 0.5 * (y3 - 4 * y2) / (y3 - 2 * y2)
            confidence = -(2 * y2 - 0.5 * y3) * (2 * y2 - 0.5 * y3) / (2 * y3 - 4 * y2) + v
            if confidence < v:
                print confidence, v
            ori_ind = ind_1 + pred
            ori = ori_ind * 1.0 / c * 2 * math.pi
            minutiae.append([j, i, ori, confidence])  # v is the confidence
    if len(minutiae) > 0:
        minutiae = np.asarray(minutiae, dtype=np.float32)
        I = np.argsort(minutiae[:, 3])
        I = I[::-1]
        minutiae = minutiae[I, :]
    return minutiae


def get_minutiae_from_cylinder2(minutiae_cylinder, thr=0.5):
    h, w, c = minutiae_cylinder.shape

    max_arg = np.argmax(minutiae_cylinder, axis=2)
    max_val = np.max(minutiae_cylinder, axis=2)

    candi_ind = np.where(max_val > thr)

    candi_num = len(candi_ind[0])

    r = 15
    r2 = int(r / 2)
    minutiae = []

    for k in range(candi_num):
        i = candi_ind[0][k]
        j = candi_ind[1][k]
        if i < r2 or j < r2 or i > h - r2 - 1 or j > w - r2 - 1:
            continue
        v = max_val[i, j]
        if v > max_val[i - 1, j - 1] and v > max_val[i - 1, j] and v > max_val[i - 1, j + 1] \
                and v > max_val[i, j - 1] and v > max_val[i, j + 1] \
                and v > max_val[i + 1, j - 1] and v > max_val[i + 1, j] and v > max_val[i + 1, j + 1]:

            v = max_val[i, j]
            ind = max_arg[i, j]

            # refine the minutiae orientation
            ind_1 = ind - 1
            if ind_1 < 0:
                ind_1 = ind_1 + c
            ind_2 = ind + 1
            if ind_2 >= c:
                ind_2 = ind_2 - c

            y1 = minutiae_cylinder[i, j, ind_1]
            y2 = minutiae_cylinder[i, j, ind] - y1
            y3 = minutiae_cylinder[i, j, ind_2] - y1
            pred = 0.5 * (y3 - 4 * y2) / (y3 - 2 * y2)
            confidence = -(2 * y2 - 0.5 * y3) * (2 * y2 - 0.5 * y3) / (2 * y3 - 4 * y2) + v
            ori_ind = ind_1 + pred
            ori = ori_ind * 1.0 / c * 2 * math.pi
            minutiae.append([j, i, ori, confidence])  # v is the confidence
    if len(minutiae) > 0:
        minutiae = np.asarray(minutiae, dtype=np.float32)
        I = np.argsort(minutiae[:, 3])
        I = I[::-1]
        minutiae = minutiae[I, :]
    return minutiae


def extract_minutiae_cylinder_2c(img, minutiae, ROI, num_ori=12):
    # for the latent or the low quality rolled print
    # extract 2 channel minutiae cylinder
    sigma = 5**2
    if ROI is not None:
        h, w = ROI.shape
        for i in range(h):
            for j in range(w):
                if ROI[i, j] == 0:
                    img[i, j] = 255

    h, w = ROI.shape
    col_sum = np.sum(ROI, axis=0)

    ind = [x for x in range(len(col_sum)) if col_sum[x] > 0]
    min_x = np.max([np.min(ind) - 32, 0])
    max_x = np.min([np.max(ind) + 32, w])

    row_sum = np.sum(ROI, axis=1)

    ind = [x for x in range(len(row_sum)) if row_sum[x] > 0]
    min_y = np.max([np.min(ind) - 32, 0])
    max_y = np.min([np.max(ind) + 32, h])

    ROI = ROI[min_y:max_y, min_x:max_x]
    img = img[min_y:max_y, min_x:max_x]
    minutiae[:, 0] = minutiae[:, 0] - min_x
    minutiae[:, 1] = minutiae[:, 1] - min_y

    h, w = ROI.shape

    minutiae_cylinder = np.zeros((h, w, 3), dtype=float) - 1

    Y, X = np.mgrid[0:h, 0:w]

    minu_num = minutiae.shape[0]
    for i in range(0, minu_num):
        xx = minutiae[i, 0]
        yy = minutiae[i, 1]
        if yy < 0 or xx < 0:
            print xx, yy
        weight = np.exp(-((X - xx) * (X - xx) + (Y - yy) * (Y - yy)) / sigma)

        ori = minutiae[i, 2]
        if ori < 0:
            ori += np.pi * 2
        if ori > np.pi * 2:
            ori -= np.pi * 2

        minutiae_cylinder[:, :, 1] += weight * math.cos(ori)
        minutiae_cylinder[:, :, 2] += weight * math.sin(ori)

    return img, ROI, minutiae_cylinder


def get_minutiae_from_cylinder_2c(minutiae_cylinder, thr=0.5):
    h, w, c = minutiae_cylinder.shape

    r = 2
    minutiae = []
    for i in range(r, h - r):
        for j in range(r, w - r):
            ind = np.argmax(minutiae_cylinder[i, j, :])
            v = minutiae_cylinder[i, j, ind]
            # print ind, v
            if v < thr:
                continue
            if ind == 0:
                local_value = np.concatenate((minutiae_cylinder[i - r:i + r + 1, j - r:j + r + 1, -1::],
                                              minutiae_cylinder[i - r:i + r + 1, j - r:j + r + 1, 0:2]), 2)
            elif ind == c - 1:
                local_value = np.concatenate((minutiae_cylinder[i - r:i + r + 1, j - r:j + r + 1, 0:1],
                                              minutiae_cylinder[i - r:i + r + 1, j - r:j + r + 1, -1::],
                                              minutiae_cylinder[i - r:i + r + 1, j - r:j + r + 1, -2:-1]), 2)
            else:
                local_value = minutiae_cylinder[i - r:i + r + 1, j - r:j + r + 1, ind - 1:ind + 2]
            local_value[r, r, r] = 0
            local_max_ind = np.argmax(local_value)
            local_max_ind = np.unravel_index(local_max_ind, local_value.shape)
            local_max_v = local_value[local_max_ind]

            if local_max_v > v:
                continue

            minutiae.append([j, i, ind * 1.0 / c * 2 * math.pi, v])  # v is the confidence

    minutiae = np.asarray(minutiae, dtype=np.float32)
    I = np.argsort(minutiae[:, 3])
    I = I[::-1]
    minutiae = minutiae[I, :]
    return minutiae


def refine_minutiae(minutiae, dist_thr=10, ori_dist=np.pi / 4):
    minu_num = len(minutiae)
    flag = np.ones((minu_num,), dtype=np.int)
    if len(minutiae) == 0:
        return minutiae
    for i in xrange(minu_num):
        x0 = minutiae[i, 0]
        y0 = minutiae[i, 1]
        ori0 = minutiae[i, 2]
        for j in xrange(i + 1, minu_num):
            x1 = minutiae[j, 0]
            y1 = minutiae[j, 1]
            ori1 = minutiae[j, 2]

            dist = np.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))

            if dist < dist_thr:
                flag[j] = 0
                continue

            ori_diff = np.fabs(ori1 - ori0)
            ori_diff = np.min([ori_diff, np.pi * 2 - ori_diff])
            if dist < 20 and ori_diff < ori_dist:
                flag[j] = 0

    minutiae = minutiae[flag == 1, :]
    return minutiae


def process_kais_markup(pathname='/home/kaicao/Dropbox/Share/markup/data/selected_prints_templates_Kai/',
                        data_path='/home/kaicao/Research/AutomatedLatentRecognition/Data/minutiae_cylinder/',
                        num_channels=12):
    subjects = glob.glob(pathname + '*')

    for i, subject in enumerate(subjects):
        print i
        feature_file = subject + '/feature.mat'
        subjectID = subject.split('/')[-1]
        if os.path.isfile(feature_file):
            print feature_file
            x = loadmat(feature_file)
            if 'minutiae_rolled' in x.keys():
                img, ROI, minutiae_cylinder = extract_minutiae_cylinder(x['img_rolled'], x['minutiae_rolled'],
                                                                        x['ROI_rolled_final'])

                img = np.uint8(img)
                if num_channels < 3:
                    minutiae_cylinder = (minutiae_cylinder + 1.0) / 2.0
                    minutiae_cylinder[minutiae_cylinder > 1] = 1.
                    minutiae_cylinder[minutiae_cylinder < 0] = 0.
                    minutiae_cylinder = np.uint8(minutiae_cylinder * 255)

                else:
                    minutiae_cylinder[minutiae_cylinder > 1] = 1
                    minutiae_cylinder = np.uint8(minutiae_cylinder * 255)

                matrix = np.concatenate((np.expand_dims(img, axis=2), np.expand_dims(ROI, axis=2),
                                         minutiae_cylinder), 2)

                outfile = data_path + subjectID + '_rolled.npy'
                np.save(outfile, matrix)

                img, ROI, minutiae_cylinder = extract_minutiae_cylinder(x['img_latent'], x['minutiae_latent'],
                                                                        x['ROI_latent_final'])

                if num_channels < 3:
                    minutiae_cylinder = (minutiae_cylinder + 1.0) / 2.0
                    minutiae_cylinder[minutiae_cylinder > 1] = 1.
                    minutiae_cylinder[minutiae_cylinder < 0] = 0.
                    minutiae_cylinder = np.uint8(minutiae_cylinder * 255)

                else:
                    minutiae_cylinder[minutiae_cylinder > 1] = 1
                    minutiae_cylinder = np.uint8(minutiae_cylinder * 255)

                matrix = np.concatenate((np.expand_dims(img, axis=2), np.expand_dims(ROI, axis=2), minutiae_cylinder),
                                        2)

                outfile = data_path + subjectID + '_latent.npy'
                np.save(outfile, matrix)
            else:
                img, ROI, minutiae_cylinder = extract_minutiae_cylinder(x['img_rolled'], x['minutiae_rolled_final'],
                                                                        x['ROI_rolled_final'])

                img = LP.local_constrast_enhancement(img)
                img = (img + 1) * 128
                img[img > 255] = 255
                img = np.uint8(img)

                if num_channels < 3:
                    minutiae_cylinder = (minutiae_cylinder + 1.0) / 2.0
                    minutiae_cylinder[minutiae_cylinder > 1] = 1.
                    minutiae_cylinder[minutiae_cylinder < 0] = 0.
                    minutiae_cylinder = np.uint8(minutiae_cylinder * 255)

                else:
                    minutiae_cylinder[minutiae_cylinder > 1] = 1
                    minutiae_cylinder = np.uint8(minutiae_cylinder * 255)

                matrix = np.concatenate((np.expand_dims(img, axis=2), np.expand_dims(ROI, axis=2), minutiae_cylinder),
                                        2)

                outfile = data_path + subjectID + '_rolled.npy'
                np.save(outfile, matrix)

                img, ROI, minutiae_cylinder = extract_minutiae_cylinder(x['img_latent'], x['minutiae_latent_final'],
                                                                        x['ROI_latent_final'])

                if num_channels < 3:
                    minutiae_cylinder = (minutiae_cylinder + 1.0) / 2.0
                    minutiae_cylinder[minutiae_cylinder > 1] = 1.
                    minutiae_cylinder[minutiae_cylinder < 0] = 0.
                    minutiae_cylinder = np.uint8(minutiae_cylinder * 255)

                else:
                    minutiae_cylinder[minutiae_cylinder > 1] = 1
                    minutiae_cylinder = np.uint8(minutiae_cylinder * 255)

                matrix = np.concatenate((np.expand_dims(img, axis=2), np.expand_dims(ROI, axis=2), minutiae_cylinder),
                                        2)

                outfile = data_path + subjectID + '_latent.npy'
                np.save(outfile, matrix)


def process_MSPLatents_markup(pathname='/home/kaicao/Dropbox/Share/markup/data/selected_prints_templates_Kai/',
                              data_path='/home/kaicao/Research/AutomatedLatentRecognition/Data/minutiae_cylinder/',
                              num_channels=12, processing='STFT'):
    files = glob.glob(pathname + '*.mat')
    files.sort()
    for file in files:
        print file
        filename = file.split('/')[-1].split('.')[0]
        latent_name, rolled_name = filename.split('_vs_')
        x = loadmat(file)
        for i, angle in enumerate(range(0, 360, 90)):
            outfile = data_path + rolled_name + '_rolled_' + str(i) + '.npy'
            img, ROI, minutiae_cylinder = extract_minutiae_cylinder(
                x['img_rolled'], x['minutiae_rolled_final'], ROI=None, angle=angle, processing=processing)

            if num_channels < 3:
                minutiae_cylinder = (minutiae_cylinder + 1.0) / 2.0
                minutiae_cylinder[minutiae_cylinder > 1] = 1.
                minutiae_cylinder[minutiae_cylinder < 0] = 0.
                minutiae_cylinder = np.uint8(minutiae_cylinder * 255)

            else:
                minutiae_cylinder[minutiae_cylinder > 1] = 1
                minutiae_cylinder = np.uint8(minutiae_cylinder * 255)

            matrix = np.concatenate((np.expand_dims(img, axis=2), np.expand_dims(ROI, axis=2),
                                     minutiae_cylinder), 2)

            np.save(outfile, matrix)

        for i, angle in enumerate(range(0, 360, 90)):
            outfile = data_path + rolled_name + '_latent_' + str(i) + '.npy'
            # if os.path.exists(outfile):
            #     continue
            img, ROI, minutiae_cylinder = extract_minutiae_cylinder(
                x['img_latent'], x['minutiae_latent_final'], ROI=None, angle=angle, processing=processing)

            if num_channels < 3:
                minutiae_cylinder = (minutiae_cylinder + 1.0) / 2.0
                minutiae_cylinder[minutiae_cylinder > 1] = 1.
                minutiae_cylinder[minutiae_cylinder < 0] = 0.
                minutiae_cylinder = np.uint8(minutiae_cylinder * 255)

            else:
                minutiae_cylinder[minutiae_cylinder > 1] = 1
                minutiae_cylinder = np.uint8(minutiae_cylinder * 255)

            matrix = np.concatenate((np.expand_dims(img, axis=2), np.expand_dims(ROI, axis=2),
                                     minutiae_cylinder), 2)
            np.save(outfile, matrix)


def process_FVC_markup_from_minutiae(pathname,
                                     data_path='/home/kaicao/Research/AutomatedLatentRecognition/Data/minutiae_cylinder/',
                                     num_channels=12):
    mat_files = glob.glob(pathname + '*.mat')
    mat_files.sort()
    for i, mat_file in enumerate(mat_files):
        fname = os.path.basename(mat_file)
        print i, fname
        x = loadmat(mat_file)
        img = x['img']
        h, w = img.shape
        ROI = np.ones((h, w))
        img, ROI, minutiae_cylinder = extract_minutiae_cylinder(img, x['minutiae'],
                                                                ROI)

        img = LP.local_constrast_enhancement(img)
        img = (img + 1) * 128
        img[img > 255] = 255
        img = np.uint8(img)
        if num_channels < 3:
            minutiae_cylinder = (minutiae_cylinder + 1.0) / 2.0
            minutiae_cylinder[minutiae_cylinder > 1] = 1.
            minutiae_cylinder[minutiae_cylinder < 0] = 0.
            minutiae_cylinder = np.uint8(minutiae_cylinder * 255)
        else:
            minutiae_cylinder[minutiae_cylinder > 1] = 1
            minutiae_cylinder = np.uint8(minutiae_cylinder * 255)

        matrix = np.concatenate((np.expand_dims(img, axis=2), np.expand_dims(ROI, axis=2), minutiae_cylinder),
                                2)
        outfile = data_path + fname + '.npy'
        np.save(outfile, matrix)


def process_HighSign_markup_from_minutiae(img_path, minu_path,
                                          data_path='/home/kaicao/Research/AutomatedLatentRecognition/Data/minutiae_cylinder/',
                                          num_channels=12, processing='STFT'):
    minu_files = glob.glob(minu_path + '*.txt')
    minu_files.sort()
    from scipy import misc
    for i, minu_file in enumerate(minu_files):
        fname = os.path.basename(minu_file)
        print i, minu_file
        img_file = img_path + fname[:-3] + 'bmp'
        img0 = misc.imread(img_file, mode='L')
        minutiae = np.loadtxt(minu_file, delimiter=',')
        for j, angle in enumerate(range(0, 360, 90)):
            outfile = data_path + fname[:-3] + '_' + str(j) + '.npy'
            img, ROI, minutiae_cylinder = extract_minutiae_cylinder(img0, minutiae,
                                                                    ROI=None, angle=angle, processing=processing)

            if num_channels < 3:
                minutiae_cylinder = (minutiae_cylinder + 1.0) / 2.0
                minutiae_cylinder[minutiae_cylinder > 1] = 1.
                minutiae_cylinder[minutiae_cylinder < 0] = 0.
                minutiae_cylinder = np.uint8(minutiae_cylinder * 255)
            else:
                minutiae_cylinder[minutiae_cylinder > 1] = 1
                minutiae_cylinder = np.uint8(minutiae_cylinder * 255)

            matrix = np.concatenate((np.expand_dims(img, axis=2), np.expand_dims(ROI, axis=2), minutiae_cylinder),
                                    2)

            np.save(outfile, matrix)


def process_FVC_markup(pathname, data_path):
    mat_files = glob.glob(pathname + '*.mat')
    mat_files.sort()

    for i, mat_file in enumerate(mat_files):
        fname = os.path.basename(mat_file)
        print i, fname
        x = loadmat(mat_file)
        minutiae_cylinder = x['minutiae_cylinder']
        minutiae_cylinder = np.uint8(minutiae_cylinder)
        img = x['img']
        ROI = x['ROI']
        ROI = np.uint8(ROI)
        matrix = np.concatenate((np.expand_dims(img, axis=2), np.expand_dims(ROI, axis=2), minutiae_cylinder),
                                2)

        outfile = data_path + fname[:-3] + 'npy'
        np.save(outfile, matrix)


if __name__ == '__main__':
    # process HighSign latents
    img_path = '/media/kaicao/Data/AutomatedLatentRecognition/Hisign/UL2Bmp/UL2Bmp/'
    minu_path = '/media/kaicao/Data/AutomatedLatentRecognition/Hisign/UL2Bmp_Manual_minu/'
    processing = 'STFT'
    data_path = '/media/kaicao/data2/AutomatedLatentRecognition/Data/minutiae_cylinder_uint8_MSPLatents_' + processing + '/'

    process_HighSign_markup_from_minutiae(img_path, minu_path,
                                          data_path=data_path, num_channels=12, processing='STFT')
