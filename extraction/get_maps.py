import numpy as np
import preprocessing
import matplotlib.pylab as plt
import math
import glob
import scipy.ndimage
from skimage.filters import gaussian, gabor_kernel
import show
import cv2
from skimage.morphology import binary_opening, binary_closing


class local_STFT:
    def __init__(self, patch, weight=None, dBPass=None):
        if weight is not None:
            patch = patch * weight
        patch = patch - np.mean(patch)
        norm = np.linalg.norm(patch)
        patch = patch / (norm + 0.000001)

        f = np.fft.fft2(patch)
        fshift = np.fft.fftshift(f)
        if dBPass is not None:
            fshift = dBPass * fshift

        self.patch_FFT = fshift
        self.patch = patch
        self.ori = None
        self.fre = None
        self.confidence = None
        self.patch_size = patch.shape[0]

    def analysis(self, r, dir_ind_list=None, N=2):
        assert(dir_ind_list is not None)
        energy = np.abs(self.patch_FFT)
        energy = energy / (np.sum(energy) + 0.00001)
        nrof_dirs = len(dir_ind_list)

        ori_interval = math.pi / nrof_dirs
        ori_interval2 = ori_interval / 2

        pad_size = 1
        dir_norm = np.zeros((nrof_dirs + 2,))
        for i in range(nrof_dirs):
            tmp = energy[dir_ind_list[i][:, 0], dir_ind_list[i][:, 1]]
            dir_norm[i + 1] = np.sum(tmp)

        dir_norm[0] = dir_norm[nrof_dirs]
        dir_norm[nrof_dirs + 1] = dir_norm[1]

        # smooth dir_norm
        smoothed_dir_norm = dir_norm
        for i in range(1, nrof_dirs + 1):
            smoothed_dir_norm[i] = (dir_norm[i - 1] + dir_norm[i] * 4 + dir_norm[i + 1]) / 6

        smoothed_dir_norm[0] = smoothed_dir_norm[nrof_dirs]
        smoothed_dir_norm[nrof_dirs + 1] = smoothed_dir_norm[1]

        den = np.sum(smoothed_dir_norm[1:nrof_dirs + 1]) + 0.00001  # verify if den == 1
        smoothed_dir_norm = smoothed_dir_norm / den  # normalization if den == 1, this line can be removed

        ori = []
        fre = []
        confidence = []

        wenergy = energy * r
        for i in range(1, nrof_dirs + 1):
            if smoothed_dir_norm[i] > smoothed_dir_norm[i - 1] and smoothed_dir_norm[i] > smoothed_dir_norm[i + 1]:
                tmp_ori = (i - pad_size) * ori_interval + ori_interval2 + math.pi / 2
                ori.append(tmp_ori)
                confidence.append(smoothed_dir_norm[i])
                tmp_fre = np.sum(wenergy[dir_ind_list[i - pad_size][:, 0],
                                 dir_ind_list[i - pad_size][:, 1]]) / dir_norm[i]
                tmp_fre = 1 / (tmp_fre + 0.00001)
                fre.append(tmp_fre)

        if len(confidence) > 0:
            confidence = np.asarray(confidence)
            fre = np.asarray(fre)
            ori = np.asarray(ori)
            ind = confidence.argsort()[::-1]
            confidence = confidence[ind]
            fre = fre[ind]
            ori = ori[ind]
            if len(confidence) >= 2 and confidence[0] / confidence[1] > 2.0:
                self.ori = [ori[0]]
                self.fre = [fre[0]]
                self.confidence = [confidence[0]]
            elif len(confidence) > N:
                fre = fre[:N]
                ori = ori[:N]
                confidence = confidence[:N]
                self.ori = ori
                self.fre = fre
                self.confidence = confidence
            else:
                self.ori = ori
                self.fre = fre
                self.confidence = confidence

    def get_features_of_topN(self, N=2):
        if self.confidence is None:
            self.border_wave = None
            return
        candi_num = len(self.ori)
        candi_num = np.min([candi_num, N])
        patch_size = self.patch_FFT.shape
        for i in range(candi_num):
            kernel = gabor_kernel(self.fre[i], theta=self.ori[i], sigma_x=10, sigma_y=10)
            kernel_f = np.fft.fft2(kernel.real, patch_size)
            kernel_f = np.fft.fftshift(kernel_f)
            patch_f = self.patch_FFT * kernel_f

            patch_f = np.fft.ifftshift(patch_f)
            rec_patch = np.real(np.fft.ifft2(patch_f))

            plt.subplot(121), plt.imshow(self.patch, cmap='gray')
            plt.title('Input patch'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(rec_patch, cmap='gray')
            plt.title('filtered patch'), plt.xticks([]), plt.yticks([])
            plt.show()

    def reconstruction(self, weight=None):
        f_ifft = np.fft.ifftshift(self.patch_FFT)
        rec_patch = np.real(np.fft.ifft2(f_ifft))
        if weight is not None:
            rec_patch = rec_patch * weight
        return rec_patch

    def gabor_filtering(self, theta, fre, weight=None):
        patch_size = self.patch_FFT.shape
        kernel = gabor_kernel(fre, theta=theta, sigma_x=4, sigma_y=4)
        f = kernel.real
        f = f - np.mean(f)
        f = f / (np.linalg.norm(f) + 0.0001)

        kernel_f = np.fft.fft2(f, patch_size)
        kernel_f = np.fft.fftshift(kernel_f)
        patch_f = self.patch_FFT * kernel_f

        patch_f = np.fft.ifftshift(patch_f)  # *np.sqrt(np.abs(fshift)))
        rec_patch = np.real(np.fft.ifft2(patch_f))
        if weight is not None:
            rec_patch = rec_patch * weight
        return rec_patch


def get_dir_map_gradient(img, mask=None, block_size=16, sigma=5):
    h, w = img.shape
    blkH = h // block_size + 1
    blkW = w // block_size + 1

    img = img.astype(dtype=np.float, copy=False)
    img = gaussian(img, 0.8, multichannel=False, mode='reflect')
    gy = np.gradient(img, axis=0)
    gx = np.gradient(img, axis=1)

    Gxx = gx * gx
    Gxy = gx * gy
    Gyy = gy * gy

    Gxx = gaussian(Gxx, sigma, multichannel=False, mode='reflect')
    Gxy = gaussian(Gxy, sigma, multichannel=False, mode='reflect')
    Gyy = gaussian(Gyy, sigma, multichannel=False, mode='reflect')

    if mask is not None:
        Gxx[mask == 0] = 0
        Gxy[mask == 0] = 0
        Gyy[mask == 0] = 0
    dir_map = np.zeros((blkH, blkW)) - 10
    if block_size > 1:
        for i in range(blkH):
            for j in range(blkW):
                if mask[i * block_size + block_size // 2, j * block_size + block_size // 2] == 0:
                    continue
                blk_Gxx = np.sum(Gxx[i * block_size:(i+1) * block_size, j * block_size:(j+1) * block_size])
                blk_Gxy = np.sum(Gxy[i * block_size:(i+1) * block_size, j * block_size:(j+1) * block_size])
                blk_Gyy = np.sum(Gyy[i * block_size:(i+1) * block_size, j * block_size:(j+1) * block_size])
                dir_map[i, j] = 0.5 * np.arctan2(2 * blk_Gxy, blk_Gxx - blk_Gyy) + math.pi / 2
    else:
        dir_map = 0.5 * np.arctan2(2 * Gxy, Gxx - Gyy) + math.pi / 2
        quality_map = np.sqrt( (Gxx - Gyy) * (Gxx - Gyy) + 4*Gxy*Gxy) / (Gxx + Gyy + 500)
    return dir_map, quality_map


def get_ridge_flow_top(local_info):
    blkH, blkW = local_info.shape
    dir_map = np.zeros((blkH, blkW)) - 10
    fre_map = np.zeros((blkH, blkW)) - 10
    for i in range(blkH):
        for j in range(blkW):
            if local_info[i, j].ori is None:
                continue

            dir_map[i, j] = local_info[i, j].ori[0]
            fre_map[i, j] = local_info[i, j].fre[0]
    return dir_map, fre_map


def get_ridge_flow_optimal_N(local_info, N=2):
    blkH, blkW = local_info.shape
    dir_map = np.zeros((blkH, blkW)) - 10

    pad_size = 2
    neighbor_ind = []
    for i in range(-pad_size, pad_size + 1):
        for j in range(-pad_size, pad_size + 1):
            if i == 0 and j == 0:
                continue
            neighbor_ind.append((i, j))

    for i in range(pad_size, blkH - pad_size):
        for j in range(pad_size, blkW - pad_size):
            if local_info[i, j].ori is None:
                continue

            candi_num = len(local_info[i, j].ori)
            reliable = np.zeros((candi_num,))
            for k in range(candi_num):
                ori = local_info[i, j].ori[k]

                for ind in neighbor_ind:
                    ii = ind[0] + i
                    jj = ind[1] + j
                    if local_info[ii, jj].ori is None:
                        continue
                    ori_nb = np.asarray(local_info[ii, jj].ori)
                    diff = (ori - ori_nb) * 2
                    simi = (np.cos(diff) + 1) / 2
                    reliable[k] += np.max(simi)

            ind = np.argmax(reliable)
            dir_map[i, j] = local_info[i, j].ori[ind] + math.pi * 0.5
    return dir_map


def smooth_dir_map(dir_map, sigma=2.0, mask=None):

    cos2Theta = np.cos(dir_map * 2)
    sin2Theta = np.sin(dir_map * 2)
    if mask is not None:
        assert (dir_map.shape[0] == mask.shape[0])
        assert (dir_map.shape[1] == mask.shape[1])
        cos2Theta[mask == 0] = 0
        sin2Theta[mask == 0] = 0

    cos2Theta = gaussian(cos2Theta, sigma, multichannel=False, mode='reflect')
    sin2Theta = gaussian(sin2Theta, sigma, multichannel=False, mode='reflect')

    dir_map = np.arctan2(sin2Theta, cos2Theta) * 0.5

    return dir_map


def construct_dictionary(ori_num=30):
    ori_dict = []
    s = []
    for i in range(ori_num):
        ori_dict.append([])
        s.append([])

    patch_size2 = 16
    patch_size = 32
    dict_all = []
    spacing_all = []
    ori_all = []
    Y, X = np.meshgrid(range(-patch_size2, patch_size2), range(-patch_size2, patch_size2))

    for spacing in range(4, 13):
        for valley_spacing in range(max(2, spacing // 2 - 2), spacing // 2):
            ridge_spacing = spacing - valley_spacing
            for k in range(ori_num):
                theta = np.pi/2 - k*np.pi/ori_num
                X_r = X * np.cos(theta) - Y * np.sin(theta)
                for offset in range(0, spacing - 1, 2):
                    X_r_offset = X_r + offset + ridge_spacing / 2
                    X_r_offset = np.remainder(X_r_offset, spacing)
                    Y1 = np.zeros((patch_size, patch_size))
                    Y2 = np.zeros((patch_size, patch_size))
                    Y1[X_r_offset <= ridge_spacing] = X_r_offset[X_r_offset <= ridge_spacing]
                    Y2[X_r_offset > ridge_spacing] = X_r_offset[X_r_offset > ridge_spacing] - ridge_spacing
                    element = -np.sin(2*math.pi*(Y1 / ridge_spacing / 2)) + np.sin(2*math.pi*(Y2 / valley_spacing / 2))
                    element = element.reshape(patch_size * patch_size,)
                    element = element - np.mean(element)
                    element = element / np.linalg.norm(element)
                    ori_dict[k].append(element)
                    s[k].append(spacing)
                    dict_all.append(element)
                    spacing_all.append(1.0 / spacing)
                    ori_all.append(theta)
    for i in range(len(ori_dict)):
        ori_dict[i] = np.asarray(ori_dict[i])
        s[k] = np.asarray(s[k])
    dict_all = np.asarray(dict_all)
    dict_all = np.transpose(dict_all)
    spacing_all = np.asarray(spacing_all)
    ori_all = np.asarray(ori_all)

    return ori_dict, s, dict_all, ori_all, spacing_all


def construct_dictionary_rolled(ori_num=30):
    ori_dict = []
    s = []
    for i in range(ori_num):
        ori_dict.append([])
        s.append([])

    patch_size2 = 16
    patch_size = 32
    dict_all = []
    spacing_all = []
    ori_all = []
    Y, X = np.meshgrid(range(-patch_size2, patch_size2), range(-patch_size2, patch_size2))

    for spacing in range(6, 13):
        for valley_spacing in range(3, spacing//2):
            ridge_spacing = spacing - valley_spacing
            for k in range(ori_num):
                theta = np.pi/2 - k*np.pi/ori_num
                X_r = X * np.cos(theta) - Y * np.sin(theta)
                for offset in range(0, spacing - 1, 2):
                    X_r_offset = X_r + offset + ridge_spacing / 2
                    X_r_offset = np.remainder(X_r_offset, spacing)
                    Y1 = np.zeros((patch_size, patch_size))
                    Y2 = np.zeros((patch_size, patch_size))
                    Y1[X_r_offset <= ridge_spacing] = X_r_offset[X_r_offset <= ridge_spacing]
                    Y2[X_r_offset > ridge_spacing] = X_r_offset[X_r_offset > ridge_spacing] - ridge_spacing
                    element = -np.sin(2*math.pi*(Y1 / ridge_spacing / 2)) + np.sin(2*math.pi*(Y2 / valley_spacing / 2))
                    element = element.reshape(patch_size * patch_size,)
                    element = element - np.mean(element)
                    element = element / np.linalg.norm(element)
                    ori_dict[k].append(element)
                    s[k].append(spacing)
                    dict_all.append(element)
                    spacing_all.append(1.0 / spacing)
                    ori_all.append(theta)
    for i in range(len(ori_dict)):
        ori_dict[i] = np.asarray(ori_dict[i])
        s[k] = np.asarray(s[k])
    dict_all = np.asarray(dict_all)
    dict_all = np.transpose(dict_all)
    spacing_all = np.asarray(spacing_all)
    ori_all = np.asarray(ori_all)

    return ori_dict, s, dict_all, ori_all, spacing_all


def get_quality_map_intensity(img):
    img = img.astype(np.float32)
    nimg = preprocessing.local_constrast_enhancement(img)

    gy = np.gradient(nimg, axis=0)
    gx = np.gradient(nimg, axis=1)

    mag = np.abs(gx) + np.abs(gy)
    mag = cv2.GaussianBlur(mag, (21, 21), 7)

    mag = mag[::8, ::8]
    mask = mag > 0.3 * 127.5
    mask = mask.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.resize(mask, (0, 0), fx=8, fy=8)

    return mask


def get_quality_map_ori_dict(img, dict, spacing, dir_map=None, block_size=16):
    if img.dtype == 'uint8':
        img = img.astype(np.float)
    img = preprocessing.FastCartoonTexture(img)
    h, w = img.shape
    blkH, blkW = dir_map.shape

    quality_map = np.zeros((blkH, blkW), dtype=np.float)
    fre_map = np.zeros((blkH, blkW), dtype=np.float)
    ori_num = len(dict)
    dir_ind = dir_map * ori_num / math.pi
    dir_ind = dir_ind.astype(np.int)
    dir_ind = dir_ind % ori_num

    patch_size = np.sqrt(dict[0].shape[1])
    patch_size = patch_size.astype(np.int)
    pad_size = (patch_size - block_size) // 2
    img = np.lib.pad(img, (pad_size, pad_size), 'symmetric')
    for i in range(0, blkH):
        for j in range(0, blkW):
            ind = dir_ind[i, j]
            patch = img[i * block_size:i * block_size + patch_size, j * block_size:j * block_size + patch_size]

            patch = patch.reshape(patch_size * patch_size,)
            patch = patch - np.mean(patch)
            patch = patch / (np.linalg.norm(patch) + 0.0001)
            patch[patch > 0.05] = 0.05
            patch[patch < -0.05] = -0.05

            simi = np.dot(dict[ind], patch)
            similar_ind = np.argmax(abs(simi))
            quality_map[i, j] = np.max(abs(simi))
            fre_map[i, j] = 1. / spacing[ind][similar_ind]
    quality_map = gaussian(quality_map, sigma=2)
    return quality_map, fre_map


def SSIM(img, temp_img, block_size=16, thr=0.65):
    h, w = img.shape[:2]
    patch_size = 64
    blkH = h / block_size
    blkW = w / block_size
    blkH = int(blkH)
    blkW = int(blkW)
    quality = np.zeros((blkH, blkW))

    R = 500
    blocks_in_patch = int(patch_size / block_size)

    def get_weights(h, w, c, sigma=None):
        Y, X = np.mgrid[0:h, 0:w]
        x0 = w // 2
        y0 = h // 2
        if sigma is None:
            sigma = (np.max([h, w]) * 1. / 2) ** 2
        weight = np.exp(-((X - x0) * (X - x0) + (Y - y0) * (Y - y0)) / sigma)
        weight = np.stack((weight,) * c, axis=2)
        return weight

    weight = get_weights(blocks_in_patch, blocks_in_patch, 1, sigma=None)
    weight_pixel = get_weights(patch_size, patch_size, 1, sigma=(patch_size / 2)**2)
    for i in range(blkH - blocks_in_patch + 1):
        for j in range(blkW - blocks_in_patch + 1):
            patch = img[i * block_size:i * block_size + patch_size, j * block_size:j * block_size + patch_size]

            patch = patch - np.median(patch)
            patch = patch / (np.linalg.norm(patch) + R)
            patch = patch * weight_pixel[:, :, 0]
            patch = patch.reshape(patch_size * patch_size, )
            temp_patch = temp_img[i * block_size:i * block_size + patch_size,
                                  j * block_size:j * block_size + patch_size]
            temp_patch = temp_patch - np.median(temp_patch)
            temp_patch = temp_patch / (np.linalg.norm(temp_patch) + R)
            temp_patch = temp_patch * weight_pixel[:, :, 0]
            temp_patch = temp_patch.reshape(patch_size * patch_size, )

            simi = (np.dot(patch, temp_patch))
            quality[i:i + blocks_in_patch, j:j + blocks_in_patch] += simi * weight[:, :, 0]

    quality = cv2.GaussianBlur(quality, (5, 5), 0)
    blkmask = quality > thr
    blkmask = binary_closing(blkmask, np.ones((3, 3))).astype(np.int)
    blkmask = binary_opening(blkmask, np.ones((3, 3))).astype(np.int)

    return blkmask


def get_quality_map_dict(img, dict, ori, spacing, block_size=16, process=False, R=500.0, t=0.05):
    if img.dtype == 'uint8':
        img = img.astype(np.float)
    if process:
        img = preprocessing.FastCartoonTexture(img)
    h, w = img.shape

    blkH, blkW = h // block_size, w // block_size
    quality_map = np.zeros((blkH, blkW), dtype=np.float)
    dir_map = np.zeros((blkH, blkW), dtype=np.float)
    fre_map = np.zeros((blkH, blkW), dtype=np.float)

    patch_size = np.sqrt(dict.shape[0])
    patch_size = patch_size.astype(np.int)
    pad_size = (patch_size - block_size) // 2
    img = np.lib.pad(img, (pad_size, pad_size), 'symmetric')

    patches = []
    pixel_list = []

    r = 1

    for i in range(r, blkH - r):
        for j in range(r, blkW - r):
            pixel_list.append((i, j))
            patch = img[i * block_size:i * block_size + patch_size, j * block_size:j * block_size + patch_size].copy()

            patch = patch.reshape(patch_size * patch_size,)
            patch = patch - np.mean(patch)
            patch = patch / (np.linalg.norm(patch) + R)
            patch[patch > t] = 0.0
            patch[patch < -t] = -0.0
            patches.append(patch)

    patches = np.asarray(patches)
    simi = abs(np.dot(patches, dict))
    similar_ind = np.argmax(simi, axis=1)

    n = 0
    for i in range(r, blkH - r):
        for j in range(r, blkW - r):
            quality_map[i, j] = simi[n, similar_ind[n]]
            dir_map[i, j] = ori[similar_ind[n]]
            fre_map[i, j] = spacing[similar_ind[n]]
            n += 1

    for i in range(r):
        fre_map[i] = fre_map[r]
        dir_map[i] = dir_map[r]
        fre_map[-(r - i) - 1] = fre_map[-r - 1]
        dir_map[-(r - i) - 1] = fre_map[-r - 1]
        fre_map[:, i] = fre_map[:, r]
        dir_map[:, i] = dir_map[:, r]
        fre_map[:, -(r - i) - 1] = fre_map[:, -r - 1]
        dir_map[:, -(r - i) - 1] = fre_map[:, -r - 1]

    quality_map = cv2.GaussianBlur(quality_map, (5, 5), 0)
    dir_map = smooth_dir_map(dir_map, sigma=1.5)
    fre_map = cv2.GaussianBlur(fre_map, (3, 3), 1)
    return quality_map, dir_map, fre_map


def get_quality_map_dict_coarse(img, dict, ori, spacing, block_size=16, process=False, R=500.0, t=0.5):
    if img.dtype == 'uint8':
        img = img.astype(np.float)
    if process:
        img = preprocessing.FastCartoonTexture(img)
    h, w = img.shape

    blkH, blkW = h // block_size, w // block_size
    quality_map = np.zeros((blkH, blkW), dtype=np.float)
    dir_map = np.zeros((blkH, blkW), dtype=np.float)
    fre_map = np.zeros((blkH, blkW), dtype=np.float)

    patch_size = np.sqrt(dict.shape[0])
    patch_size = patch_size.astype(np.int)
    pad_size = (patch_size - block_size) // 2
    img = np.lib.pad(img, (pad_size, pad_size), 'symmetric')

    patches = []
    pixel_list = []

    r = 1
    x, y = np.meshgrid(range(-patch_size / 2, patch_size / 2), range(-patch_size / 2, patch_size / 2))
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    weight = np.exp(-(x * x + y * y) / (patch_size * patch_size / 3.0))
    # t = 0.02
    for i in range(r, blkH - r):
        for j in range(r, blkW - r):
            pixel_list.append((i, j))
            patch = img[i*block_size:i*block_size + patch_size, j*block_size:j*block_size + patch_size].copy()
            patch = patch - np.median(patch)
            patch = patch / (np.linalg.norm(patch) + R)
            patch[patch > t] = 0.0
            patch[patch < -t] = 0.

            patch = patch * weight
            patch = patch.reshape(patch_size * patch_size,)

            patches.append(patch)

    patches = np.asarray(patches)
    simi = abs(np.dot(patches, dict))
    similar_ind = np.argmax(simi, axis=1)

    n = 0
    # blks_in_patch = patch_size / block_size
    for i in range(r, blkH - r):
        for j in range(r, blkW - r):
            quality_map[i, j] = simi[n, similar_ind[n]]
            dir_map[i, j] = -math.atan2(ori[32 + 64, similar_ind[n]], ori[32, similar_ind[n]]) / 2.0
            fre_map[i, j] = spacing[32, similar_ind[n]]
            n += 1

    for i in range(r):
        fre_map[i] = fre_map[r]
        dir_map[i] = dir_map[r]
        fre_map[-(r - i) - 1] = fre_map[-r - 1]
        dir_map[-(r - i) - 1] = fre_map[-r - 1]
        fre_map[:, i] = fre_map[:, r]
        dir_map[:, i] = dir_map[:, r]
        fre_map[:, -(r - i) - 1] = fre_map[:, -r - 1]
        dir_map[:, -(r - i) - 1] = fre_map[:, -r - 1]

    quality_map = cv2.GaussianBlur(quality_map, (5, 5), 2)
    dir_map = smooth_dir_map(dir_map, sigma=1.5)
    fre_map = cv2.GaussianBlur(fre_map, (3, 3), 1)
    return quality_map, dir_map, fre_map


def get_maps_STFT(img, patch_size=64, block_size=16, preprocess=False):
    assert len(img.shape) == 2

    nrof_dirs = 16
    ovp_size = (patch_size - block_size) // 2
    if preprocess:
        img = preprocessing.FastCartoonTexture(img, sigma=2.5, show=False)
    h0, w0 = img.shape
    img = np.lib.pad(img, (ovp_size, ovp_size), 'symmetric')

    h, w = img.shape
    blkH = (h - patch_size) // block_size + 1
    blkW = (w - patch_size) // block_size + 1
    local_info = np.empty((blkH, blkW), dtype=object)

    x, y = np.meshgrid(range(-patch_size / 2, patch_size / 2), range(-patch_size / 2, patch_size / 2))
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    r = np.sqrt(x*x + y*y) + 0.0001

    # if preprocess:
    # -------------------------
    # Bandpass filter
    # -------------------------
    RMIN = 3  # min allowable ridge spacing
    RMAX = 18  # maximum allowable ridge spacing
    FLOW = patch_size / RMAX
    FHIGH = patch_size / RMIN
    dRLow = 1. / (1 + (r / FHIGH) ** 4)  # low pass     butterworth     filter
    dRHigh = 1. / (1 + (FLOW / r) ** 4)  # high    pass     butterworth     filter
    dBPass = dRLow * dRHigh  # bandpass

    dir = np.arctan2(y, x)
    dir[dir < 0] = dir[dir < 0] + math.pi
    dir_ind = np.floor(dir / (math.pi / nrof_dirs))
    dir_ind = dir_ind.astype(np.int, copy=False)
    dir_ind[dir_ind == nrof_dirs] = 0

    dir_ind_list = []
    for i in range(nrof_dirs):
        tmp = np.argwhere(dir_ind == i)
        dir_ind_list.append(tmp)

    sigma = patch_size / 3
    weight = np.exp(-(x*x + y*y) / (sigma*sigma))

    for i in range(0, blkH):
        for j in range(0, blkW):
            patch = img[i * block_size:i * block_size + patch_size, j * block_size:j * block_size + patch_size].copy()
            local_info[i, j] = local_STFT(patch, weight, dBPass)
            local_info[i, j].analysis(r, dir_ind_list)

    # get the ridge flow from the local information
    dir_map, fre_map = get_ridge_flow_top(local_info)
    dir_map = smooth_dir_map(dir_map)

    return dir_map, fre_map


def STFT_main(img, patch_size=64, block_size=16):
    assert len(img.shape) == 2

    rec_img, dir_map = STFT_enhancement(img, patch_size=64, block_size=16, preprocessing=True)

    rec_img2, dir_map = STFT_enhancement(rec_img, patch_size=64, block_size=16, preprocessing=False)

    return rec_img2, dir_map


def test_dir_map_STFT():
    img_path = '/Data/Latent/NISTSD27/image/'
    imgfiles = glob.glob(img_path + '*.bmp')
    imgfiles.sort()
    get_maps_STFT(img, patch_size=64, block_size=16, preprocess=True)


if __name__ == '__main__':
    imgfile = 'images/F0000001.bmp'
    img = scipy.ndimage.imread(imgfile)
    mask = get_quality_map_intensity(img)
    show.show_mask(mask, img=img)
