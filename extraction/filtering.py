import matplotlib.pylab as plt
import scipy.ndimage
from skimage.filters import gabor_kernel
import math
import numpy as np


def gabor_filtering_block(img, dir_map, fre_map, mask=None, patch_size=64, block_size=16):
    img = img.astype(np.float)
    ovp_size = (patch_size - block_size) // 2
    h0, w0 = img.shape
    if mask is None:
        mask = np.ones((h0, w0), dtype=np.int)
    img = np.lib.pad(img, (ovp_size, ovp_size), 'symmetric')

    x, y = np.meshgrid(range(-patch_size / 2, patch_size / 2), range(-patch_size / 2, patch_size / 2))
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    sigma = patch_size / 3

    weight = np.exp(-(x * x + y * y) / (sigma * sigma))

    h, w = img.shape
    blkH = (h - patch_size) // block_size + 1
    blkW = (w - patch_size) // block_size + 1
    rec_img = np.zeros((h, w))
    for i in range(0, blkH):
        for j in range(0, blkW):
            if mask[i * block_size + block_size // 2, j * block_size + block_size // 2] == 0:
                continue
            patch = img[i * block_size:i * block_size + patch_size, j * block_size:j * block_size + patch_size].copy()
            patch -= np.mean(patch)
            patch_FFT = np.fft.fft2(patch)
            if fre_map[i, j] < 0:
                continue
            kernel = gabor_kernel(fre_map[i, j], theta=dir_map[i, j], sigma_x=4, sigma_y=4)
            f = kernel.real
            f = f - np.mean(f)
            f = f / (np.linalg.norm(f) + 0.00001)
            kernel_FFT = np.fft.fft2(f, (patch_size, patch_size))
            patch_FFT = patch_FFT * kernel_FFT
            rec_patch = np.real(np.fft.ifft2(patch_FFT))
            rec_patch = np.fft.ifftshift(rec_patch)
            plt.subplot(121), plt.imshow(patch, cmap='gray')
            plt.title('Input mage'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(rec_patch, cmap='gray')
            plt.title('Enhanced image'), plt.xticks([]), plt.yticks([])
            plt.show(block=True)
            plt.close()
            rec_patch = rec_patch * weight

            rec_img[i * block_size:i * block_size + patch_size, j * block_size:j * block_size + patch_size] += rec_patch

    rec_img = rec_img[ovp_size:ovp_size + h0, ovp_size:ovp_size + w0]

    rec_img = (rec_img - np.min(rec_img)) / (np.max(rec_img) - np.min(rec_img)) * 255

    return rec_img


def get_gabor_filters(angle_inc=3, fre_num=30):
    ori_num = 180 // angle_inc
    gaborfilter = np.zeros((ori_num, fre_num), dtype=object)
    for i in range(ori_num):
        ori = i * angle_inc / 180.0 * math.pi
        for j in range(fre_num):
            if j < 5:
                continue

            kernel = gabor_kernel(j * 0.01, theta=ori, sigma_x=3, sigma_y=3)
            kernel = kernel.real

            kernel = kernel - np.mean(kernel)
            norm = np.linalg.norm(kernel)

            kernel = kernel / (norm + 0.00001)
            kernel = kernel.real * 255
            t = np.asarray(kernel, np.int16)
            gaborfilter[i, j] = t

    return gaborfilter


def gabor_filtering_pixel(img, dir_map, fre_map, mask=None, block_size=16, angle_inc=3, gabor_filters=None):
    # block_size is the block size of the dir_map and fre_map
    h, w = img.shape
    if mask is None:
        mask = np.ones((h, w), dtype=np.uint8)
    if block_size > 1:
        cos2Theta = np.cos(dir_map * 2)
        sin2Theta = np.sin(dir_map * 2)
        cos2Theta = scipy.ndimage.interpolation.zoom(cos2Theta, block_size)
        sin2Theta = scipy.ndimage.interpolation.zoom(sin2Theta, block_size)
        frequency = scipy.ndimage.interpolation.zoom(fre_map, block_size)
        angle = np.arctan2(sin2Theta, cos2Theta) * 0.5
    else:
        angle = dir_map
        frequency = fre_map
    angle = angle / math.pi * 180
    angle = angle.astype(int)
    angle[angle < 0] = angle[angle < 0] + 180
    angle[angle == 180] = 0
    angle_ind = angle // angle_inc
    frequency_ind = np.around(frequency * 100).astype(int)
    # get gabor filters
    if gabor_filters is None:
        gabor_filters = get_gabor_filters()

    img = img.astype(np.int32)
    h, w = img.shape
    enh_img = np.zeros((h, w), dtype=np.int32)

    sh = 10
    sw = 10
    for i in xrange(10, h-10-1):
        for j in xrange(10, w-10-1):
            if mask[i, j] == 0:
                continue

            frequency_ind_b = frequency_ind[i, j]
            angle_ind_b = angle_ind[i, j]

            if frequency_ind_b < 5 or frequency_ind_b >= 30:
                continue
            kernel = gabor_filters[angle_ind_b, frequency_ind_b]
            kh, kw = kernel.shape
            sh = kh // 2
            sw = kw // 2

            enh_img[i, j] = np.sum(img[i - sh:i + sh + 1, j - sw:j + sw + 1] * kernel)

    enh_img = (enh_img - np.min(enh_img)) / (np.max(enh_img) - np.min(enh_img) + 0.0001) * 255
    return enh_img


def gabor_filtering_pixel2(img, dir_map, fre_map, mask=None, block_size=16, angle_inc=3, gabor_filters=None):
    # block_size is the block size of the dir_map and fre_map
    h, w = img.shape
    if mask is None:
        mask = np.ones((h, w), dtype=np.uint8)

    if block_size > 1:
        cos2Theta = np.cos(dir_map * 2)
        sin2Theta = np.sin(dir_map * 2)
        cos2Theta = scipy.ndimage.interpolation.zoom(cos2Theta, block_size)
        sin2Theta = scipy.ndimage.interpolation.zoom(sin2Theta, block_size)
        frequency = scipy.ndimage.interpolation.zoom(fre_map, block_size)
        angle = np.arctan2(sin2Theta, cos2Theta) * 0.5
    else:
        angle = dir_map
        frequency = fre_map

    angle = angle / math.pi * 180
    angle = angle.astype(int)
    angle[angle < 0] = angle[angle < 0] + 180
    angle[angle == 180] = 0
    angle_ind = angle // angle_inc
    frequency_ind = np.around(frequency * 100).astype(int)
    # get gabor filters
    if gabor_filters is None:
        gabor_filters = get_gabor_filters()

    img = img.astype(np.int32)
    h, w = img.shape
    enh_img = np.zeros((h, w), dtype=np.int32)

    mask[:15, :] = 0
    mask[:, :15] = 0
    mask[h-15:h, :] = 0
    mask[:, w-15:w] = 0

    candi_ind = np.where(mask > 0)

    candi_num = len(candi_ind[0])

    def apply_filter(src_arr, dst_arr, gabor_filters, idx_list, frequency_ind, angle_ind):
        for p, i, j in idx_list:
            frequency_ind_b = frequency_ind[i, j]
            if frequency_ind_b < 5 or frequency_ind_b >= 30:
                continue
            angle_ind_b = angle_ind[i, j]
            kernel = gabor_filters[angle_ind_b, frequency_ind_b]
            kh, kw = kernel.shape
            sh = kh >> 1
            sw = kw >> 1
            dst_arr[p] = np.sum(src_arr[i - sh:i + sh + 1, j - sw:j + sw + 1] * kernel)
        return

    from multiprocessing import Process, Array
    threads = []
    thread_num = 1
    pixels_per_thread = candi_num // thread_num
    result_array = Array('f', candi_num)

    for k in xrange(0, thread_num):
        idx_list = []
        for n in xrange(0, pixels_per_thread):
            p = k * pixels_per_thread + n
            if p >= candi_num:
                break
            idx_list.append((p, candi_ind[0][p], candi_ind[1][p]))
        t = Process(target=apply_filter, args=(img, result_array, gabor_filters, idx_list, frequency_ind, angle_ind))
        t.daemon = True
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    for k in xrange(candi_num):
        i = candi_ind[0][k]
        j = candi_ind[1][k]
        enh_img[i, j] = result_array[k]

    enh_img = (enh_img - np.min(enh_img) + 0.0001) / (np.max(enh_img) - np.min(enh_img) + 0.0001) * 255
    return enh_img


if __name__ == '__main__':
    get_gabor_filters(angle_inc=3, fre_num=30)
