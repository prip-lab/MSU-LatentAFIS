import glob
import matplotlib.pylab as plt
import numpy as np
import math
import sys
import timeit
import cv2
import scipy.spatial.distance
import json
import get_maps
import preprocessing
import filtering
import descriptor
import template
import minutiae_AEC
import show
import enhancement_AEC
import os
from skimage import io
from timeit import default_timer as timer
from skimage.morphology import binary_opening, binary_closing
import argparse
import descriptor_PQ
import descriptor_DR
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

plt.switch_backend('agg')

config = {}


class FeatureExtraction_Latent:
    def __init__(self, patch_types=None, des_model_dirs=None, minu_model_dirs=None, enhancement_model_dir=None,
                 ROI_model_dir=None, coarsenet_dir=None, FineNet_dir=None):
        self.des_models = None
        self.patch_types = patch_types
        self.minu_model = None
        self.minu_model_dirs = minu_model_dirs
        self.des_model_dirs = des_model_dirs
        self.enhancement_model_dir = enhancement_model_dir
        self.ROI_model_dir = ROI_model_dir
        self.dict, self.spacing, self.dict_all, self.dict_ori, self.dict_spacing = get_maps.construct_dictionary(ori_num=60)
        print("Loading models, this may take some time...")

        if self.minu_model_dirs is not None:
            self.minu_model = []
            for i, minu_model_dir in enumerate(minu_model_dirs):
                print("Loading minutiae model (" + str(i+1) + " of " + str(len(minu_model_dirs)) + "): " + minu_model_dir)
                self.minu_model.append(minutiae_AEC.ImportGraph(minu_model_dir))

        self.coarsenet_dir = coarsenet_dir

        patchSize = 160
        oriNum = 64
        self.patchIndexV = descriptor.get_patch_index(patchSize, patchSize, oriNum, isMinu=1)

        if self.des_model_dirs is not None:
            self.des_models = []
            for i, model_dir in enumerate(des_model_dirs):
                print("Loading descriptor model (" + str(i+1) + " of " + str(len(des_model_dirs)) + "): " + model_dir)
                self.des_models.append(descriptor.ImportGraph(model_dir, input_name="inputs:0",
                                                              output_name='embedding:0'))
        if self.enhancement_model_dir is not None:
            print("Loading enhancement model: " + self.enhancement_model_dir)
            self.enhancement_model = enhancement_AEC.ImportGraph(enhancement_model_dir)
        print("Finished loading models.")

    def feature_extraction_single_latent(self, img_file, output_dir=None, ppi=500, show_processes=False,
                                         show_minutiae=False, minu_file=None):
        block = False
        block_size = 16
        img0 = io.imread(img_file, mode='L')  # / 255.0

        img = img0.copy()

        if ppi != 500:
            img = cv2.resize(img, (0, 0), fx=500.0 / ppi, fy=500.0 / ppi)
        img = preprocessing.adjust_image_size(img, block_size)
        name = os.path.basename(img_file)
        start = timer()
        h, w = img.shape

        if h > 1000 and w > 1000:
            return None, None

        # cropping using two dictionary based approach
        if minu_file is not None:
            manu_minu = np.loadtxt(minu_file)
            # #     # remove low quality minutiae points
            input_minu = np.array(manu_minu)
            input_minu[:, 2] = input_minu[:, 2] / 180.0 * np.pi
        else:
            input_minu = []

        descriptor_imgs = []
        texture_img = preprocessing.FastCartoonTexture(img, sigma=2.5, show=False)
        STFT_texture_img = preprocessing.STFT(texture_img)

        contrast_img_guassian = preprocessing.local_constrast_enhancement_gaussian(img)
        STFT_img = preprocessing.STFT(img)
        constrast_STFT_img = preprocessing.STFT(contrast_img_guassian)

        # step 1: enhance the latent based on our autoencoder
        AEC_img = self.enhancement_model.run_whole_image(STFT_texture_img)
        quality_map_AEC, dir_map_AEC, fre_map_AEC = get_maps.get_quality_map_dict(AEC_img, self.dict_all, self.dict_ori,
                                                                                  self.dict_spacing, R=500.0)
        blkmask_AEC = quality_map_AEC > 0.45
        blkmask_AEC = binary_closing(blkmask_AEC, np.ones((3, 3))).astype(np.int)
        blkmask_AEC = binary_opening(blkmask_AEC, np.ones((3, 3))).astype(np.int)
        blkmask_SSIM = get_maps.SSIM(STFT_texture_img, AEC_img, thr=0.2)
        blkmask = blkmask_SSIM * blkmask_AEC
        blkH, blkW = blkmask.shape
        mask = cv2.resize(blkmask.astype(float), (block_size * blkW, block_size * blkH), interpolation=cv2.INTER_LINEAR)
        mask[mask > 0] = 1

        minutiae_sets = []

        mnt_STFT = self.minu_model[0].run_whole_image(STFT_img, minu_thr=0.05)
        minutiae_sets.append(mnt_STFT)
        if show_minutiae:
            fname = output_dir + os.path.splitext(name)[0] + '_STFT_img.jpeg'
            show.show_minutiae_sets(STFT_img, [input_minu, mnt_STFT], mask=None, block=block, fname=fname)

        mnt_STFT = self.minu_model[0].run_whole_image(constrast_STFT_img, minu_thr=0.1)
        minutiae_sets.append(mnt_STFT)

        mnt_AEC = self.minu_model[1].run_whole_image(AEC_img, minu_thr=0.25)
        mnt_AEC = self.remove_spurious_minutiae(mnt_AEC, mask)
        minutiae_sets.append(mnt_AEC)
        if show_minutiae:
            fname = output_dir + os.path.splitext(name)[0] + '_AEC_img.jpeg'
            show.show_minutiae_sets(AEC_img, [input_minu, mnt_AEC], mask=mask, block=block, fname=fname)

        enh_contrast_img = filtering.gabor_filtering_pixel2(contrast_img_guassian, dir_map_AEC + math.pi / 2,
                                                            fre_map_AEC, mask=np.ones((h, w)), block_size=16,
                                                            angle_inc=3)
        mnt_contrast = self.minu_model[1].run_whole_image(enh_contrast_img, minu_thr=0.25)
        mnt_contrast = self.remove_spurious_minutiae(mnt_contrast, mask)
        minutiae_sets.append(mnt_contrast)

        enh_texture_img = filtering.gabor_filtering_pixel2(texture_img, dir_map_AEC + math.pi / 2,
                                                           fre_map_AEC,
                                                           mask=np.ones((h, w)), block_size=16, angle_inc=3)

        mnt_texture = self.minu_model[1].run_whole_image(enh_texture_img, minu_thr=0.25)
        mnt_texture = self.remove_spurious_minutiae(mnt_texture, mask)
        minutiae_sets.append(mnt_texture)

        h, w = img.shape
        latent_template = template.Template()

        # template set 1: no ROI and enhancement are required
        # texture image is used for coase segmentation
        descriptor_imgs = []

        descriptor_imgs.append(STFT_img)
        descriptor_imgs.append(texture_img)
        descriptor_imgs.append(enh_texture_img)
        descriptor_imgs.append(enh_contrast_img)

        mnt2 = self.get_common_minutiae(minutiae_sets, thr=2)

        mnt3 = self.get_common_minutiae(minutiae_sets, thr=3)

        minutiae_sets.append(mnt3)
        minutiae_sets.append(mnt2)
        if show_minutiae:
            fname = output_dir + os.path.splitext(name)[0] + '_common_2.jpeg'
            show.show_minutiae_sets(img, [input_minu, mnt2], mask=mask, block=block, fname=fname)
        end = timer()
        print('Time for minutiae extraction: %f' % (end - start))

        start = timer()
        for mnt in minutiae_sets:
            for des_img in descriptor_imgs:
                des = descriptor.minutiae_descriptor_extraction(des_img, mnt, self.patch_types, self.des_models,
                                                                self.patchIndexV, batch_size=128)
                minu_template = template.MinuTemplate(h=h, w=w, blkH=blkH, blkW=blkW, minutiae=mnt,
                                                      des=des, oimg=dir_map_AEC, mask=mask)
                latent_template.add_minu_template(minu_template)
        end = timer()
        print('Time for minutiae descriptor generation: %f' % (end - start))

        start = timer()
        # texture templates
        stride = 16
        x = np.arange(24, w - 24, stride)
        y = np.arange(24, h - 24, stride)

        virtual_minutiae = []
        distFromBg = scipy.ndimage.morphology.distance_transform_edt(mask)
        for y_i in y:
            for x_i in x:
                if (distFromBg[y_i][x_i] <= 16):
                    continue
                ofY = int(y_i / 16)
                ofX = int(x_i / 16)

                ori = -dir_map_AEC[ofY][ofX]
                virtual_minutiae.append([x_i, y_i, ori])
                virtual_minutiae.append([x_i, y_i, math.pi + ori])
        virtual_minutiae = np.asarray(virtual_minutiae)

        texture_template = []
        if len(virtual_minutiae) > 3:
            virtual_des = descriptor.minutiae_descriptor_extraction(enh_contrast_img, virtual_minutiae,
                                                                    self.patch_types, self.des_models, self.patchIndexV,
                                                                    batch_size=128, patch_size=96)

            texture_template = template.TextureTemplate(h=h, w=w, minutiae=virtual_minutiae, des=virtual_des, mask=None)
            latent_template.add_texture_template(texture_template)

        end = timer()

        print('Time for texture template generation: %f' % (end - start))
        return latent_template, texture_template

    def get_common_minutiae(self, minutiae_sets, thr=3):

        nrof_minutiae_sets = len(minutiae_sets)

        init_ind = 3
        if len(minutiae_sets[init_ind]) == 0:
            return []
        mnt = list(minutiae_sets[init_ind][:, :4])
        count = list(np.ones(len(mnt),))
        for i in range(0, nrof_minutiae_sets):
            if i == init_ind:
                continue
            for j in range(len(minutiae_sets[i])):
                x2 = minutiae_sets[i][j, 0]
                y2 = minutiae_sets[i][j, 1]
                ori2 = minutiae_sets[i][j, 2]
                found = False
                for k in range(len(mnt)):
                    x1 = mnt[k][0]
                    y1 = mnt[k][1]
                    ori1 = mnt[k][2]
                    dist = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))

                    ori_dist = math.fabs(ori1 - ori2)
                    if ori_dist > math.pi / 2:
                        ori_dist = math.pi - ori_dist
                    if dist <= 10 and ori_dist < math.pi / 6:
                        count[k] += 1
                        found = True
                        break
                if not found:
                    mnt.append([x2, y2, ori2, 1])
                    count.append(1)
        count = np.asarray(count)
        ind = np.where(count >= thr)[0]
        mnt = np.asarray(mnt)
        mnt = mnt[ind, :]
        mnt[:, 3] = 1
        return mnt

    def remove_spurious_minutiae(self, mnt, mask):

        minu_num = len(mnt)
        if minu_num <= 0:
            return mnt
        flag = np.ones((minu_num,), np.uint8)
        h, w = mask.shape[:2]
        R = 10
        for i in range(minu_num):
            x = mnt[i, 0]
            y = mnt[i, 1]
            x = np.int(x)
            y = np.int(y)
            if x < R or y < R or x > w - R - 1 or y > h - R - 1:
                flag[i] = 0
            elif (mask[y - R, x - R] == 0 or mask[y - R, x + R] == 0 or mask[y + R, x - R] == 0 or
                  mask[y + R, x + R] == 0):
                flag[i] = 0
        mnt = mnt[flag > 0, :]
        return mnt

    def feature_extraction(self, image_dir, template_dir=None, minu_path=None, N1=0, N2=258):

        img_files = glob.glob(image_dir + '*.bmp')
        assert(len(img_files) > 0)

        if not os.path.exists(template_dir):
            os.makedirs(template_dir)

        img_files.sort()
        if minu_path is not None:
            minu_files = glob.glob(minu_path + '*.txt')

            minu_files.sort(key=lambda filename: int(''.join(filter(str.isdigit, filename))))
        for i, img_file in enumerate(img_files):
            print i, img_file
            img_name = os.path.basename(img_file)
            if template_dir is not None:
                fname = template_dir + os.path.splitext(img_name)[0] + '.dat'
                if os.path.exists(fname):
                    continue

            start = timeit.default_timer()
            if minu_path is not None and len(minu_files) > i:
                latent_template, texture_template = self.feature_extraction_single_latent(img_file,
                                                                                          output_dir=template_dir,
                                                                                          show_processes=False,
                                                                                          minu_file=minu_files[i],
                                                                                          show_minutiae=False)
            else:
                latent_template, texture_template = self.feature_extraction_single_latent(img_file,
                                                                                          output_dir=template_dir,
                                                                                          show_processes=False,
                                                                                          show_minutiae=False)

            stop = timeit.default_timer()
            print stop - start

            fname = template_dir + os.path.splitext(img_name)[0] + '.dat'
            template.Template2Bin_Byte_TF_C(fname, latent_template, isLatent=True)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0')
    parser.add_argument('--N1', type=int, help='rolled index from which the enrollment starts', default=0)
    parser.add_argument('--N2', type=int, help='rolled index from which the enrollment starts', default=100)
    parser.add_argument('--tdir', type=str, help='Path to location where extracted templates should be stored')
    parser.add_argument('--idir', type=str, help='Path to directory containing input images')
    parser.add_argument('--i', type=str, help='Path to single input image')
    return parser.parse_args(argv)


def main(image_dir, template_dir):
    global config
    if not config:
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        with open(dir_path + 'config') as config_file:
            config = json.load(config_file)

    des_model_dirs = []
    patch_types = []
    model_dir = config['DescriptorModelPatch2']
    des_model_dirs.append(model_dir)
    patch_types.append(2)
    model_dir = config['DescriptorModelPatch8']
    des_model_dirs.append(model_dir)
    patch_types.append(8)
    model_dir = config['DescriptorModelPatch11']
    des_model_dirs.append(model_dir)
    patch_types.append(11)

    # minutiae extraction model
    minu_model_dirs = []
    minu_model_dirs.append(config['MinutiaeExtractionModelLatentSTFT'])
    minu_model_dirs.append(config['MinutiaeExtractionModel'])

    # enhancement model
    enhancement_model_dir = config['EnhancementModel']

    LF_Latent = FeatureExtraction_Latent(patch_types=patch_types, des_model_dirs=des_model_dirs,
                                         enhancement_model_dir=enhancement_model_dir, minu_model_dirs=minu_model_dirs)
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    print("Starting feature extraction (batch)...")
    LF_Latent.feature_extraction(image_dir=image_dir, template_dir=template_dir, minu_path=config['MinuPath'])


def main_single_image(image_file, template_dir):
    global config
    if not config:
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        with open(dir_path + 'config') as config_file:
            config = json.load(config_file)

    des_model_dirs = []
    patch_types = []
    model_dir = config['DescriptorModelPatch2']
    des_model_dirs.append(model_dir)
    patch_types.append(2)
    model_dir = config['DescriptorModelPatch8']
    des_model_dirs.append(model_dir)
    patch_types.append(8)
    model_dir = config['DescriptorModelPatch11']
    des_model_dirs.append(model_dir)
    patch_types.append(11)

    # minutiae extraction model
    minu_model_dirs = []
    minu_model_dirs.append(config['MinutiaeExtractionModelLatentSTFT'])
    minu_model_dirs.append(config['MinutiaeExtractionModel'])

    # enhancement model
    enhancement_model_dir = config['EnhancementModel']

    LF_Latent = FeatureExtraction_Latent(patch_types=patch_types, des_model_dirs=des_model_dirs,
                                         enhancement_model_dir=enhancement_model_dir, minu_model_dirs=minu_model_dirs)

    # minu_path = None
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    print("Latent query: " + image_file)
    print("Starting feature extraction (single latent)...")
    latent_template, texture_template = LF_Latent.feature_extraction_single_latent(image_file, output_dir=template_dir,
                                                                                   show_processes=False, minu_file=None,
                                                                                   show_minutiae=False)
    fname = template_dir + os.path.splitext(os.path.basename(image_file))[0] + '.dat'
    template.Template2Bin_Byte_TF_C(fname, latent_template, isLatent=True)

    return fname


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])

    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    with open(dir_path + '/afis.config') as config_file:
        config = json.load(config_file)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.i:
        template_dir = args.tdir if args.tdir else config['LatentTemplateDirectory']
        template_fname = main_single_image(args.i, template_dir)
        print("Finished feature extraction. Starting dimensionality reduction...")
        descriptor_DR.template_compression_single(input_file=template_fname, output_dir=template_dir,
                                                  model_path=config['DimensionalityReductionModel'],
                                                  isLatent=True, config=None)
        print("Finished dimensionality reduction. Starting product quantization...")
        descriptor_PQ.encode_PQ_single(input_file=template_fname, output_dir=template_dir, fprint_type='latent')
        print("Finished product quantization. Exiting...")
    elif(args.idir):
        template_dir = args.tdir if args.tdir else config['LatentTemplateDirectory']
        main(args.idir, template_dir)
        print("Finished feature extraction. Starting dimensionality reduction...")
        descriptor_DR.template_compression(input_dir=template_dir, output_dir=template_dir,
                                           model_path=config['DimensionalityReductionModel'],
                                           isLatent=True, config=None)
        print("Finished dimensionality reduction. Starting product quantization...")
        descriptor_PQ.encode_PQ(input_dir=template_dir, output_dir=template_dir, fprint_type='latent')
        print ("Finished product quantization. Exiting...")
    else:
        print("Using arguments from config file.")
        template_dir = args.tdir if args.tdir else config['LatentTemplateDirectory']
        image_dir = args.idir if args.idir else config['LatentImageDirectory']
        main(image_dir=image_dir, template_dir=template_dir)
        print("Finished feature extraction. Starting dimensionality reduction...")
        descriptor_DR.template_compression(input_dir=template_dir, output_dir=template_dir,
                                           model_path=config['DimensionalityReductionModel'],
                                           isLatent=True, config=None)
        print("Finished dimensionality reduction. Starting product quantization...")
        descriptor_PQ.encode_PQ(input_dir=template_dir, output_dir=template_dir, fprint_type='latent')
        print("Finished product quantization. Exiting...")
