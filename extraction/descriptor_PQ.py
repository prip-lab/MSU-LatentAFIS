import struct
import sys
from scipy.cluster.vq import vq, kmeans2
import os
import glob
import numpy as np
from random import shuffle
import template_2 as template
import argparse
import json


class TrainedPQEncoder(object):
    def __init__(self, codewords, code_dtype):
        # type: (np.array, type) -> None
        self.codewords, self.code_dtype = codewords, code_dtype
        self.M, _, self.Ds = codewords.shape

    def encode_multi(self, data_matrix):
        data_matrix = np.array(data_matrix)
        N, D = data_matrix.shape
        assert self.Ds * self.M == D, "input dimension must be Ds * M"

        codes = np.empty((N, self.M), dtype=self.code_dtype)
        for m in range(self.M):
            codes[:, m], _ = vq(data_matrix[:, m * self.Ds: (m + 1) * self.Ds], self.codewords[m])
        return codes

    def decode_multi(self, codes):
        codes = np.array(codes)
        N, M = codes.shape
        assert M == self.M
        assert codes.dtype == self.code_dtype

        decoded = np.empty((N, self.Ds * self.M), dtype=np.float)
        for m in range(self.M):
            decoded[:, m * self.Ds: (m + 1) * self.Ds] = self.codewords[m][codes[:, m], :]
        return decoded


def training(rolled_template_path, subdim=8, Ks=256):
    rolled_template_files = glob.glob(rolled_template_path + '*.dat')

    rolled_template_files.sort(key=lambda filename: int(''.join(filter(str.isdigit, filename.encode("utf-8")))))

    des = None
    for i in range(259, 1000):
        rolled_template = template.Bin2Template_Byte_TF_C(rolled_template_files[i], isLatent=False)
        if len(rolled_template.texture_template) < 1:
            continue
        one_des = rolled_template.texture_template[0].des

        if len(one_des) < 100:
            continue
        ind = np.arange(len(one_des))
        shuffle(ind)

        one_des = one_des[::2, :]
        if des is None:
            des = one_des
        else:
            des = np.concatenate((des, one_des), axis=0)

    N, D = des.shape
    assert Ks < N, "the number of training vector should be more than Ks"
    assert D % subdim == 0, "   input dimension must be dividable by nunm_subdim"
    nrof_subdim = int(D / subdim)

    codewords = np.zeros((nrof_subdim, Ks, subdim), dtype=np.float)
    iteration = 20
    for m in range(nrof_subdim):
        des_sub = des[:, m * subdim: (m + 1) * subdim].astype(np.float)
        codewords[m], _ = kmeans2(des_sub, Ks, iter=iteration, minit='points')

    code_dtype = np.uint8 if Ks <= 2 ** 8 else (np.uint16 if Ks <= 2 ** 16 else np.uint32)
    codebook = TrainedPQEncoder(codewords, code_dtype)
    return codebook


def Template2Bin_Byte_latent(outfile, version=1, T=None):
    # version = 1  # template version
    Max_BlkSize = 50  # maximum block size

    Max_Nrof_minutiae = 2000
    with open(outfile, 'wb') as file:

        # file header, preserved for identifier information
        tmp = np.zeros((12,), dtype=np.int16)
        tmp[0] = version
        tmp = tuple(tmp)
        file.write(struct.pack('H' * 12, *tmp))
        if template is None or len(T.minu_template) == 0:
            tmp = (0, 0, 0, 0)
            file.write(struct.pack('H' * 4, *tmp))
            return

        blkH = T.minu_template[0].blkH
        if blkH > Max_BlkSize:
            blkH = Max_BlkSize
        blkW = T.minu_template[0].blkW
        if blkW > Max_BlkSize:
            blkW = Max_BlkSize

        tmp = (T.minu_template[0].h, T.minu_template[0].w, blkH, blkW)

        file.write(struct.pack('H' * 4, *tmp))
        num_minu_template = len(T.minu_template)
        tmp = (num_minu_template,)
        file.write(struct.pack('B', *tmp))

        for i in range(num_minu_template):
            if len(T.minu_template[i].minutiae) > Max_Nrof_minutiae:
                T.minu_template[i].minutiae = T.minu_template[i].minutiae[:Max_Nrof_minutiae]
            minu_num = len(T.minu_template[i].minutiae)
            tmp = (minu_num,)
            file.write(struct.pack('H', *tmp))
            if minu_num <= 0:
                continue
            x = T.minu_template[i].minutiae[:, 0]
            x = tuple(np.uint16(x))
            file.write(struct.pack('H' * minu_num, *x))
            y = T.minu_template[i].minutiae[:, 1]
            y = tuple(np.uint16(y))
            file.write(struct.pack('H' * minu_num, *y))
            # orientation
            ori = T.minu_template[i].minutiae[:, 2]
            ori = tuple(np.float32(ori))
            file.write(struct.pack('f' * minu_num, *ori))

            des = T.minu_template[i].des
            des_len = des.shape[1]

            tmp = (des_len,)
            file.write(struct.pack('H', *tmp))
            descriptor = np.reshape(des, (des_len * minu_num,))
            descriptor_tuple = tuple(np.float32(descriptor))
            file.write(struct.pack('f' * des_len * minu_num, *(descriptor_tuple)))

        num_texture_template = len(T.texture_template)
        tmp = (num_texture_template,)
        file.write(struct.pack('B', *tmp))
        if num_texture_template == 0:
            return
        for i in range(num_texture_template):
            if len(T.texture_template[i].minutiae) > Max_Nrof_minutiae:
                T.texture_template[i].minutiae = T.texture_template[i].minutiae[:Max_Nrof_minutiae]
            minu_num = len(T.texture_template[i].minutiae)
            tmp = (minu_num,)
            file.write(struct.pack('H', *tmp))
            if minu_num > 0:
                x = T.texture_template[i].minutiae[:, 0]
                x = (x - 24) / 16
                x = tuple(np.uint16(x))
                file.write(struct.pack('H' * minu_num, *x))
                y = T.texture_template[i].minutiae[:, 1]
                y = (y - 24) / 16
                y = tuple(np.uint16(y))
                file.write(struct.pack('H' * minu_num, *y))
                # orientation
                ori = T.texture_template[i].minutiae[:, 2]
                ori = tuple(np.float32(ori))
                file.write(struct.pack('f' * minu_num, *ori))

                des = T.texture_template[i].des
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


def Template2Bin_Byte_PQ_rolled(outfile, version=1, T=None):
    # version = 1  # template version
    Max_BlkSize = 50  # maximum block size

    Max_Nrof_minutiae = 2000
    with open(outfile, 'wb') as file:

        # file header, preserved for identifier information
        tmp = np.zeros((12,), dtype=np.int16)
        tmp[0] = version
        tmp = tuple(tmp)
        file.write(struct.pack('H' * 12, *tmp))
        if T is None or len(T.minu_template) == 0:
            tmp = (0, 0, 0, 0)
            file.write(struct.pack('H' * 4, *tmp))
            return

        blkH = T.minu_template[0].blkH
        if blkH > Max_BlkSize:
            blkH = Max_BlkSize
        blkW = T.minu_template[0].blkW
        if blkW > Max_BlkSize:
            blkW = Max_BlkSize

        tmp = (T.minu_template[0].h, T.minu_template[0].w, blkH, blkW)

        file.write(struct.pack('H' * 4, *tmp))
        num_minu_template = len(T.minu_template)
        tmp = (num_minu_template,)
        file.write(struct.pack('B', *tmp))

        for i in range(num_minu_template):
            if len(T.minu_template[i].minutiae) > Max_Nrof_minutiae:
                T.minu_template[i].minutiae = T.minu_template[i].minutiae[:Max_Nrof_minutiae]
            minu_num = len(T.minu_template[i].minutiae)
            tmp = (minu_num,)
            file.write(struct.pack('H', *tmp))
            if minu_num <= 0:
                continue
            x = T.minu_template[i].minutiae[:, 0]
            x = tuple(np.uint16(x))
            file.write(struct.pack('H' * minu_num, *x))
            y = T.minu_template[i].minutiae[:, 1]
            y = tuple(np.uint16(y))
            file.write(struct.pack('H' * minu_num, *y))
            # orientation
            ori = T.minu_template[i].minutiae[:, 2]
            ori = tuple(np.float32(ori))
            file.write(struct.pack('f' * minu_num, *ori))

            des = T.minu_template[i].des
            des_len = des.shape[1]

            tmp = (des_len,)
            file.write(struct.pack('H', *tmp))
            descriptor = np.reshape(des, (des_len * minu_num,))
            descriptor_tuple = tuple(np.float32(descriptor))
            file.write(struct.pack('f' * des_len * minu_num, *(descriptor_tuple)))

        num_texture_template = len(T.texture_template)
        tmp = (num_texture_template,)
        file.write(struct.pack('B', *tmp))
        if num_texture_template == 0:
            return
        for i in range(num_texture_template):
            if len(T.texture_template[i].minutiae) > Max_Nrof_minutiae:
                T.texture_template[i].minutiae = T.texture_template[i].minutiae[:Max_Nrof_minutiae]
            minu_num = len(T.texture_template[i].minutiae)
            tmp = (minu_num,)
            file.write(struct.pack('H', *tmp))
            if minu_num > 0:
                x = T.texture_template[i].minutiae[:, 0]
                x = (x - 24) / 16
                x = tuple(np.uint16(x))
                file.write(struct.pack('H' * minu_num, *x))
                y = T.texture_template[i].minutiae[:, 1]
                y = (y - 24) / 16
                y = tuple(np.uint16(y))
                file.write(struct.pack('H' * minu_num, *y))
                # orientation
                ori = T.texture_template[i].minutiae[:, 2]
                ori = tuple(np.float32(ori))
                file.write(struct.pack('f' * minu_num, *ori))

                codes = T.texture_template[i].des
                des_len = len(codes[0])
                tmp = (des_len,)
                file.write(struct.pack('H', *tmp))

                # pdb.set_trace()
                if len(codes) > Max_Nrof_minutiae:
                    codes = codes[:Max_Nrof_minutiae]
                descriptor = np.reshape(codes, (des_len * minu_num,))
                descriptor_tuple = tuple(np.uint8(descriptor))
                file.write(struct.pack('B' * des_len * minu_num, *(descriptor_tuple)))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--fprint_type', type=str, help='Type of fingerprint templates (latent or rolled)',
                        default='latent')
    parser.add_argument('--output_dir', type=str, help='data path for minutiae descriptor and minutiae extraction')
    parser.add_argument('--input_dir', type=str, help='data path for images')
    parser.add_argument('--input_file', type=str, help='data path for images')
    return parser.parse_args(argv)


def encode_PQ(input_dir, output_dir, fprint_type):
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    with open(dir_path + '/afis.config') as config_file:
        config = json.load(config_file)
    embedding_size = 96
    stride = 16
    subdim = 6
    istraining = False
    isLatent = True if (fprint_type).lower() == 'latent' else False
    if isLatent:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        latent_template_files = glob.glob(input_dir + '*.dat')
        latent_template_files.sort()

        n = 0
        for i, file in enumerate(latent_template_files):
            print("PQ: " + file)
            latent_template = template.Bin2Template_Byte_TF_C(latent_template_files[i], isLatent=True)

            outfile = output_dir + os.path.basename(file).split('.')[0] + '.dat'

            Template2Bin_Byte_latent(outfile, version=1, T=latent_template)
    else:
        code_file = config["CodebookPath"]
        if istraining:
            print(istraining)
        else:

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # load codebook
            with open(code_file, 'rb') as file:
                tmp = struct.unpack('H' * 3, file.read(6))
                nrof_subs, nrof_clusters, sub_dim = tmp
                codewords = np.zeros((nrof_subs, nrof_clusters, sub_dim), dtype=np.float32)
                for i in range(nrof_subs):
                    for j in range(nrof_clusters):
                        tmp = struct.unpack('f' * sub_dim, file.read(4 * sub_dim))
                        codewords[i, j, :] = np.array(list(tmp))

            code_dtype = np.uint8 if nrof_clusters <= 2 ** 8 else (np.uint16 if nrof_clusters <= 2 ** 16 else np.uint32)
            PQEncoder = TrainedPQEncoder(codewords, code_dtype)

            # for rolled
            rolled_template_files = glob.glob(input_dir + '*.dat')
            rolled_template_files.sort(key=lambda filename: int(''.join(filter(str.isdigit, filename.encode("utf-8")))))
            #
            n = 0
            for i, file in enumerate(rolled_template_files):
                print("PQ: " + file)
                rolled_template = template.Bin2Template_Byte_TF_C(rolled_template_files[i], isLatent=False)
                outfile = output_dir + os.path.basename(file).split('.')[0] + '.dat'
                if rolled_template is None or len(rolled_template.texture_template) < 1:
                    with open(outfile, 'wb') as f:
                        tmp = (0,)
                        f.write(struct.pack('H', *tmp))
                    continue
                one_des = rolled_template.texture_template[0].des
                minutiae = rolled_template.texture_template[0].minutiae
                n = n + len(minutiae)
                codes = PQEncoder.encode_multi(one_des)

                rolled_template.texture_template[0].des = codes
                Template2Bin_Byte_PQ_rolled(outfile, version=1, T=rolled_template)


def encode_PQ_single(input_file, output_dir, fprint_type):
    isLatent = True if (fprint_type).lower() == 'latent' else False
    if isLatent:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        n = 0
        print("PQ: " + input_file)
        latent_template = template.Bin2Template_Byte_TF_C(input_file, isLatent=True)

        outfile = output_dir + os.path.basename(input_file).split('.')[0] + '.dat'

        Template2Bin_Byte_latent(outfile, version=1, T=latent_template)
    else:
        print("Single template PQ is not available for rolled prints. Please specify an input directory instead.")


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])

    if args.fprint_type and args.input_dir and args.output_dir:
        encode_PQ(input_dir=args.input_dir, output_dir=args.output_dir, fprint_type=args.fprint_type)
    elif args.fprint_type and args.input_file and args.output_dir:
        encode_PQ_single(input_file=args.input_file, output_dir=args.output_dir, fprint_type=args.fprint_type)
    else:
        print("Missing args.")
