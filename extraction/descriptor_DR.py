import os
import sys
import torch
from torch.autograd import Variable
import random
import models
import datasets
import numpy as np
import template_2 as template
import argparse
import json


def load_model(model, filename, cuda):
    if os.path.isdir(filename):
        print("Loading dimentionality reduction model: '{}'".format(filename))
        state_dict = {}
        if cuda:
            for key in list(model):
                state_dict[key] = torch.load(os.path.join(filename,
                                                          'model_%sD_%s.pth' % (key, os.path.basename(filename))))
        else:
            for key in list(model):
                state_dict[key] = torch.load(os.path.join(filename,
                                                          'model_%sD_%s.pth' % (key, os.path.basename(filename))),
                                             map_location=lambda storage, loc: storage)
    elif os.path.isfile(filename):
        print("Loading dimentionality reduction model: '{}'".format(filename))
        state_dict = {}
        if cuda:
            for key in list(model):
                state_dict[key] = torch.load(filename)
        else:
            for key in list(model):
                state_dict[key] = torch.load(
                    filename, map_location=lambda storage, loc: storage)
    else:
        raise (Exception("No checkpoint found at '{}'".format(filename)))
    for key_model in list(model):
        model_dict = model[key_model].state_dict()
        update_dict = {}
        valid_keys = list(model_dict)
        for i, key in enumerate(state_dict[key_model]):
            update_dict[valid_keys[i]] = state_dict[key_model][key]
        model[key_model].load_state_dict(update_dict)
    return model


def setup(filename_model, cuda):
    model_type = 'CompNet'
    model_options = [{"in_dims": 192, "out_dims": 96}]
    model = {}
    keys = [str(x['out_dims']) for x in model_options]
    for i, key in enumerate(keys):
        model[key] = getattr(models, model_type)(**model_options[i])

    if cuda:
        for i, key in enumerate(keys):
            model[key] = model[key].cuda()

    model = load_model(model, filename_model, cuda)

    return model


def extract_features(model, dataloader, cuda):
    torch.cuda.empty_cache()
    for key in list(model):
        model[key].eval()

    model_options = [{"in_dims": 192, "out_dims": 96}]
    target_dim = model_options[-1]['out_dims']

    inputs = torch.zeros(target_dim)

    if cuda:
        inputs = inputs.cuda()

    inputs = Variable(inputs)

    # extract features
    for i, (data) in enumerate(dataloader):

        inputs.data.resize_(data.size()).copy_(data)

        # output features
        outputs = {}
        keys = list(model)
        for j, key in enumerate(keys):
            if j == 0:
                outputs[key] = model[key](inputs)
            else:
                outputs[key] = model[key](outputs[keys[j - 1]])
        if i == 0:
            features = outputs[str(target_dim)].data
        else:
            features = torch.cat((features, outputs[str(target_dim)].data), 0)

    feat = features.data.cpu().numpy()

    return feat


def template_compression(input_dir='', output_dir=None, model_path=None, isLatent=False, config=None):
    if not config and not (input_dir and model_path):
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        with open(dir_path + 'config') as config_file:
            config = json.load(config_file)
    if not output_dir:
        output_dir = input_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # parse the arguments
    random.seed(0)
    torch.manual_seed(0)
    cuda = True

    import glob
    file_list = glob.glob(input_dir + '*.dat')
    if file_list is None or len(file_list) == 0:
        print ("File list empty", file_list)
        return

    file_list.sort(key=lambda filename: int(''.join(filter(str.isdigit, filename.encode("utf-8")))))

    # Create Model
    # filename_model = './dim_reduction/testmodel'

    model = setup(model_path, cuda)
    batch_size = 128
    nthreads = 8
    assert(output_dir is not None)
    for i, file in enumerate(file_list):
        if i > 100000:
            break
        print ("DR: ", file)

        output_file = output_dir + file.split('/')[-1]
        T = template.Bin2Template_Byte_TF_C(file, isLatent=isLatent)
        num = len(T.minu_template)
        for j in range(num):
            des = T.minu_template[j].des.copy()
            if len(des) == 0:
                continue
            dataset_feat = datasets.Featarray(des)
            dataloader = torch.utils.data.DataLoader(dataset_feat, batch_size=batch_size, num_workers=int(nthreads),
                                                     shuffle=False, pin_memory=True)
            features = extract_features(model, dataloader, cuda)
            for k in range(features.shape[0]):
                norm = np.linalg.norm(features[k])
                features[k] = features[k] / norm * 1.73
            T.minu_template[j].des = features.copy()
        num = len(T.texture_template)
        for j in range(num):
            des = T.texture_template[j].des.copy()
            if len(des) == 0:
                print("Skipping, length of descriptor is 0.")
                continue
            dataset_feat = datasets.Featarray(des)
            dataloader = torch.utils.data.DataLoader(dataset_feat, batch_size=batch_size, num_workers=int(nthreads),
                                                     shuffle=False, pin_memory=True)
            features = extract_features(model, dataloader, cuda)
            for k in range(features.shape[0]):
                norm = np.linalg.norm(features[k])
                features[k] = features[k] / norm * 1.73
            T.texture_template[j].des = features.copy()
            template.Template2Bin_Byte_TF_C(output_file, T, isLatent=isLatent, save_mask=False)


def template_compression_single(input_file='', output_dir=None, model_path=None, isLatent=True, config=None):
    if not input_file:
        print("No input file specified for single template processing.")
        return
    if not config and not model_path:
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        with open(dir_path + 'config') as config_file:
            config = json.load(config_file)
    if not output_dir:
        output_dir = os.path.dirname(input_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # parse the arguments
    random.seed(0)
    torch.manual_seed(0)
    cuda = True

    # Create Model
    # filename_model = './dim_reduction/testmodel'

    model = setup(model_path, cuda)
    batch_size = 128
    nthreads = 8
    assert(output_dir is not None)
    output_file = input_file
    T = template.Bin2Template_Byte_TF_C(input_file, isLatent=isLatent)
    num = len(T.minu_template)
    for j in range(num):
        des = T.minu_template[j].des.copy()
        if len(des) == 0:
            continue
        dataset_feat = datasets.Featarray(des)
        dataloader = torch.utils.data.DataLoader(dataset_feat, batch_size=batch_size, num_workers=int(nthreads),
                                                 shuffle=False, pin_memory=True)
        features = extract_features(model, dataloader, cuda)
        for k in range(features.shape[0]):
            norm = np.linalg.norm(features[k])
            features[k] = features[k] / norm * 1.73
        T.minu_template[j].des = features.copy()
    num = len(T.texture_template)
    for j in range(num):
        des = T.texture_template[j].des.copy()
        if len(des) == 0:
            print("Skipping, length of descriptor is 0.")
            continue
        dataset_feat = datasets.Featarray(des)
        dataloader = torch.utils.data.DataLoader(dataset_feat, batch_size=batch_size, num_workers=int(nthreads),
                                                 shuffle=False, pin_memory=True)
        features = extract_features(model, dataloader, cuda)
        for k in range(features.shape[0]):
            norm = np.linalg.norm(features[k])
            features[k] = features[k] / norm * 1.73
        T.texture_template[j].des = features.copy()
        template.Template2Bin_Byte_TF_C(output_file, T, isLatent=isLatent, save_mask=False)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--fprint_type', type=str, help='Type of fingerprint templates (latent or rolled)',
                        default='latent')
    parser.add_argument('--output_dir', type=str, help='Path to directory where reduced templates should be saved')
    parser.add_argument('--input_dir', type=str, help='Path to directory of templates to reduce (batch operation)')
    parser.add_argument('--input_file', type=str, help='Path to single template file to be reduced')
    parser.add_argument('--model_path', type=str, help='Path to dimensionality reduction model')
    return parser.parse_args(argv)


if __name__ == "__main__":

    args = parse_arguments(sys.argv[1:])
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    with open(dir_path + '/afis.config') as config_file:
        config = json.load(config_file)

    if args.fprint_type and args.input_dir:
        if not args.output_path:
            args.output_path = args.input_dir
        template_compression(input_dir=args.input_dir, output_path=args.output_path,
                             model_path=(args.model_path if args.model_path else config['DimensionalityReductionModel']),
                             isLatent=(True if (args.fprint_type).lower() == 'latent' else False))
    elif(args.fprint_type and args.input_file):
        if not args.output_path:
            args.output_path = os.path.dirname(args.input_file)
        template_compression_single(input_file=args.input_file, output_path=args.output_path,
                                    model_path=(args.model_path if args.model_path else config['DimensionalityReductionModel']),
                                    isLatent=(True if (args.fprint_type).lower() == 'latent' else False))
    else:
        print("Using arguments from config file, assuming latent batch processing.")
        if not os.path.exists(config['LatentTemplateDirectory']):
            os.makedirs(config['LatentTemplateDirectory'])
        template_compression(input_path=config['LatentTemplateDirectory'],
                             model_path=config['DimensionalityReductionModel'],
                             output_path=config['LatentTemplateDirectory'], isLatent=True)
