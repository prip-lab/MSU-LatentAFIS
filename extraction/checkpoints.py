# checkpoints.py

import os
import torch

class Checkpoints:
    def __init__(self, args):
        self.dir_save = args.save_dir
        self.model_filename = args.resume
        self.save_results = args.save_results
        self.cuda = args.cuda

        if self.save_results and not os.path.isdir(self.dir_save):
            os.makedirs(self.dir_save)

    def latest(self, name):
        if name == 'resume':
            return self.model_filename

    def save(self, epoch, acc, model, best):
        if best is True:
            acc = "{0:.2f}".format(acc)
            subdir_save = os.path.join(self.dir_save,acc)
            if not os.path.isdir(subdir_save):
                os.makedirs(subdir_save)
            for key in list(model):
                torch.save(model[key].state_dict(),
                           '%s/model_%d_%sD_%s.pth' % (subdir_save, epoch, key, acc))

    def load(self, model, filename):
        if os.path.isdir(filename):
            print("=> loading checkpoint '{}'".format(filename))
            state_dict = {}
            if self.cuda:
                for key in list(model):
                    state_dict[key] = torch.load(os.path.join(filename,'model_%sD_%s.pth' % (key, os.path.basename(filename))))
            else:
                for key in list(model):
                    state_dict[key] = torch.load(
                        os.path.join(filename,'model_%sD_%s.pth' % (key, os.path.basename(filename))), \
                        map_location=lambda storage, loc: storage)
        elif os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            state_dict = {}
            if self.cuda:
                for key in list(model):
                    state_dict[key] = torch.load(filename)
            else:
                for key in list(model):
                    state_dict[key] = torch.load(
                        filename, map_location=lambda storage, loc: storage)
        else:
            raise (Exception("=> no checkpoint found at '{}'".format(filename)))
        for key_model in list(model):
            model_dict = model[key_model].state_dict()
            update_dict = {}
            valid_keys = list(model_dict)
            for i,key in enumerate(state_dict[key_model]):
                update_dict[valid_keys[i]] = state_dict[key_model][key] 
            model[key_model].load_state_dict(update_dict)
        return model
