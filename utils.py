import glob
import random
import numpy as np
import torch
from torch.autograd import Variable


class get():
    def __init__(self, root):
        self.root = root

    def __call__(self, extension='.jpg'):
        path_list = []

        for fname in sorted(glob.glob(self.root + '/**/*.dcm', recursive=True)):
            # print(fname)
            path_list.append(fname)

        return path_list


class HUrescale:
    def __init__(self, img):
        self.img = img

    def img2var(self, base, range):
        # output clipped and scaled to [-1,1]
        return(2 * (np.clip(self.img, base, base + range) - base) / range - 1.0)

    def var2img(self, var, base, range):
        # inverse of img2var
        return(0.5 * (1.0 + var) * range + base)

    def to256(self, min, max, type=None):
        if type:
            img_ = (255 * (np.clip(self.img, min, max) - min) / (max - min)).astype(type)
        else:
            img_ = 255 * (np.clip(self.img, min, max) - min) / (max - min)
        return img_


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


# 過去の生成データ(50iter分)を保持しておく
class ReplayBuffer():
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            #
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class weights_init_normal():
    def __init__(self, m):
        self.m = m
        self.classname = m.__class__.__name__

    def __call__(self):
        if self.classname.find('Conv') != -1:
            torch.nn.init.normal(self.m.weight.data, 0.0, 0.02)
            # print(self.classname, 'is initialized')
        elif self.classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal(self.m.weight.data, 1.0, 0.02)
            torch.nn.init.constant(self.m.bias.data, 0.0)
            # print(self.classname, 'is initialized')
        elif self.classname.find('InstanceNorm2d') != -1:
            torch.nn.init.normal(self.m.weight.data, 1.0, 0.02)
            torch.nn.init.constant(self.m.bias.data, 0.0)
            # print(self.classname, 'is initialized')
