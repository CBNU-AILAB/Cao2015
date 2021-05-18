import numpy as np

import torch

from network import TailoredCNN, SpikingCNN


def load_model(pretrained):
    model = TailoredCNN()
    model.load_state_dict(torch.load(pretrained)['model'])

    named_param = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            named_param[name] = param.data.detach().cpu().numpy()

    # w = model.conv1.conv2d.weight.data.detach().cpu().numpy()
    # model.conv1.conv2d.weight.data.copy_()

    model = SpikingCNN()
    model.conv1.conv2d.weight.data.copy_(torch.Tensor(named_param['extractor.0.weight']))
    model.conv2.conv2d.weight.data.copy_(torch.Tensor(named_param['extractor.3.weight']))
    model.conv3.conv2d.weight.data.copy_(torch.Tensor(named_param['extractor.6.weight']))
    model.fc1.linear.weight.data.copy_(torch.Tensor(named_param['classifier.0.weight']))
    model.fc2.linear.weight.data.copy_(torch.Tensor(named_param['classifier.2.weight']))
    # model.conv1.conv2d.weight.data.copy_(torch.Tensor(named_param['conv1.weight']))
    # model.conv2.conv2d.weight.data.copy_(torch.Tensor(named_param['conv2.weight']))
    # model.conv3.conv2d.weight.data.copy_(torch.Tensor(named_param['conv3.weight']))
    # model.fc1.linear.weight.data.copy_(torch.Tensor(named_param['fc1.weight']))
    # model.fc2.linear.weight.data.copy_(torch.Tensor(named_param['fc2.weight']))

    return model


class SpikeGenerator:
    def __init__(self, c=0.3):
        self.c = c

    def __call__(self, x):
        r = np.random.uniform(0, 1, x.shape)
        f = np.zeros(x.shape)
        f[self.c*x > r] = 1
        return f


class CIFAR10_SpikeGenerator:
    def __init__(self, channel, height, width, threshold=1):
        self.channel = channel
        self.height = height
        self.width = width
        self.th = threshold
        self.v_min = -10*self.th

        self.v = np.zeros((channel, height, width))

    def __call__(self, x):
        assert self.v.shape == x.shape
        self.v[:] = self.v + x
        f = np.zeros(self.v.shape)
        f[self.v >= self.th] = 1
        self.v[self.v >= self.th] = self.v[self.v >= self.th] - self.th
        self.v[self.v < self.v_min] = self.v_min
        return f


class SpikeCounter:
    def __init__(self):
        pass

    def __call__(self, x):
        o = np.zeros(x[0].shape[0])
        for v in x:
            o += v
        return o