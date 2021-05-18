import numpy as np

import torch
import torch.nn as nn


class Conv2d:
    def __init__(self, in_planes, planes, kernerl_size, stride=1, bias=False):
        self.in_planes = in_planes
        self.planes = planes
        self.kernerl_size = kernerl_size
        self.stride = stride
        self.bias = bias

        self.conv2d = nn.Conv2d(self.in_planes, self.planes, self.kernerl_size, stride=self.stride, bias=self.bias)

    def __call__(self, x):
        x = torch.Tensor(x).unsqueeze(0)
        o = self.conv2d(x).squeeze(0)
        return o.detach().cpu().numpy()


class AvgPool2d:
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

        self.avgpool2d = nn.AvgPool2d(self.kernel_size, self.stride)

    def __call__(self, x):
        x = torch.Tensor(x).unsqueeze(0)
        o = self.avgpool2d(x).squeeze(0)
        return o.detach().cpu().numpy()


class Linear:
    def __init__(self, in_neurons, neurons, bias=False):
        self.in_neurons = in_neurons
        self.neurons = neurons
        self.bias = bias

        self.linear = nn.Linear(self.in_neurons, self.neurons, self.bias)

    def __call__(self, x):
        x = torch.Tensor(x).unsqueeze(0)
        o = self.linear(x).squeeze(0)
        return o.detach().cpu().numpy()


class IF:
    def __init__(self):
        raise NotImplementedError

    def __call__(self, x):
        f = np.zeros(self.v.shape)
        self.v[:] = self.v + self.leak + x
        f[self.v >= self.th] = 1
        self.v[self.v >= self.th] = self.reset
        self.v[self.v < self.v_min] = self.v_min
        return f


class IF1d(IF):
    def __init__(self, neurons, leak=0.0, threshold=1.0, resting=0.0, v_min=None):
        self.neruons = neurons
        self.leak = leak
        self.th = threshold
        self.reset = resting
        if v_min is None:
            self.v_min = -10.0 * self.th

        self.v = np.zeros(self.neruons)


class IF2d(IF):
    def __init__(self, planes, width, height, leak=0.0, threshold=1.0, resting=0.0, v_min=None):
        self.planes = planes
        self.width = width
        self.height = height
        self.leak = leak
        self.th = threshold
        self.reset = resting
        if v_min is None:
            self.v_min = -10.0 * self.th

        self.v = np.zeros((self.planes, self.width, self.height))
