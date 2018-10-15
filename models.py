import torchvision
import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import numbers
import types
import collections
import warnings
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TTF


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        x = self.stn(x)

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class DoubleGauss(object):

    def __init__(self, resample=False, expand=False, center=None):
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params():
        lr = random.randint(0, 1)
        if lr == 0:
            angle = random.uniform(-90, -45)
            # angle = random.gauss(-45, 5)
        else:
            angle = random.uniform(45, 90)
            # angle = random.gauss(-135, 5)
        # print('%d-%.2f' % (lr, angle))
        return angle

    def __call__(self, img):
        angle = self.get_params()
        return TTF.rotate(img, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = 'null'
        return format_string
