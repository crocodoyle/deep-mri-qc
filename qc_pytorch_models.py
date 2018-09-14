import torch
import torch.nn as nn

from torch.autograd import Variable

import numpy as np

class ConvolutionalQCNet(nn.Module):
    def __init__(self, input_shape):
        super(ConvolutionalQCNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            # nn.BatchNorm2d(16),
            nn.InstanceNorm2d(16),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3),
            # nn.BatchNorm2d(32),
            nn.InstanceNorm2d(32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.InstanceNorm2d(32),
            # nn.BatchNorm2d(32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.InstanceNorm2d(64),
            # nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.InstanceNorm2d(128),
            nn.Dropout(0.5),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(128, 256, kernel_size=3),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.MaxPool2d(2)
        )

        self.flat_features = self.get_flat_features(input_shape, self.features)

        self.classifier = nn.Sequential(
            nn.Linear(self.flat_features, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

        # self.output = nn.LogSoftmax(dim=-1)

        # print('ConvolutionalQC structure:')
        # print(self.features)
        # print(self.classifier)
        # print(self.output)

    def get_flat_features(self, image_shape, features):
        f = features(Variable(torch.ones(1,*image_shape)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        # print('input shape:', x.shape)
        x = self.features(x)
        # print('features shape:', x.shape)
        x = x.view(x.size(0), -1)
        # print('features reshaped:', x.shape)
        # print('flattened features:', self.flat_features)
        x = self.classifier(x)
        # print('classifier shape:', x.shape)
        # x = self.output(x)
        return x