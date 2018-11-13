import torch
import torch.nn as nn

from torch.autograd import Variable

import numpy as np

class ConvolutionalQCNet(nn.Module):
    def __init__(self, input_shape):
        super(ConvolutionalQCNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            # nn.BatchNorm2d(16),
            nn.InstanceNorm2d(16),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5),
            # nn.BatchNorm2d(32),
            nn.InstanceNorm2d(32),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.InstanceNorm2d(32),
            # nn.BatchNorm2d(32),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.InstanceNorm2d(64),
            # nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5),
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


class BigConvolutionalQCNet(nn.Module):
    def __init__(self, input_shape):
        super(BigConvolutionalQCNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            # nn.InstanceNorm2d(64),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            # nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.InstanceNorm2d(32),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3),
            # nn.InstanceNorm2d(64),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            # nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.InstanceNorm2d(32),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3),
            # nn.InstanceNorm2d(64),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.InstanceNorm2d(32),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3),
            # nn.InstanceNorm2d(64),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            # nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
        )

        self.flat_features = self.get_flat_features(input_shape, self.features)

        self.classifier = nn.Sequential(
            nn.Linear(self.flat_features, 256),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 256),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
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


class ModelWithBagDistribution(nn.Module):
    """
    Wraps a model that predicts slice-wise to learn distribution across slices
    """
    def __init__(self, model, n_slices):
        super(ModelWithBagDistribution, self).__init__()
        self.features = nn.Sequential(model.features)
        self.slice_classifier = nn.Sequential(model.classifier)

        self.n_slices = n_slices

        self.bag_classifier = nn.Sequential(
            nn.Linear(n_slices*2, n_slices*2, bias=False),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(n_slices*2, n_slices, bias=False),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5)
        )

        self.output = nn.Softmax(dim=-1)

        for m in self.modules():
            print(m)

    def forward(self, input):
        print('input:', input.shape)
        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.slice_classifier(x)
        x = x.view(1, -1)

        out = self.bag_classifier(x)
        return self.output(out)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature