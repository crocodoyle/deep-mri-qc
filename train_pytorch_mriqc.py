from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable

import h5py, pickle, os
import numpy as np

from ml_experiment import setup_experiment
from visualizations import plot_roc, plot_sens_spec

from sklearn.metrics import confusion_matrix

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description='PyTorch DeepMRIQC training.')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# workdir = '/data1/users/adoyle/'
workdir = '/home/users/adoyle/deepqc/'

class QCDataset(Dataset):
    def __init__(self, hdf5_file_path, all_indices, augmentation_type=None):
        f = h5py.File(hdf5_file_path)
        self.images = f['MRI']
        self.labels = f['qc_label']

        self.n_subjects = len(all_indices)
        self.indices = np.zeros((self.n_subjects))

        for i, index in enumerate(all_indices):
            self.indices[i] = index

    def __getitem__(self, index):
        good_index = self.indices[index]

        image = self.images[good_index, ...][np.newaxis, :, :, 192//2]
        label = np.argmax(self.labels[good_index, ...])

        return image, label

    def __len__(self):
        return self.n_subjects


abide_indices = pickle.load(open(workdir + 'abide_indices.pkl', 'rb'))
ds030_indices = pickle.load(open(workdir + 'ds030_indices.pkl', 'rb'))
ibis_indices = pickle.load(open(workdir + 'ibis_indices.pkl', 'rb'))
ping_indices = pickle.load(open(workdir + 'ping_indices.pkl', 'rb'))

train_indices = abide_indices + ds030_indices + ibis_indices + ping_indices

train_dataset = QCDataset(workdir + 'deepqc-all-sets.hdf5', train_indices)
test_dataset = QCDataset(workdir + 'deepqc-all-sets.hdf5', ds030_indices)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, **kwargs)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv5_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(3072, 256)
        self.fc2 = nn.Linear(256, 64)
        self.output = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = F.relu(F.max_pool2d(self.conv5_drop(self.conv5(x)), 2))
        x = x.view(-1, 3072)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.output(x)
        return F.log_softmax(x, dim=1)

model = Net()
if args.cuda:
    model.cuda()

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

pass_weight, fail_weight = 0, 0
train_ground_truth = np.zeros(len(train_indices))

print('Counting PASS/FAIL images...')
for batch_idx, (img_data, target) in enumerate(train_loader):
    train_ground_truth[args.batch_size*batch_idx:args.batch_size*(1+batch_idx)] = target

n_pass = np.sum(train_ground_truth, dtype='int')
n_fail = len(train_indices) - np.sum(train_ground_truth, dtype='int')

print('Training set has ' + str(n_pass) + ' PASS and ' + str(n_fail) + ' FAIL images')
fail_weight = n_pass / len(train_indices)
pass_weight = n_fail / len(train_indices)
print('Setting class weighting to ' + str(fail_weight) + ' for FAIL class and ' + str(pass_weight) + ' for PASS class')

def train(epoch, fold_num=-1):
    model.train()

    truth, probabilities = np.zeros((len(train_indices))), np.zeros((len(train_indices), 2))

    for batch_idx, (data, target) in enumerate(train_loader):
        class_weight = torch.FloatTensor([fail_weight, pass_weight])
        if args.cuda:
            data, target, class_weight = data.cuda(), target.cuda(), class_weight.cuda()
        data, target, class_weight = Variable(data), Variable(target).type(torch.cuda.LongTensor), Variable(class_weight)
        optimizer.zero_grad()
        output = model(data)
        # print('P(qc=0):', np.exp(output.data.cpu().numpy())[0])
        loss = nn.CrossEntropyLoss(weight=class_weight)
        loss_val = loss(output, target)
        loss_val.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_val.data[0]))

        truth[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size] = target.data.cpu().numpy()
        probabilities[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size] = output.data.cpu().numpy()

    print('Training GT:', truth)
    print('Predictions:', probabilities)

    return truth, probabilities

def test():
    model.eval()
    test_loss, correct = 0, 0

    truth, probabilities = np.zeros((len(ds030_indices))), np.zeros((len(ds030_indices), 2))

    for batch_idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        loss_function = nn.CrossEntropyLoss()

        test_loss += loss_function(output, target).data[0]      # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]              # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        truth[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size] = target.data.cpu().numpy()
        probabilities[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size] = output.data.cpu().numpy()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return truth, probabilities

def example_pass_fails(results_dir):
    model.eval()

    os.makedirs(results_dir + '/imgs/')
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        target_batch = target.data.cpu().numpy()
        image_batch = data.data.cpu().numpy()

        if batch_idx == 0:
            print(target_batch.shape, image_batch.shape)

        try:
            for i in range(args.batch_size):
                if target_batch[i] == 0:
                    qc_decision = 'FAIL'
                else:
                    qc_decision = 'PASS'

                plt.close()
                plt.imshow(image_batch[i, 0, :, :], cmap='gray', origin='lower')
                plt.axis('off')
                plt.savefig(results_dir + '/imgs/' + qc_decision + '_batch_' + str(batch_idx) + '_img_' + str(i) + '.png', bbox_inches='tight')
        except IndexError as e:
            pass

if __name__ == '__main__':
    print('PyTorch implementation of DeepMRIQC.')

    results_dir, experiment_number = setup_experiment(workdir)
    n_train = len(train_indices)

    training_sensitivity, training_specificity, validation_sensitivity, validation_specificity, test_sensitivity, test_specificity = np.zeros(args.epochs), np.zeros(args.epochs), np.zeros(args.epochs), np.zeros(args.epochs), np.zeros(args.epochs), np.zeros(args.epochs)

    for epoch_idx, epoch in enumerate(range(1, args.epochs + 1)):
        train_truth, train_probabilities = train(epoch)
        train_predictions = np.argmax(train_probabilities, axis=-1)

        plot_roc(train_truth, train_probabilities, results_dir, epoch)

        test_truth, test_probabilities = test()
        test_predictions = np.argmax(test_probabilities, axis=-1)

        [[train_tp, train_fn], [train_fp, train_tn]] = confusion_matrix(train_truth, train_predictions)
        [[test_tp, test_fn], [test_fp, test_tn]] = confusion_matrix(test_truth, test_predictions)

        training_sensitivity[epoch_idx] = train_tp / (train_tp + train_fn)
        training_specificity[epoch_idx] = train_tn / (train_tn + train_fp)

        test_sensitivity[epoch_idx] = test_tp / (test_tp + test_fn)
        test_specificity[epoch_idx] = test_tn / (test_tn + test_fp)

    example_pass_fails(results_dir)
    plot_sens_spec(training_sensitivity, training_specificity, None, None, test_sensitivity, test_specificity, results_dir)