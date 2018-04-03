from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import h5py, pickle, os, time
import numpy as np

from ml_experiment import setup_experiment
from visualizations import plot_roc, plot_sens_spec, make_roc_gif, GradCam

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Training settings
parser = argparse.ArgumentParser(description='PyTorch DeepMRIQC training.')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--val-batch-size', type=int, default=8, metavar='N', help='input batch size for validation (default: 8')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--folds', type=int, default=10, metavar='N',
                    help='number of folds to cross-validate over (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# workdir = '/data1/users/adoyle/'
workdir = '/home/users/adoyle/deepqc/'
input_filename = 'deepqc-all-sets.hdf5'

class QCDataset(Dataset):
    def __init__(self, hdf5_file_path, all_indices, random_slice=False, augmentation_type=None):
        f = h5py.File(hdf5_file_path)
        self.images = f['MRI']
        self.labels = f['qc_label']
        self.site = f['dataset']

        self.n_subjects = len(all_indices)
        self.indices = np.zeros((self.n_subjects))

        self.random_slice = random_slice

        for i, index in enumerate(all_indices):
            self.indices[i] = index

    def __getitem__(self, index):
        good_index = self.indices[index]

        if self.random_slice:
            slice_modifier = np.random.randint(-10, 10)
        else:
            slice_modifier = 0

        image = self.images[good_index, ...][np.newaxis, :, :, 192//2 + slice_modifier]
        label = np.argmax(self.labels[good_index, ...])
        site = self.site[good_index].decode('UTF-8')

        return np.float32(image), label, site

    def __len__(self):
        return self.n_subjects


abide_indices = pickle.load(open(workdir + 'abide_indices.pkl', 'rb'))
ds030_indices = pickle.load(open(workdir + 'ds030_indices.pkl', 'rb'))
ibis_indices = pickle.load(open(workdir + 'ibis_indices.pkl', 'rb'))
ping_indices = pickle.load(open(workdir + 'ping_indices.pkl', 'rb'))

# all_train_indices = list(abide_indices) + list(ibis_indices) + list(ping_indices)
# all_train_indices = abide_indices

all_train_indices = list(ibis_indices)

train_dataset = QCDataset(workdir + input_filename, all_train_indices, random_slice=True)
test_dataset = QCDataset(workdir + input_filename, all_train_indices)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, **kwargs)


class FullyConnectedQCNet(nn.Module):
    def __init__(self):
        super(FullyConnectedQCNet, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.Dropout(),
            nn.Linear(2)
        )

        self.output = nn.LogSoftmax()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.output(x)
        return x


class ConvolutionalQCNet(nn.Module):
    def __init__(self):
        super(ConvolutionalQCNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.Dropout(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            # nn.BatchNorm2d(64),
            nn.Dropout(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.Dropout(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Dropout(),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.output = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # print(x)
        x = self.classifier(x)
        x = self.output(x)
        return x


def train(epoch):
    model.train()
    train_loss, correct = 0, 0

    truth, probabilities = np.zeros((len(train_loader.dataset))), np.zeros((len(train_loader.dataset), 2))

    for batch_idx, (data, target, sites) in enumerate(train_loader):
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

        train_loss += loss_val.data[0]                          # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]              # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    train_loss /= len(train_loader.dataset)
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))

    return truth, probabilities

def validate():
    model.eval()
    validation_loss, correct = 0, 0

    truth, probabilities = np.zeros((len(validation_loader.dataset))), np.zeros((len(validation_loader.dataset), 2))

    for batch_idx, (data, target, sites) in enumerate(validation_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        loss_function = nn.CrossEntropyLoss()

        validation_loss += loss_function(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # print('val batch shape:', output.data.cpu().numpy().shape)
        truth[batch_idx * args.val_batch_size:(batch_idx + 1) * args.val_batch_size] = target.data.cpu().numpy()
        probabilities[batch_idx * args.val_batch_size:(batch_idx + 1) * args.val_batch_size] = output.data.cpu().numpy()

    validation_loss /= len(validation_loader.dataset)

    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))

    return truth, probabilities

def test():
    model.eval()
    test_loss, correct = 0, 0

    truth, probabilities = np.zeros((len(test_loader.dataset))), np.zeros((len(test_loader.dataset), 2))

    for batch_idx, (data, target, sites) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        loss_function = nn.CrossEntropyLoss()

        test_loss += loss_function(output, target).data[0]      # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]              # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        truth[batch_idx * args.test_batch_size:(batch_idx + 1) * args.test_batch_size] = target.data.cpu().numpy()
        probabilities[batch_idx * args.test_batch_size:(batch_idx + 1) * args.test_batch_size] = output.data.cpu().numpy()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return truth, probabilities

model = ConvolutionalQCNet()

def example_pass_fails(model, train_loader, test_loader, results_dir, grad_cam):
    model.eval()

    histograms = {}

    mri_sites = ['IBIS', 'PING', 'PITT', 'OLIN', 'OHSU', 'SDSU', 'TRINITY', 'UM', 'USM', 'YALE', 'CMU', 'LEUVEN', 'KKI',
             'NYU', 'STANFORD', 'UCLA', 'MAX_MUN', 'CALTECH', 'SBL', 'ds030']

    for site in mri_sites:
        histograms[site] = np.zeros(256, dtype='float32')

    bins = np.linspace(0.0, 1.0, 257)

    os.makedirs(results_dir + '/imgs/', exist_ok=True)
    for batch_idx, (data, target, sites) in enumerate(train_loader):

        data, target = Variable(data, volatile=True), Variable(target)

        target_batch = target.data.cpu().numpy()
        image_batch = data.data.cpu().numpy()

        for img, site in zip(image_batch, sites):

            try:
                histo = np.histogram(img, bins=bins)
                histograms[site] += histo[0]
            except KeyError:
                print('Site missing')

        if batch_idx == 0:
            print(target_batch.shape, image_batch.shape)

        try:
            for i in range(data.shape[0]):
                if target_batch[i] == 0:
                    qc_decision = 'FAIL'
                else:
                    qc_decision = 'PASS'

                plt.close()
                plt.imshow(image_batch[i, 0, :, :], cmap='gray', origin='lower')
                plt.axis('off')
                filename = results_dir + '/imgs/' + qc_decision + '_' + site + '_' + str(batch_idx) + '_img_' + str(i) + '.png'
                plt.savefig(filename, bbox_inches='tight')
        except IndexError as e:
            print('Couldnt save one file')

    for batch_idx, (data, target, sites) in enumerate(test_loader):
        data, target = Variable(data, volatile=True), Variable(target)
        
        target_batch = target.data.cpu().numpy()
        image_batch = data.data.cpu().numpy()

        for img, site in zip(image_batch, sites):
            try:
                histo = np.histogram(img, bins=bins)
                histograms[site] += histo[0]
            except KeyError:
                print('Site missing')

        if batch_idx == 0:
            print(target_batch.shape, image_batch.shape)

        try:
            for i in range(data.shape[0]):
                if target_batch[i] == 0:
                    qc_decision = 'FAIL'
                else:
                    qc_decision = 'PASS'

                # mask = grad_cam(data[i, ...][np.newaxis, ...], target[i, ...][np.newaxis, ...])
                #
                # heatmap = np.uint8(cm.jet(mask)[:,:,0,:3]*255)
                # gray = np.uint8(cm.gray(data[i, ...]))
                #
                # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True)
                # ax1.imshow(image_batch[i, 0, :, :], cmap='gray', origin='lower')
                # ax2.imshow(gray, origin='lower')
                # ax2.imshow(heatmap, alpha=0.2, origin='lower')
                #
                # plt.savefig(results_dir + '/imgs/' + qc_decision + '_ds030_batch_' + str(batch_idx) + '_img_' + str(i) + '.png', bbox_inches='tight')
        except IndexError as e:
            pass

    fig, axes = plt.subplots(len(mri_sites), 1, sharex=True, figsize=(4, 24))
    for i, site in enumerate(mri_sites):
        # print(site, histograms[site])
        try:
            histograms[site] = np.divide(histograms[site], np.sum(histograms[site]))
            axes[i].plot(bins[:-1], histograms[site], lw=2, label=site)

            # axes[i].set_ylim([0, 0.2])
            axes[i].set_xlim([0, 1])

            axes[i].set_ylabel(site, fontsize=16)
            axes[i].set_xscale('log')
            axes[i].set_yscale('log')
        except:
            print('Problem normalizing histogram for', site)


    # plt.title('histogram of grey values', fontsize='24')
    plt.tight_layout()
    plt.savefig(results_dir + 'histograms.png', bbox_inches='tight')


if __name__ == '__main__':
    print('PyTorch implementation of DeepMRIQC.')
    start_time = time.time()
    print("Convolutional QC Model")
    print(model)

    results_dir, experiment_number = setup_experiment(workdir)
    n_train = len(all_train_indices)

    n_folds = args.folds
    results_shape = (args.folds, args.epochs)

    training_sensitivity, training_specificity, validation_sensitivity, validation_specificity, test_sensitivity, test_specificity, val_aucs = np.zeros(results_shape), np.zeros(results_shape), np.zeros(results_shape), np.zeros(results_shape), np.zeros(results_shape), np.zeros(results_shape), np.zeros(results_shape)
    best_auc_score = np.zeros(n_folds)

    if args.cuda:
        model.cuda()

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    pass_weight, fail_weight = 0, 0
    train_ground_truth = np.zeros(len(all_train_indices))
    groups = []

    print('Counting PASS/FAIL images...')
    for batch_idx, (img_data, target, sites) in enumerate(train_loader):
        train_ground_truth[args.batch_size * batch_idx:args.batch_size * (1 + batch_idx)] = target
        for site in sites:
            # print(site)
            groups.append(site)

    n_pass = np.sum(train_ground_truth, dtype='int')
    n_fail = len(all_train_indices) - np.sum(train_ground_truth, dtype='int')

    print('Whole training set has ' + str(n_pass) + ' PASS and ' + str(n_fail) + ' FAIL images')
    fail_weight = n_pass / len(all_train_indices)
    pass_weight = n_fail / len(all_train_indices)
    print('Setting class weighting to ' + str(fail_weight) + ' for FAIL class and ' + str(
        pass_weight) + ' for PASS class')

    # logo = LeaveOneGroupOut()
    # for fold_idx, (train_indices, validation_indices) in enumerate(logo.split(all_train_indices, train_ground_truth, groups=groups)):

    skf = StratifiedKFold(n_splits=3)
    for fold_idx, (train_indices, validation_indices) in enumerate(skf.split(all_train_indices, train_ground_truth)):
        fold_num = fold_idx + 1
        print("Starting fold", str(fold_num))
        model = ConvolutionalQCNet()
        if args.cuda:
            model.cuda()

        train_dataset = QCDataset(workdir + input_filename, train_indices, random_slice=True)
        validation_dataset = QCDataset(workdir + input_filename, validation_indices, random_slice=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.val_batch_size, shuffle=False, **kwargs)

        print('This fold has', str(len(train_loader.dataset)), 'training images and', str(len(validation_loader.dataset)), 'validation images. There are', str(len(test_loader.dataset)), 'images in the test dataset')

        optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

        for epoch_idx, epoch in enumerate(range(1, args.epochs + 1)):
            train_truth, train_probabilities = train(epoch)
            train_predictions = np.argmax(train_probabilities, axis=-1)

            val_truth, val_probabilities = validate()
            val_predictions = np.argmax(val_probabilities, axis=-1)

            test_truth, test_probabilities = test()
            test_predictions = np.argmax(test_probabilities, axis=-1)

            train_auc, val_auc, test_auc = plot_roc(train_truth, train_probabilities, val_truth, val_probabilities, test_truth, test_probabilities, results_dir, epoch, fold_num)

            try:
                [[train_tp, train_fn], [train_fp, train_tn]] = confusion_matrix(train_truth, train_predictions)
                [[val_tp, val_fn], [val_fp, val_tn]] = confusion_matrix(val_truth, val_predictions)
                [[test_tp, test_fn], [test_fp, test_tn]] = confusion_matrix(test_truth, test_predictions)

                training_sensitivity[fold_idx, epoch_idx] = train_tp / (train_tp + train_fn)
                training_specificity[fold_idx, epoch_idx] = train_tn / (train_tn + train_fp)

                validation_sensitivity[fold_idx, epoch_idx] = val_tp / (val_tp + val_fn)
                validation_specificity[fold_idx, epoch_idx] = val_tn / (val_tn + val_fp)

                test_sensitivity[fold_idx, epoch_idx] = test_tp / (test_tp + test_fn)
                test_specificity[fold_idx, epoch_idx] = test_tn / (test_tn + test_fp)

                val_aucs[fold_idx, epoch_idx] = val_auc
            except:
                print('ERROR could not calculate confusion matrix properly, probably only one class predicted/present in ground truth.')

            if val_auc + train_auc > best_auc_score[fold_idx]:
                best_auc_score[fold_idx] = val_auc + train_auc
                torch.save(model.state_dict(), results_dir + 'qc_torch_fold_' + str(fold_num) + '.tch')

        try:
            plot_sens_spec(training_sensitivity[fold_idx, :], training_specificity[fold_idx, :], validation_sensitivity[fold_idx, :], validation_specificity[fold_idx, :],
                   test_sensitivity[fold_idx, :], test_specificity[fold_idx, :], results_dir, fold_num)
        except:
            print('ERROR could not save sensitivity/specificity plot for epoch', epoch)


    grad_cam = GradCam(model = model, target_layer_names=['output'], use_cuda=args.cuda)
    # example_pass_fails(model, train_loader, test_loader, results_dir, grad_cam)

    dummy_input = Variable(torch.randn(1, 1, 192, 256))

    input_names = ["coronal_slice"]
    output_names = ["pass_fail"]

    model = ConvolutionalQCNet()
    model.load_state_dict(torch.load(results_dir + 'qc_torch_fold_1.tch'))
    model.eval()

    torch.onnx.export(model, dummy_input, "ibis_qc_net_v1.proto", verbose=True)

    for fold in range(skf.get_n_splits()):
        make_roc_gif(results_dir, args.epochs, fold+1)

    time_elapsed = time.time() - start_time
    print('Whole experiment took', time_elapsed/216000, 'hours')

    print('This experiment was brought to you by the number:', experiment_number)

