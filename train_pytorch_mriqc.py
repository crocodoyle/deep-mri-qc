from __future__ import print_function
import argparse
import torch
# import torch.multiprocessing
# torch.multiprocessing.set_start_method('spawn')
from shutil import copyfile, SameFileError

import torch.nn as nn
import torch.optim as optim
import torch.onnx

import densenet

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.autograd import Variable

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from temperature_scaling import ModelWithTemperature, ModelWithSoftmax, ECELoss

from qc_pytorch_models import ConvolutionalQCNet, BigConvolutionalQCNet, ModelWithBagDistribution, BagDistributionModel

import h5py, pickle, os, time, sys, csv
import numpy as np

from ml_experiment import setup_experiment
from visualizations import plot_roc, plot_sens_spec, make_roc_gif, GradCam, sens_spec_across_folds, plot_confidence

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.utils import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.cm as cm

workdir = '/data1/users/adoyle/deepqc/'

abide_filename = 'abide.hdf5'
ds030_filename = 'ds030.hdf5'

epsilon = 1e-6

image_shape = (189, 233, 197)

class RandomSlicesQCDataset(Dataset):
    def __init__(self, f, all_indices, n_slices=40):
        self.images = f['MRI']
        self.labels = f['qc_label']
        self.confidence = f['label_confidence']

        self.n_subjects = len(all_indices)
        self.indices = np.zeros((self.n_subjects))

        self.n_slices = n_slices

        for i, index in enumerate(all_indices):
            self.indices[i] = index

    def __getitem__(self, index):
        good_index = self.indices[index]

        slice_modifier = np.random.randint(-self.n_slices, self.n_slices)

        label = self.labels[good_index]
        label_confidence = self.confidence[good_index]
        image_slice = self.images[good_index, :, image_shape[0] // 2 + slice_modifier, :, :]

        return image_slice, int(label), label_confidence

    def __len__(self):
        return self.n_subjects

class AllSlicesQCDataset(Dataset):
    def __init__(self, f, all_indices, n_slices=40):
        self.images = f['MRI']
        self.labels = f['qc_label']
        self.confidence = f['label_confidence']

        self.n_subjects = len(all_indices)
        self.indices = np.zeros((self.n_subjects))

        self.n_slices = n_slices

        for i, index in enumerate(all_indices):
            self.indices[i] = index

    def __getitem__(self, index):
        good_index = self.indices[index]

        label = self.labels[good_index]
        label_confidence = self.confidence[good_index]
        image_slices = self.images[good_index, 0, image_shape[0] // 2 - n_slices : image_shape[0] // 2 + n_slices, :, :]

        return image_slices, int(label), label_confidence

    def __len__(self):
        return self.n_subjects

class SlicePredictionQCDataset(Dataset):
    def __init__(self, slice_predictions, labels, all_indices):
        self.n_subjects = len(all_indices)

        self.indices = np.zeros((self.n_subjects), dtype='uint8')

        self.slice_predictions = slice_predictions
        self.labels = labels

        for i, index in enumerate(all_indices):
            self.indices[i] = index

    def __getitem__(self, index):
        good_index = self.indices[index]

        label = self.labels[good_index]
        slice_predictions = self.slice_predictions[good_index, :]

        return slice_predictions, label

    def __len__(self):
        return self.n_subjects


def train(epoch, class_weight=None):
    model.train()

    truth, probabilities = np.zeros((len(train_loader.dataset))), np.zeros((len(train_loader.dataset), 2))
    m = torch.nn.Softmax(dim=-1)
    m = m.cuda()
    if not class_weight is None:
        w = torch.FloatTensor(class_weight)
    else:
        w = None

    for batch_idx, (data, target, target_confidence) in enumerate(train_loader):
        n_in_batch = data.shape[0]
        truth[batch_idx * args.batch_size:batch_idx * args.batch_size + n_in_batch] = target

        target = target.type(torch.LongTensor)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            if not class_weight is None:
                w.cuda()
                w = Variable(w).type(torch.cuda.FloatTensor)

        optimizer.zero_grad()
        output = model(data)

        if not w is None:
            loss = nn.CrossEntropyLoss(w)
        else:
            loss = nn.CrossEntropyLoss()

        loss_val = loss(output, target)
        loss_val.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * args.batch_size, len(train_loader.dataset), 100. * batch_idx * args.batch_size / len(train_loader.dataset), loss_val.data.cpu().numpy()))

        output = m(output)

        probabilities[batch_idx * args.batch_size:batch_idx * args.batch_size + n_in_batch] = output.data.cpu().numpy()

    return truth, probabilities

def test_bags(loader, n_slices):
    model.eval()
    model.cuda()

    bag_model.eval()
    bag_model.cuda()

    m = torch.nn.Softmax(dim=-1)
    m = m.cuda()

    all_predictions = torch.zeros((len(loader), 1, n_slices*2), dtype=torch.float32)
    bag_predictions = torch.zeros((len(loader), 2), dtype=torch.float32)

    truth = np.zeros((len(loader)), dtype='uint8')

    for sample_idx, (data, target, sample_weight) in enumerate(loader):
        data = data.permute(1, 0, 2, 3)

        for slice_idx in range(n_slices * 2):
            slice = data[slice_idx:slice_idx + 1, ...]
            slice = slice.cuda()

            output = model(slice)
            slice_prediction = output[:, 0:1]
            slice_prediction = slice_prediction.permute(1, 0)
            slice_prediction = slice_prediction.data

            all_predictions[sample_idx, :, slice_idx] = slice_prediction

            truth[sample_idx] = target

    for sample_idx in range(len(loader)):
        slice_predictions = all_predictions[sample_idx, :, :]
        slice_predictions = slice_predictions.cuda()

        output = bag_model(slice_predictions)
        output = m(output)

        bag_predictions[sample_idx, :] = output.data

    return truth.data.cpu().numpy(), bag_predictions.data.cpu().numpy()

def test_slices(loader, n_slices, softmax=True):
    model.eval()

    all_predictions = torch.zeros((len(loader), n_slices*2, 2), dtype=torch.float32)
    truth = np.zeros((len(loader)), dtype='uint8')

    m = torch.nn.Softmax(dim=-1)
    m = m.cuda()

    for i, (data, target, sample_weight) in enumerate(loader):
        truth[i] = target
        data = data.permute(1, 0, 2, 3)
        for slice_idx in range(n_slices*2):
            slice = data[slice_idx:slice_idx+1, ...]
            slice = slice.cuda()

            output = model(slice)
            if softmax:
                output = m(output)

            all_predictions[i, slice_idx, :] = output.data

    return truth, all_predictions.data.cpu().numpy()


def learn_bag_distribution(train_loader_bag, validation_loader, n_slices, batch_size, n_epochs):
    model.eval()
    bag_model.train()
    bag_model.cpu()
    model.cuda()

    m = nn.Softmax(dim=-1)
    m = m.cuda()

    bag_optimizer = torch.optim.Adam(bag_model.parameters(), lr=0.0002)

    all_train_slice_predictions = torch.zeros((len(train_loader_bag), 1, n_slices*2), dtype=torch.float32)
    all_train_targets = torch.zeros((len(train_loader_bag)), dtype=torch.int64)
    all_train_sample_weights = torch.zeros((len(train_loader_bag)), dtype=torch.float32)

    all_validation_slice_predictions = torch.zeros((len(validation_loader), 1, n_slices*2), dtype=torch.float32)
    # all_test_slice_predictions = torch.zeros((len(test_loader), 1, n_slices*2), dtype=torch.float32)
    # all_ds030_slice_predictions = torch.zeros((len(ds030_loader), 1, n_slices*2), dtype=torch.float32)

    train_truth, train_probabilities = np.zeros(len(train_loader_bag), dtype='uint8'), np.zeros((len(train_loader_bag), 2), dtype='float32')
    validation_truth, validation_probabilities = np.zeros(len(validation_loader), dtype='uint8'), np.zeros((len(validation_loader), 2), dtype='float32')
    # test_truth, test_probabilities = np.zeros(len(test_loader), dtype='uint8'), np.zeros((len(test_loader), 2), dtype='float32')
    # ds030_truth, ds030_probabilities = np.zeros(len(ds030_loader), dtype='uint8'), np.zeros((len(ds030_loader), 2), dtype='float32')

    for sample_idx, (data, target, sample_weight) in enumerate(train_loader_bag):
        data = data.permute(1, 0, 2, 3)

        for slice_idx in range(n_slices * 2):
            slice = data[slice_idx:slice_idx + 1, ...]
            slice = slice.cuda()

            output = model(slice)
            slice_prediction = output[:, 0:1]
            slice_prediction = slice_prediction.permute(1, 0)
            slice_prediction = slice_prediction.data.cpu()

            all_train_slice_predictions[sample_idx, :, slice_idx] = slice_prediction

        all_train_targets[sample_idx] = target
        all_train_sample_weights[sample_idx] = sample_weight
        train_truth[sample_idx] = target

    print('Predicted all slices in training set(', len(train_loader_bag)*n_slices*2, 'total)')

    for sample_idx, (data, target, sample_weight) in enumerate(validation_loader):
        data = data.permute(1, 0, 2, 3)

        for slice_idx in range(n_slices * 2):
            slice = data[slice_idx:slice_idx + 1, ...]
            slice = slice.cuda()

            output = model(slice)
            slice_prediction = output[:, 0:1]
            slice_prediction = slice_prediction.permute(1, 0)
            slice_prediction = slice_prediction.data.cpu()

            all_validation_slice_predictions[sample_idx, :, slice_idx] = slice_prediction

        validation_truth[sample_idx] = target

    print('Predicted all slices in validation set')

    # for sample_idx, (data, target, sample_weight) in enumerate(test_loader):
    #     data = data.permute(1, 0, 2, 3)
    #
    #     for slice_idx in range(n_slices * 2):
    #         slice = data[slice_idx:slice_idx + 1, ...]
    #         slice = slice.cuda()
    #
    #         output = model(slice)
    #         slice_prediction = output[:, 0:1]
    #         slice_prediction = slice_prediction.permute(1, 0)
    #         slice_prediction = slice_prediction.data.cpu()
    #
    #         all_test_slice_predictions[sample_idx, :, slice_idx] = slice_prediction
    #
    #     test_truth[sample_idx] = target
    #
    # print('Predicted all slices in test set')
    #
    # for sample_idx, (data, target, sample_weight) in enumerate(ds030_loader):
    #     data = data.permute(1, 0, 2, 3)
    #
    #     for slice_idx in range(n_slices * 2):
    #         slice = data[slice_idx:slice_idx + 1, ...]
    #         slice = slice.cuda()
    #
    #         output = model(slice)
    #         slice_prediction = output[:, 0:1]
    #         slice_prediction = slice_prediction.permute(1, 0)
    #         slice_prediction = slice_prediction.data.cpu()
    #
    #         all_ds030_slice_predictions[sample_idx, :, slice_idx] = slice_prediction
    #
    #     ds030_truth[sample_idx] = target

    model.cpu()
    bag_model.cuda()

    print('Predicted all slices in ds030')
    print('Starting to train bag classifier...')

    for epoch_idx in range(n_epochs):
        print('Epoch', epoch_idx, 'of', n_epochs)
        for sample_idx in range(len(all_train_targets)):
            slice_predictions = all_train_slice_predictions[sample_idx, :, :]
            target = all_train_targets[sample_idx].unsqueeze(0)
            sample_weight = all_train_sample_weights[sample_idx]

            # print('slice predictions', slice_predictions.shape)
            # print('target', target.shape)

            slice_predictions = slice_predictions.cuda()
            target = target.cuda()
            sample_weight = sample_weight.cuda()

            output = bag_model(slice_predictions)
            # print('output', output.shape)

            loss = nn.CrossEntropyLoss()
            loss_val = loss(output, target)
            loss_val = loss_val * sample_weight
            loss_val.backward()

            if (sample_idx + 1) % batch_size == 0:
                bag_optimizer.step()
                bag_optimizer.zero_grad()
                print('Bag classifier training epoch:', epoch_idx, '[', sample_idx, '/', len(train_loader_bag), ']\t Loss:', loss_val.data.cpu().numpy())

        bag_optimizer.step()
        bag_optimizer.zero_grad()

    bag_model.eval()

    for sample_idx in range(len(all_train_targets)):
        slice_predictions = all_train_slice_predictions[sample_idx, :, :]

        slice_predictions = slice_predictions.cuda()

        output = bag_model(slice_predictions)
        output = m(output)

        train_probabilities[sample_idx, :] = output.data.cpu().numpy()

    for sample_idx in range(len(validation_loader)):
        slice_predictions = all_validation_slice_predictions[sample_idx, :, :]
        slice_predictions = slice_predictions.cuda()

        output = bag_model(slice_predictions)
        output = m(output)

        validation_probabilities[sample_idx, :] = output.data.cpu().numpy()

    # for sample_idx in range(len(test_loader)):
    #     slice_predictions = all_test_slice_predictions[sample_idx, :, :]
    #     slice_predictions = slice_predictions.cuda()
    #
    #     output = bag_model(slice_predictions)
    #     output = m(output)
    #
    #     test_probabilities[sample_idx, :] = output.data.cpu().numpy()
    #
    # for sample_idx in range(len(ds030_loader)):
    #     slice_predictions = all_ds030_slice_predictions[sample_idx, :, :]
    #     slice_predictions = slice_predictions.cuda()
    #
    #     output = bag_model(slice_predictions)
    #     output = m(output)
    #
    #     ds030_probabilities[sample_idx, :] = output.data.cpu().numpy()

    # return (train_truth, train_probabilities), (validation_truth, validation_probabilities), (test_truth, test_probabilities), (ds030_truth, ds030_probabilities)
    return (train_truth, train_probabilities), (validation_truth, validation_probabilities)


def set_temperature(model, f, validation_indices, n_slices):
    """
    Tune the temperature of the model (using the validation set).
    We're going to set it to optimize NLL.
    valid_loader (DataLoader): validation set loader
    """
    model.cuda()
    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = ECELoss().cuda()

    images = f['MRI']
    labels = f['qc_label']

    # First: collect all the logits and labels for the validation set
    logits_list, labels_list = [], []
    for i, val_idx in enumerate(validation_indices):
        target = torch.LongTensor([int(labels[val_idx])])
        data = torch.FloatTensor(n_slices*2, 1, image_shape[1], image_shape[2]).pin_memory()

        for j in range(n_slices*2):
            data[j, 0, :, :] = torch.from_numpy(images[val_idx, 0, image_shape[0] // 2 - n_slices + j, :, :])
            labels_list.append(target)

        input_var = Variable(data).cuda()
        # print('input var:', input_var.shape)
        logits_var = model(input_var)
        logits_list.append(logits_var.data)

    logits = torch.cat(logits_list).cuda()
    labels = torch.cat(labels_list).cuda()
    logits_var = Variable(logits)
    labels_var = Variable(labels)
    # print('logits, labels', logits_var, labels_var)
    # print(logits_var.shape, labels_var.shape)

    # Calculate NLL and ECE before temperature scaling
    before_temperature_nll = nll_criterion(logits_var, labels_var).data.cpu().numpy()
    before_temperature_ece = ece_criterion(logits_var, labels_var).data.cpu().numpy()
    print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

    # Next: optimize the temperature w.r.t. NLL
    optimizer = optim.LBFGS([model.temperature], lr=0.02, max_iter=100)
    def eval():
        loss = nll_criterion(model.temperature_scale(logits_var), labels_var)
        loss.backward()
        return loss
    optimizer.step(eval)

    # Calculate NLL and ECE after temperature scaling
    after_temperature_nll = nll_criterion(model.temperature_scale(logits_var), labels_var).data.cpu().numpy()
    after_temperature_ece = ece_criterion(model.temperature_scale(logits_var), labels_var).data.cpu().numpy()
    print('Optimal temperature: %.3f' % model.temperature.data.cpu().numpy())
    print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

    return model


if __name__ == '__main__':
    print('PyTorch implementation of DeepMRIQC.')
    start_time = time.time()

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch DeepMRIQC training.')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--val-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for validation (default: 32')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--slice-epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train at the slice level (default: 200)')
    parser.add_argument('--bag-epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train at the bag level (default: 20)')
    parser.add_argument('--folds', type=int, default=10, metavar='N',
                        help='number of folds to cross-validate over (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status (default: 5)')
    parser.add_argument('--n-slices', type=int, default=50, metavar='N',
                        help='specifies how many slices to include about the centre for testing (default: 50)')
    parser.add_argument('--gpu', type=int, default=0, metavar='N',
                        help='specifies which GPU to use')
    parser.add_argument('--no-scheduler', action='store_false', default=False,
                        help='disable learning rate scheduling')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print('Using GPU:', str(args.gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # densenet parameters
    network_input = (1,) + (image_shape[1],) + (image_shape[2],)    # channels, width, height
    growth_rate = 4
    depth = 100
    reduction = 0.5
    bottleneck=True

    model = densenet.DenseNet(input_shape=network_input, growthRate=growth_rate, depth=depth, reduction=reduction, bottleneck=bottleneck, nClasses=2)
    if args.cuda:
        model.cuda()
    # model = BigConvolutionalQCNet(input_shape=(1,) + (image_shape[1],) + (image_shape[2],))

    print('Parameters:', sum([p.data.nelement() for p in model.parameters()]))

    results_dir, experiment_number = setup_experiment(workdir)

    abide_f = h5py.File(workdir + abide_filename, 'r')
    ds030_f = h5py.File(workdir + ds030_filename, 'r')

    abide_indices = pickle.load(open(workdir + 'abide_indices.pkl', 'rb'))
    ds030_indices = pickle.load(open(workdir + 'ds030_indices.pkl', 'rb'))

    # np.random.shuffle(abide_indices)

    labels = abide_f['qc_label'][abide_indices]

    n_total = len(abide_indices)

    n_folds = args.folds
    n_slices = args.n_slices
    n_slice_strategies = 2

    results_shape = (n_folds, args.slice_epochs)

    training_sensitivity, training_specificity = np.zeros(results_shape), np.zeros(results_shape)

    results_shape = (n_folds, args.slice_epochs, n_slice_strategies)

    validation_sensitivity, validation_specificity, test_sensitivity, test_specificity, val_aucs = np.zeros(results_shape), np.zeros(results_shape), np.zeros(
        results_shape), np.zeros(results_shape), np.zeros(results_shape)

    ds030_sensitivity, ds030_specificity = np.zeros(results_shape)
    ds030_results = np.zeros(n_folds, 4)

    best_auc_score, best_sensitivity, best_specificity = np.zeros(n_folds), np.zeros((n_folds, 7)), np.zeros((n_folds, 7))
    best_sens_spec_score = np.zeros((n_folds))

    all_test_truth, all_val_truth, all_ds030_truth, all_test_slice_avg_probs, all_test_slice_max_probs, all_val_slice_avg_probs, all_val_slice_max_probs, all_ds030_slice_avg_probs, all_ds030_slice_max_probs = [], [], [], [], [], [], [], [], []

    all_bagged_test_truth, all_bagged_val_truth, all_bagged_ds030_truth, all_bagged_test_probs, all_bagged_val_probs, all_bagged_ds030_probs = [], [], [], [], [], []

    if args.cuda:
        model.cuda()

    n_pass = np.sum(labels)
    n_fail = len(abide_indices) - n_pass

    print('ABIDE dataset has ' + str(len(abide_indices)) + ' images (' + str(n_pass) + ' PASS, ' + str(n_fail) + ' FAIL)')
    fail_weight = (n_pass / n_total)
    pass_weight = n_fail / n_total
    # print('Setting class weighting to ' + str(fail_weight) + ' for FAIL class and ' + str(
    #     pass_weight) + ' for PASS class')

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    best_epoch_idx = np.zeros((n_folds, 2), dtype='uint8')

    skf = StratifiedKFold(n_splits=n_folds)
    for fold_idx, (train_val_indices, test_indices) in enumerate(skf.split(abide_indices, labels)):
        fold_num = fold_idx + 1

        model = densenet.DenseNet(input_shape=network_input, growthRate=growth_rate, depth=depth, reduction=reduction,
                                  bottleneck=bottleneck, nClasses=2)

        if args.cuda:
            model.cuda()

        validation_indices = train_val_indices[::10]
        train_indices = list(set(train_val_indices) - set(validation_indices))

        train_labels = labels[list(train_indices)]
        validation_labels = labels[list(validation_indices)]
        test_labels = labels[list(test_indices)]

        n_train_pass = np.sum(train_labels)
        n_val_pass = np.sum(validation_labels)
        n_test_pass = np.sum(test_labels)

        n_train_fail = len(train_indices) - n_train_pass
        n_val_fail = len(validation_indices) - n_val_pass
        n_test_fail = len(test_indices) - n_test_pass

        print('Fold', fold_num, 'has', n_train_pass, 'pass images and', n_train_fail, 'fail images in the training set.')
        print('Fold', fold_num, 'has', n_val_pass, 'pass images and', n_val_fail, 'fail images in the validation set.')
        print('Fold', fold_num, 'has', n_test_pass, 'pass images and', n_test_fail, 'fail images in the test set.')

        class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)
        print('Class weights are:', class_weights)

        train_sample_weights = np.zeros((len(train_labels)))
        for i, label in enumerate(train_labels):
            if label == 1:
                train_sample_weights[i] = class_weights[1]
            else:
                train_sample_weights[i] = class_weights[0]

        train_sample_weights = torch.DoubleTensor(train_sample_weights)

        # optimizer = optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9, dampening=0, weight_decay=0, nesterov=True)
        if not args.no_scheduler:
            print('Using learning rate scheduler')
            scheduler = StepLR(optimizer, args.slice_epochs // 4, gamma=0.5)

        # set up DataSets and DataLoaders
        abide_f = h5py.File(workdir + 'abide.hdf5', 'r')
        ds030_f = h5py.File(workdir + ds030_filename, 'r')

        train_dataset = RandomSlicesQCDataset(abide_f, train_indices, n_slices=n_slices)

        train_dataset_bag = AllSlicesQCDataset(abide_f, train_indices, n_slices=n_slices)
        validation_dataset = AllSlicesQCDataset(abide_f, validation_indices, n_slices=n_slices)
        test_dataset = AllSlicesQCDataset(abide_f, test_indices, n_slices=n_slices)
        ds030_dataset = AllSlicesQCDataset(ds030_f, ds030_indices, n_slices=n_slices)

        sampler = WeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_sample_weights))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False, **kwargs)

        train_loader_bag = DataLoader(train_dataset_bag)
        validation_loader = DataLoader(validation_dataset)
        test_loader = DataLoader(test_dataset)
        ds030_loader = DataLoader(ds030_dataset)

        class_weights = [0.5, 0.5]

        for epoch_idx, epoch in enumerate(range(1, args.slice_epochs + 1)):
            epoch_start = time.time()

            if not args.no_scheduler:
                scheduler.step()

            # training results
            print('Training')
            train_truth, train_probabilities = train(epoch, class_weight=None)
            train_predictions = np.argmax(train_probabilities, axis=-1)                  # slicewise prediction

            # validation results
            print('Validation')
            val_truth, val_probabilities = test_slices(validation_loader, n_slices)
            val_average_probs = np.mean(val_probabilities, axis=1)                       # average of slice predictions

            val_maximum_probs = np.zeros_like(val_average_probs)
            val_maximum_probs[:, 0] = np.max(val_probabilities[:, :, 0], axis=1)
            val_maximum_probs[:, 1] = 1 - val_maximum_probs[:, 0]                        # worst slice prediction

            val_predictions_avg = np.argmax(val_average_probs, axis=-1)
            val_predictions_max = np.argmax(val_maximum_probs, axis=-1)

            # test results
            print('Test')
            test_truth, test_probabilities = test_slices(test_loader, n_slices)
            test_average_probs = np.mean(test_probabilities, axis=1)                     # average of slice predictions

            test_maximum_probs = np.zeros_like(test_average_probs)
            test_maximum_probs[:, 0] = np.max(test_probabilities[:, :, 0], axis=1)
            test_maximum_probs[:, 1] = 1 - test_maximum_probs[:, 0]                      # worst slice prediction

            test_predictions_avg = np.argmax(test_average_probs, axis=-1)
            test_predictions_max = np.argmax(test_maximum_probs, axis=-1)

            # test results
            print('ds030')
            ds030_truth, ds030_probabilities = test_slices(ds030_loader, n_slices)
            ds030_average_probs = np.mean(ds030_probabilities, axis=1)                     # average of slice predictions

            ds030_maximum_probs = np.zeros_like(test_average_probs)
            ds030_maximum_probs[:, 0] = np.max(ds030_probabilities[:, :, 0], axis=1)
            ds030_maximum_probs[:, 1] = 1 - ds030_maximum_probs[:, 0]                      # worst slice prediction

            ds030_predictions_avg = np.argmax(ds030_average_probs, axis=-1)
            ds030_predictions_max = np.argmax(ds030_maximum_probs, axis=-1)

            # plot intermediate results
            # ROC
            truths = [train_truth, val_truth, val_truth, test_truth, test_truth, ds030_truth, ds030_truth]
            probs = [train_probabilities, val_predictions_avg, val_predictions_max, test_predictions_avg, test_predictions_max, ds030_predictions_avg, ds030_predictions_max]

            plot_labels = ['Train', 'Val (slice avg)', 'Val (slice max)', 'Test (slice avg)', 'Test (slice max)', 'ds030 (slice avg)', 'ds030 (slice max)']

            aucs = plot_roc(truths, probs, plot_labels, results_dir, epoch, fold_num)
            print('AUCS:', aucs)

            # Training
            train_tn, train_fp, train_fn, train_tp = confusion_matrix(np.asarray(train_truth, dtype='uint8'), np.asarray(train_predictions, dtype='uint8')).ravel()
            print('Training TP:', train_tp, 'TN:', train_tn, 'FP:', train_fp, 'FN:', train_fn)

            # Validation
            # avg slice prediction
            val_tn, val_fp, val_fn, val_tp = confusion_matrix(np.asarray(val_truth, dtype='uint8'), np.asarray(val_predictions_avg, dtype='uint8')).ravel()
            print('Validation TP:', val_tp, 'TN:', val_tn, 'FP:', val_fp, 'FN:', val_fn)

            validation_sensitivity[fold_idx, epoch_idx, 0] = val_tp / (val_tp + val_fn + epsilon)
            validation_specificity[fold_idx, epoch_idx, 0] = val_tn / (val_tn + val_fp + epsilon)

            # max slice prediction
            val_tn, val_fp, val_fn, val_tp = confusion_matrix(np.asarray(val_truth, dtype='uint8'), np.asarray(val_predictions_max, dtype='uint8')).ravel()
            print('Validation TP:', val_tp, 'TN:', val_tn, 'FP:', val_fp, 'FN:', val_fn)

            validation_sensitivity[fold_idx, epoch_idx, 1] = val_tp / (val_tp + val_fn + epsilon)
            validation_specificity[fold_idx, epoch_idx, 1] = val_tn / (val_tn + val_fp + epsilon)

            # Test
            # avg slice prediction
            test_tn, test_fp, test_fn, test_tp = confusion_matrix(np.asarray(test_truth, dtype='uint8'), np.asarray(test_predictions_avg, dtype='uint8')).ravel()
            print('Testing TP:', test_tp, 'TN:', test_tn, 'FP:', test_fp, 'FN:', test_fn)

            test_sensitivity[fold_idx, epoch_idx, 0] = test_tp / (test_tp + test_fn + epsilon)
            test_specificity[fold_idx, epoch_idx, 0] = test_tn / (test_tn + test_fp + epsilon)

            test_tn, test_fp, test_fn, test_tp = confusion_matrix(np.asarray(test_truth, dtype='uint8'), np.asarray(test_predictions_max, dtype='uint8')).ravel()
            print('Testing TP:', test_tp, 'TN:', test_tn, 'FP:', test_fp, 'FN:', test_fn)

            test_sensitivity[fold_idx, epoch_idx, 1] = test_tp / (test_tp + test_fn + epsilon)
            test_specificity[fold_idx, epoch_idx, 1] = test_tn / (test_tn + test_fp + epsilon)

            # ds030
            # avg slice prediction
            ds030_tn, ds030_fp, ds030_fn, ds030_tp = confusion_matrix(np.asarray(ds030_truth, dtype='uint8'), np.asarray(ds030_predictions_avg, dtype='uint8')).ravel()
            print('ds030 TP:', ds030_tp, 'TN:', ds030_tn, 'FP:', ds030_fp, 'FN:', ds030_fn)

            ds030_sensitivity[fold_idx, epoch_idx, 0] = ds030_tp / (ds030_tp + ds030_fn + epsilon)
            ds030_specificity[fold_idx, epoch_idx, 0] = ds030_tn / (ds030_tn + ds030_fp + epsilon)

            ds030_tn, ds030_fp, ds030_fn, ds030_tp = confusion_matrix(np.asarray(ds030_truth, dtype='uint8'), np.asarray(ds030_predictions_max, dtype='uint8')).ravel()
            print('ds030 TP:', ds030_tp, 'TN:', ds030_tn, 'FP:', ds030_fp, 'FN:', ds030_fn)

            ds030_sensitivity[fold_idx, epoch_idx, 1] = ds030_tp / (ds030_tp + ds030_fn + epsilon)
            ds030_specificity[fold_idx, epoch_idx, 1] = ds030_tn / (ds030_tn + ds030_fp + epsilon)

            val_aucs[fold_idx, epoch_idx, 0] = aucs[1]
            val_aucs[fold_idx, epoch_idx, 1] = aucs[2]

            # print('Train sensitivity/specificity:', training_sensitivity[fold_idx, epoch_idx],
            #       training_specificity[fold_idx, epoch_idx])
            # print('Validation sensitivity/specificity:', validation_sensitivity[fold_idx, epoch_idx],
            #       validation_specificity[fold_idx, epoch_idx])
            # print('Test sensitivity/specificity:', test_sensitivity[fold_idx, epoch_idx],
            #       test_specificity[fold_idx, epoch_idx])

            auc_score = aucs[1]

            # sens_score = 0.6*validation_sensitivity[fold_idx, epoch_idx] + 0.4*training_sensitivity[fold_idx, epoch_idx]
            # spec_score = 0.6*validation_specificity[fold_idx, epoch_idx] + 0.4*training_specificity[fold_idx, epoch_idx]
            #
            # sens_spec_score = (sens_score + spec_score) / 2

            if auc_score > best_auc_score[fold_idx]:
                print('This epoch is the new best model on the train/validation set!')
                best_auc_score[fold_idx] = auc_score
                # best_sens_spec_score[fold_idx] = sens_spec_score

                best_epoch_idx[fold_idx] = epoch_idx

                best_sensitivity[fold_idx, 0] = training_sensitivity[fold_idx, epoch_idx]
                best_specificity[fold_idx, 0] = training_specificity[fold_idx, epoch_idx]

                best_sensitivity[fold_idx, 1] = validation_sensitivity[fold_idx, epoch_idx, 0]
                best_specificity[fold_idx, 1] = validation_specificity[fold_idx, epoch_idx, 0]

                best_sensitivity[fold_idx, 2] = validation_sensitivity[fold_idx, epoch_idx, 1]
                best_specificity[fold_idx, 2] = validation_specificity[fold_idx, epoch_idx, 1]

                best_sensitivity[fold_idx, 3] = test_sensitivity[fold_idx, epoch_idx, 0]
                best_specificity[fold_idx, 3] = test_specificity[fold_idx, epoch_idx, 0]

                best_sensitivity[fold_idx, 4] = test_sensitivity[fold_idx, epoch_idx, 1]
                best_specificity[fold_idx, 4] = test_specificity[fold_idx, epoch_idx, 1]

                best_sensitivity[fold_idx, 5] = ds030_sensitivity[fold_idx, epoch_idx, 0]
                best_specificity[fold_idx, 5] = ds030_specificity[fold_idx, epoch_idx, 0]

                best_sensitivity[fold_idx, 6] = ds030_sensitivity[fold_idx, epoch_idx, 1]
                best_specificity[fold_idx, 6] = ds030_specificity[fold_idx, epoch_idx, 1]

                torch.save(model.state_dict(), results_dir + 'qc_torch_fold_' + str(fold_num) + '.tch')

            epoch_elapsed = time.time() - epoch_start
            print('Epoch ' + str(epoch) + ' of fold ' + str(fold_num) + ' took ' + str(epoch_elapsed / 60) + ' minutes')


        # re-test images using best model this fold
        torch.cuda.empty_cache()

        model.load_state_dict(torch.load(results_dir + 'qc_torch_fold_' + str(fold_num) + '.tch'))
        model.cuda()

        val_truth, val_probabilities = test_slices(validation_loader, n_slices)
        test_truth, test_probabilities = test_slices(test_loader, n_slices)
        ds030_truth, ds030_probabilities = test_slices(ds030_loader, n_slices)

        val_average_probs = np.mean(val_probabilities, axis=1)

        val_maximum_probs = np.zeros_like(val_average_probs)
        val_maximum_probs[:, 0] = np.max(val_probabilities[:, :, 0], axis=1)
        val_maximum_probs[:, 1] = 1 - val_maximum_probs[:, 0]

        test_average_probs = np.mean(test_probabilities, axis=1)

        test_maximum_probs = np.zeros_like(test_average_probs)
        test_maximum_probs[:, 0] = np.max(test_probabilities[:, :, 0], axis=1)
        test_maximum_probs[:, 1] = 1 - test_maximum_probs[:, 0]

        ds030_average_probs = np.mean(ds030_probabilities, axis=1)

        ds030_maximum_probs = np.zeros_like(ds030_average_probs)
        ds030_maximum_probs[:, 0] = np.max(ds030_probabilities[:, :, 0], axis=1)
        ds030_maximum_probs[:, 1] = 1 - ds030_maximum_probs[:, 0]

        # print('ds030 truth shape:', ds030_truth.shape)
        # print('ds030 probabilities shape:', ds030_probabilities.shape)
        # ds030_predictions = np.argmax(np.mean(ds030_probabilities, axis=1), axis=-1)
        # print('ds030 predictions shape:', ds030_predictions.shape)
        #
        # (ds030_tn, ds030_fp, ds030_fn, ds030_tp) = confusion_matrix(ds030_truth, ds030_predictions).ravel()
        #
        # ds030_results[fold_idx, 0] = ds030_tp / (ds030_tp + ds030_fn + epsilon)
        # ds030_results[fold_idx, 1] = ds030_tn / (ds030_tn + ds030_fp + epsilon)
        # ds030_results[fold_idx, 2] = accuracy_score(ds030_truth, ds030_predictions)
        # ds030_results[fold_idx, 3] = roc_auc_score(ds030_truth, ds030_predictions)

        bag_model = BagDistributionModel(n_slices)
        bag_model.cuda()

        (train_truth_bag, train_probs_bag), (val_truth_bag, val_probs_bag) = learn_bag_distribution(train_loader_bag, validation_loader, n_slices, batch_size=32, n_epochs=args.bag_epochs)

        test_truth, test_probs_bag = test_bags(test_loader, n_slices)
        ds030_truth, ds030_probs_bag = test_bags(ds030_loader, n_slices)

        #calibrate model probability on validation set
        # model_with_temperature = ModelWithTemperature(bag_model)
        # model_with_temperature.cuda()
        # model_with_temperature = set_temperature(model_with_temperature, abide_f, validation_indices, n_slices)

        for i, val_idx in enumerate(validation_indices):
            all_val_slice_avg_probs.append(val_average_probs[i, ...])
            all_val_slice_max_probs.append(val_maximum_probs[i, ...])
            all_val_truth.append(val_truth[i, ...])

            all_bagged_val_probs.append(val_probs_bag[i, ...])
            all_bagged_val_truth.append(val_truth_bag[i, ...])

        for i, test_idx in enumerate(test_indices):
            all_test_slice_avg_probs.append(test_average_probs[i, ...])
            all_test_slice_max_probs.append(test_maximum_probs[i, ...])
            all_test_truth.append(test_truth[i, ...])

            all_bagged_test_probs.append(test_probs_bag[i, ...])
            all_bagged_test_truth.append(test_truth[i, ...])

        for i, ds030_idx in enumerate(ds030_indices):
            all_bagged_ds030_probs.append(ds030_probs_bag[i, ...])
            all_bagged_ds030_truth.append(ds030_truth[i, ...])

        model_filename = os.path.join(results_dir, 'bagged_qc_model_fold_' + str(fold_num) + '.tch')
        torch.save(bag_model, model_filename)

    # plot validation AUC
    plt.figure(figsize=(6, 4))
    for fold_idx in range(n_folds):
        if fold_idx == 0:
            label_avg = 'average'
            label_max = 'maximum'
        else:
            label_avg = None
            label_max = None
        plt.plot(val_aucs[fold_idx, :, 0], color='darkred', label=label_avg)
        plt.plot(val_aucs[fold_idx, :, 1], color='darkgreen', label=label_max)

    plt.xlabel('Epoch #')
    plt.ylabel('AUC')
    plt.legend(shadow=True, fancybox=True)
    plt.tight_layout()
    plt.savefig(results_dir + 'validation_aucs.png')

    # plot training dynamics at each epoch
    avg_senses = [training_sensitivity, validation_sensitivity[..., 0], test_sensitivity[..., 0], ds030_sensitivity[..., 0]]
    avg_specs = [training_specificity, validation_specificity[..., 0], test_specificity[..., 0], ds030_specificity[..., 0]]

    max_senses = [training_sensitivity, validation_sensitivity[..., 1], test_sensitivity[..., 1], ds030_sensitivity[..., 1]]
    max_specs = [training_specificity, validation_specificity[..., 1], test_specificity[..., 1], ds030_specificity[..., 1]]

    plot_labels = ['Training', 'Validation', 'Test', 'ds030']

    plot_sens_spec(avg_senses, avg_specs, plot_labels, best_epoch_idx, results_dir, 'average_slice_predictions')
    plot_sens_spec(max_senses, max_specs, plot_labels, best_epoch_idx, results_dir, 'max_slice_predictions')

    # plot_confidence(np.asarray(all_test_probs, dtype='float32'), np.asarray(all_test_probs_cal, dtype='float32'), np.asarray(all_test_truth, dtype='uint8'), results_dir)


    # plot results across folds
    all_val_truth = np.asarray(all_val_truth, dtype='float32')
    all_test_truth = np.asarray(all_test_truth, dtype='float32')
    all_ds030_truth = np.asarray(all_ds030_truth, dtype='float32')

    all_bagged_val_truth = np.asarray(all_bagged_val_truth, dtype='float32')
    all_bagged_test_truth = np.asarray(all_bagged_test_truth, dtype='float32')
    all_bagged_ds030_truth = np.asarray(all_bagged_ds030_truth, dtype='float32')

    all_val_slice_avg_probs = np.asarray(all_val_slice_avg_probs, dtype='float32')
    all_val_slice_max_probs = np.asarray(all_val_slice_max_probs, dtype='float32')

    all_test_slice_avg_probs = np.asarray(all_val_slice_avg_probs, dtype='float32')
    all_test_slice_max_probs = np.asarray(all_val_slice_max_probs, dtype='float32')

    all_ds030_slice_avg_probs = np.asarray(all_ds030_slice_avg_probs, dtype='float32')
    all_ds030_slice_max_probs = np.asarray(all_ds030_slice_max_probs, dtype='float32')

    all_bagged_val_probs = np.asarray(all_bagged_val_probs, dtype='float32')
    all_bagged_test_probs = np.asarray(all_bagged_test_probs, dtype='float32')
    all_bagged_ds030_probs = np.asarray(all_bagged_ds030_probs, dtype='float32')


    slice_ground_truth = [all_val_truth, all_test_truth, all_ds030_truth]
    bagged_ground_truth = [all_bagged_val_truth, all_bagged_test_truth, all_bagged_ds030_truth]

    slice_avg_probabilities = [all_val_slice_avg_probs, all_test_slice_avg_probs, all_ds030_slice_avg_probs]
    slice_max_probabilities = [all_val_slice_max_probs, all_test_slice_max_probs, all_ds030_slice_max_probs]
    bagged_probabilities = [all_bagged_val_probs, all_bagged_test_probs, all_bagged_ds030_probs]

    plot_labels = ['Val', 'Test', 'ds030']

    plot_roc(slice_ground_truth, slice_avg_probabilities, plot_labels, results_dir, -1, fold_num=-1, title='Average Slice Predictions', filename='average_slice_predictions_roc')
    plot_roc(slice_ground_truth, slice_max_probabilities, plot_labels, results_dir, -1, fold_num=-1, title='Max Slice Predictions', filename='max_slice_predictions_roc')
    plot_roc(bagged_ground_truth, bagged_probabilities, plot_labels, results_dir, -1, fold_num=-1, title='Learned Slice Distribution Predictions', filename='bagged_predictions')

    # sens/spec boxplot
    sens_slices_avg = [best_sensitivity[:, 0], best_sensitivity[:, 1], best_sensitivity[:, 3], best_sensitivity[:, 5]]
    sens_slices_max = [best_sensitivity[:, 0], best_sensitivity[:, 2], best_sensitivity[:, 4], best_sensitivity[:, 6]]

    spec_slices_avg = [best_specificity[:, 0], best_specificity[:, 1], best_specificity[:, 3], best_specificity[:, 5]]
    spec_slices_max = [best_specificity[:, 0], best_specificity[:, 2], best_specificity[:, 4], best_specificity[:, 6]]

    sens_spec_across_folds(sens_slices_avg, spec_slices_avg, ['Training', 'Validation', 'Testing', 'ds030'], results_dir, 'average_slice_sensspec_boxplot')
    sens_spec_across_folds(sens_slices_max, spec_slices_max, ['Training', 'Validation', 'Testing', 'ds030'], results_dir, 'maximum_slice_sensspec_boxplot')

    # pickle.dump(sens_plot, open(results_dir + 'best_sens.pkl', 'wb'))
    # pickle.dump(spec_plot, open(results_dir + 'best_spec.pkl', 'wb'))

    # grad_cam = GradCam(model=model, target_layer_names=['output'], use_cuda=args.cuda)

    # dummy_input = Variable(torch.randn(n_slices*2, 1, image_shape[1], image_shape[2]))
    #
    # input_names = ["coronal_slice"]
    # output_names = ["pass_fail"]

    # torch.onnx.export(model, dummy_input, results_dir + "deepqc.onnx", verbose=False)

    for fold in range(skf.get_n_splits()):
        make_roc_gif(results_dir, args.slice_epochs, fold + 1)

    time_elapsed = time.time() - start_time
    print('Whole experiment took', time_elapsed / (60*60), 'hours')
    print('This experiment was brought to you by the number:', experiment_number)