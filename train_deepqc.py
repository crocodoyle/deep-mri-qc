from __future__ import print_function
import argparse
import torch
# import torch.multiprocessing
# torch.multiprocessing.set_start_method('spawn')
from shutil import copyfile, SameFileError

import torch.nn as nn
import torch.optim as optim
import torch.onnx

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.autograd import Variable

from qc_pytorch_models import ConvolutionalQCNet

import h5py, pickle, os, time, sys
import numpy as np

from ml_experiment import setup_experiment
from visualizations import plot_roc, plot_sens_spec, make_roc_gif, GradCam, sens_spec_across_folds

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

from scipy.stats import entropy

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.cm as cm

workdir = '/home/users/adoyle/deepqc/'

abide_input_filename = 'abide.hdf5'
ds030_input_filename = 'ds030.hdf5'

image_shape = (189, 233, 197)

class QCDataset(Dataset):
    def __init__(self, f, all_indices, random_slice=False):
        self.images = f['MRI']
        self.labels = f['qc_label']

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

        label = self.labels[good_index, 1]
        image_slice = self.images[good_index, image_shape[0] // 2 + slice_modifier, :, :][np.newaxis, ...]

        return image_slice, label

    def __len__(self):
        return self.n_subjects


def train(epoch):
    model.train()

    truth, probabilities = np.zeros((len(train_loader.dataset))), np.zeros((len(train_loader.dataset), 2))

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target).type(torch.cuda.LongTensor)
        optimizer.zero_grad()
        output = model(data)
        # print('output', output.shape)
        # print('P(qc|mri):', np.exp(output.data.cpu().numpy()))
        # loss = nn.NLLLoss(weight=class_weight)
        loss = nn.NLLLoss()
        loss_val = loss(output, target)
        loss_val.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * args.batch_size, len(train_loader.dataset), 100. * batch_idx * args.batch_size / len(train_loader.dataset), loss_val.data[0]))

        # print('output shape', output.shape)
        # print('batch size', args.batch_size)
        # print('indices: ', batch_idx * args.batch_size, (batch_idx + 1) * args.batch_size)
        truth[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size] = target.data.cpu().numpy()
        probabilities[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size] = output.data.cpu().numpy()

        # train_loss += loss_val.data[0]  # sum up batch loss
        # pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    # train_loss /= len(train_loader.dataset)
    # print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    #     train_loss, correct, len(train_loader.dataset),
    #     100. * correct / len(train_loader.dataset)))

    return truth, probabilities


def validate():
    model.eval()
    validation_loss, correct = 0, 0

    truth, probabilities = np.zeros((len(validation_loader.dataset))), np.zeros((len(validation_loader.dataset), 2))

    for batch_idx, (data, target) in enumerate(validation_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data, volatile=True), Variable(target).type(torch.cuda.LongTensor)
        output = model(data)
        loss_function = nn.NLLLoss()

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


def test(f2):
    model.eval()

    images = f['MRI']
    labels = f['qc_label']

    truth, probabilities = np.zeros((images.shape[0])), np.zeros((images.shape[0], 20, 2))

    test_indices = range(images.shape[0])

    for i, test_idx in enumerate(test_indices):
        data = np.zeros((20, 1, image_shape[1], image_shape[2]))
        data[:, 0, ...] = images[test_idx, 0, image_shape[0] // 2 - 10 : image_shape[0] // 2 + 10, ...]
        # print('Test input shape:', data.shape)

        target = np.zeros((data.shape[0], 1))
        target[:, 0] = labels[test_idx]
        truth[i] = target[0, 0]
        # print('Test target shape:', target.shape)

        data = torch.FloatTensor(data)
        target = torch.LongTensor(target)

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target).type(torch.cuda.LongTensor)
        output = model(data)

        probabilities[i, :, :] = output.data.cpu().numpy()

    return truth, probabilities


def example_pass_fails(model, train_loader, test_loader, results_dir, grad_cam):
    model.eval()

    histogram = np.zeros((256))

    bins = np.linspace(0.0, 1.0, 257)

    os.makedirs(results_dir + '/imgs/', exist_ok=True)
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = Variable(data, volatile=True), Variable(target).type(torch.cuda.LongTensor)

        target_batch = target.data.cpu().numpy()
        image_batch = data.data.cpu().numpy()

        for img in image_batch:

            try:
                histo = np.histogram(img, bins=bins)
                histogram += histo[0]
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
                filename = results_dir + '/imgs/' + qc_decision + '_' + str(batch_idx) + '_img_' + str(
                    i) + '.png'
                plt.savefig(filename, bbox_inches='tight')
        except IndexError as e:
            print('Couldnt save one file')

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = Variable(data, volatile=True), Variable(target).type(torch.cuda.LongTensor)

        target_batch = target.data.cpu().numpy()
        image_batch = data.data.cpu().numpy()

        for img in image_batch:
            try:
                histo = np.histogram(img, bins=bins)
                histogram += histo[0]
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

    plt.figure()
    plt.plot(bins[:-1], histogram, lw=2)

    plt.title('histogram of grey values', fontsize='24')
    plt.tight_layout()
    plt.savefig(results_dir + 'histograms.png', bbox_inches='tight')


if __name__ == '__main__':
    print('PyTorch implementation of DeepMRIQC.')
    start_time = time.time()

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch DeepMRIQC training.')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for validation (default: 32')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=75, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--folds', type=int, default=10, metavar='N',
                        help='number of folds to cross-validate over (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status (default: 5)')
    parser.add_argument('--ssd', type=bool, default=True, metavar='N',
                        help='specifies to copy the input file to the home directory (default: True')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    model = ConvolutionalQCNet(input_shape=(1,) + (image_shape[1],) + (image_shape[2],))
    print("Convolutional QC Model")
    print(model)

    results_dir, experiment_number = setup_experiment(workdir)

    data_filename = workdir + abide_input_filename
    if args.ssd:
        new_data_filename = '/home/users/adoyle/deepqc/' + abide_input_filename
        try:
            if not os.path.isfile(new_data_filename):
                copyfile(data_filename, new_data_filename)
        except SameFileError:
            print('Data file already exists at ' + new_data_filename)
        data_filename = new_data_filename

    f = h5py.File(data_filename, 'r')
    abide_indices = list(range(f['MRI'].shape[0]))

    labels = np.copy(f['qc_label'])

    n_total = len(abide_indices)

    n_folds = args.folds
    results_shape = (args.folds, args.epochs)

    training_sensitivity, training_specificity, validation_sensitivity, validation_specificity, test_sensitivity, test_specificity, val_aucs = np.zeros(
        results_shape), np.zeros(results_shape), np.zeros(results_shape), np.zeros(results_shape), np.zeros(
        results_shape), np.zeros(results_shape), np.zeros(results_shape)

    best_auc_score, best_sensitivity, best_specificity = np.zeros(n_folds), np.zeros((n_folds, 3)), np.zeros((n_folds, 3))

    if args.cuda:
        model.cuda()

    n_pass = np.sum(labels)
    n_fail = len(abide_indices) - n_pass

    print('Whole dataset has ' + str(len(abide_indices)) + ' images ('+ str(n_pass) + ' PASS, ' + str(n_fail) + ' FAIL)')
    fail_weight = (n_pass / n_total)
    pass_weight = n_fail / n_total
    print('Setting class weighting to ' + str(fail_weight) + ' for FAIL class and ' + str(
        pass_weight) + ' for PASS class')

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    skf = StratifiedKFold(n_splits=n_folds)
    for fold_idx, (train_indices, validation_indices) in enumerate(skf.split(abide_indices, labels)):
        fold_num = fold_idx + 1

        model = ConvolutionalQCNet(input_shape=(1,) + (image_shape[1],) + (image_shape[2],))
        if args.cuda:
            model.cuda()

        train_labels = labels[list(train_indices)]
        validation_labels = labels[list(validation_indices)]
        # test_labels = labels[list(test_indices)]

        n_train_pass = np.sum(train_labels)
        n_val_pass = np.sum(validation_labels)
        # n_test_pass = np.sum(test_labels)

        n_train_fail = len(train_indices) - n_train_pass
        n_val_fail = len(validation_indices) - n_val_pass
        # n_test_fail = len(test_indices) - n_test_pass

        print('Fold', fold_num, 'has', n_train_pass, 'pass images and', n_train_fail, 'fail images in the training set.')
        print('Fold', fold_num, 'has', n_val_pass, 'pass images and', n_val_fail, 'fail images in the validation set.')
        # print('Fold', fold_num, 'has', n_test_pass, 'pass images and', n_test_fail, 'fail images in the test set.')

        train_sample_weights = np.zeros((len(train_labels)))
        for i, label in enumerate(train_labels):
            if label == 1:
                train_sample_weights[i] = pass_weight
            else:
                train_sample_weights[i] = fail_weight

        train_sample_weights = torch.DoubleTensor(train_sample_weights)

            # print('This fold has', str(len(train_loader.dataset)), 'training images and',
        #       str(len(validation_loader.dataset)), 'validation images and', str(len(test_loader.dataset)),
        #       'test images.')

        optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)

        for epoch_idx, epoch in enumerate(range(1, args.epochs + 1)):
            epoch_start = time.time()
            f = h5py.File(workdir + abide_input_filename, 'r')
            f2 = h5py.File(workdir + ds030_input_filename, 'r')
            train_dataset = QCDataset(f, train_indices, random_slice=True)

            sampler = WeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_sample_weights))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False,
                                                       **kwargs)

            train_truth, train_probabilities = train(epoch)
            train_predictions = np.argmax(train_probabilities, axis=-1)

            validation_dataset = QCDataset(f, validation_indices, random_slice=True)
            validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.val_batch_size,
                                                            shuffle=False, **kwargs)

            val_truth, val_probabilities = validate()
            val_predictions = np.argmax(val_probabilities, axis=-1)

            test_truth, test_probabilities = test(f2)
            test_entropies = np.zeros((test_truth.shape[0], 1))
            for idx in range(test_truth.shape[0]):
                test_entropies[idx, 0] = entropy(test_probabilities[idx, :, 1])
            test_average_probs = np.mean(test_probabilities, axis=1)
            test_predictions = np.argmax(test_average_probs, axis=-1)

            train_auc, val_auc, test_auc = plot_roc(train_truth, train_probabilities, val_truth, val_probabilities,
                                                    test_truth, test_average_probs, results_dir, epoch, fold_num)

            try:
                train_tn, train_fp, train_fn, train_tp = confusion_matrix(np.asarray(train_truth, dtype='uint8'), np.asarray(train_predictions, dtype='uint8')).ravel()
                print('Training TP:', train_tp, 'TN:', train_tn, 'FP:', train_fp, 'FN:', train_fn)
            except:
                print('ERROR: couldnt calculate confusion matrix in training, probably only one class predicted/present in ground truth.')

            try:
                val_tn, val_fp, val_fn, val_tp = confusion_matrix(np.asarray(val_truth, dtype='uint8'), np.asarray(val_predictions, dtype='uint8')).ravel()
                print('Validation TP:', val_tp, 'TN:', val_tn, 'FP:', val_fp, 'FN:', val_fn)
            except:
                print('ERROR: couldnt calculate confusion matrix in validation, probably only one class predicted/present in ground truth.')

            try:
                test_tn, test_fp, test_fn, test_tp = confusion_matrix(np.asarray(test_truth, dtype='uint8'), np.asarray(test_predictions, dtype='uint8')).ravel()
                print('Testing TP:', test_tp, 'TN:', test_tn, 'FP:', test_fp, 'FN:', test_fn)
            except:
                print('ERROR: couldnt calculate confusion matrix in testing, probably only one class predicted/present in ground truth.')

            print('Calculating sensitivity/specificity...')
            epsilon = 1e-5
            training_sensitivity[fold_idx, epoch_idx] = train_tp / (train_tp + train_fn + epsilon)
            training_specificity[fold_idx, epoch_idx] = train_tn / (train_tn + train_fp + epsilon)

            validation_sensitivity[fold_idx, epoch_idx] = val_tp / (val_tp + val_fn + epsilon)
            validation_specificity[fold_idx, epoch_idx] = val_tn / (val_tn + val_fp + epsilon)

            test_sensitivity[fold_idx, epoch_idx] = test_tp / (test_tp + test_fn + epsilon)
            test_specificity[fold_idx, epoch_idx] = test_tn / (test_tn + test_fp + epsilon)

            val_aucs[fold_idx, epoch_idx] = val_auc

            if val_auc + train_auc > best_auc_score[fold_idx]:
                print('This epoch is the new best model on the train/validation set!')
                print('Validation sensitivity/specificity:', validation_sensitivity[fold_idx, epoch_idx], validation_specificity[fold_idx, epoch_idx])
                print('Test sensitivity/specificity:', test_sensitivity[fold_idx, epoch_idx], test_specificity[fold_idx, epoch_idx])
                best_auc_score[fold_idx] = (val_auc + train_auc)

                best_sensitivity[fold_idx, 0] = training_sensitivity[fold_idx, epoch_idx]
                best_specificity[fold_idx, 0] = training_specificity[fold_idx, epoch_idx]

                best_sensitivity[fold_idx, 1] = validation_sensitivity[fold_idx, epoch_idx]
                best_specificity[fold_idx, 1] = validation_specificity[fold_idx, epoch_idx]

                best_sensitivity[fold_idx, 2] = test_sensitivity[fold_idx, epoch_idx]
                best_specificity[fold_idx, 2] = test_specificity[fold_idx, epoch_idx]

                torch.save(model.state_dict(), results_dir + 'qc_torch_fold_' + str(fold_num) + '.tch')

            epoch_elapsed = time.time() - epoch_start
            print('Epoch ' + str(epoch) + ' of fold ' + str(fold_num) + ' took ' + str(epoch_elapsed / 60) + ' minutes')

            continue
        try:
            plot_sens_spec(training_sensitivity[fold_idx, :], training_specificity[fold_idx, :],
                           validation_sensitivity[fold_idx, :], validation_specificity[fold_idx, :],
                           test_sensitivity[fold_idx, :], test_specificity[fold_idx, :], results_dir, fold_num)
        except:
            print('ERROR could not save sensitivity/specificity plot for epoch', epoch)

    # done training

    sens_plot = [best_sensitivity[:, 0], best_sensitivity[:, 1], best_sensitivity[:, 2]]
    spec_plot = [best_specificity[:, 0], best_specificity[:, 1], best_specificity[:, 2]]

    pickle.dump(sens_plot, workdir + 'best_sens.pkl')
    pickle.dump(spec_plot, workdir + 'best_spec.pkl')

    sens_spec_across_folds(sens_plot, spec_plot, results_dir)

    grad_cam = GradCam(model=model, target_layer_names=['output'], use_cuda=args.cuda)
    # example_pass_fails(model, train_loader, test_loader, results_dir, grad_cam)

    dummy_input = Variable(torch.randn(1, 1, 256, 224))

    input_names = ["coronal_slice"]
    output_names = ["pass_fail"]

    model = ConvolutionalQCNet(input_shape=(1,) + (image_shape[1],) + (image_shape[2],))
    model.load_state_dict(torch.load(results_dir + 'qc_torch_fold_1.tch'))
    model.eval()

    torch.onnx.export(model, dummy_input, results_dir + "ibis_qc_net_v1.onnx", verbose=True)

    for fold in range(skf.get_n_splits()):
        make_roc_gif(results_dir, args.epochs, fold + 1)

    time_elapsed = time.time() - start_time
    print('Whole experiment took', time_elapsed / (60*60), 'hours')

    print('This experiment was brought to you by the number:', experiment_number)

