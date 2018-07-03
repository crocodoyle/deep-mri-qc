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

from torch.optim.lr_scheduler import ReduceLROnPlateau
from temperature_scaling import ModelWithTemperature, ModelWithSoftmax, ECELoss

from qc_pytorch_models import ConvolutionalQCNet

import h5py, pickle, os, time, sys
import numpy as np

from ml_experiment import setup_experiment
from visualizations import plot_roc, plot_sens_spec, make_roc_gif, GradCam, sens_spec_across_folds, plot_confidence

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.cm as cm

workdir = '/data1/users/adoyle/IBIS/'
# workdir = '/home/users/adoyle/deepqc/'
input_filename = 'IBIS_QC.hdf5'

image_shape = (160, 256, 224)

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

        label = self.labels[good_index]
        image_slice = self.images[good_index, :, image_shape[0] // 2 + slice_modifier, :, :]

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

        loss = nn.CrossEntropyLoss()
        loss_val = loss(output, target)
        loss_val.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * args.batch_size, len(train_loader.dataset), 100. * batch_idx * args.batch_size / len(train_loader.dataset), loss_val.data[0]))

        truth[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size] = target.data.cpu().numpy()
        probabilities[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size] = output.data.cpu().numpy()

    return truth, probabilities


def test(f, test_indices):
    model.eval()

    truth, probabilities = np.zeros((len(test_indices))), np.zeros((len(test_indices), 20, 2))

    images = f['MRI']
    labels = f['qc_label']

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
        data, target = Variable(data), Variable(target).type(torch.cuda.LongTensor)
        output = model(data)

        probabilities[i, :, :] = output.data.cpu().numpy()

    return truth, probabilities


def set_temperature(model, f, validation_indices):
    """
    Tune the tempearature of the model (using the validation set).
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

        for j in range(20):
            data = torch.FloatTensor(images[val_idx, 0, image_shape[0] // 2 - 10 + j, ...][np.newaxis, np.newaxis, ...])
            input_var = Variable(data).cuda()
            logits_var = model(input_var)
            logits_list.append(logits_var.data)
            labels_list.append(target)

    logits = torch.cat(logits_list).cuda()
    labels = torch.cat(labels_list).cuda()
    logits_var = Variable(logits)
    labels_var = Variable(labels)
    # print('logits, labels', logits_var, labels_var)

    # Calculate NLL and ECE before temperature scaling
    before_temperature_nll = nll_criterion(logits_var, labels_var).data[0]
    before_temperature_ece = ece_criterion(logits_var, labels_var).data[0]
    print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

    # Next: optimize the temperature w.r.t. NLL
    optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=50)
    def eval():
        loss = nll_criterion(model.temperature_scale(logits_var), labels_var)
        loss.backward()
        return loss
    optimizer.step(eval)

    # Calculate NLL and ECE after temperature scaling
    after_temperature_nll = nll_criterion(model.temperature_scale(logits_var), labels_var).data[0]
    after_temperature_ece = ece_criterion(model.temperature_scale(logits_var), labels_var).data[0]
    print('Optimal temperature: %.3f' % model.temperature.data[0])
    print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

    return model

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
    parser.add_argument('--epochs', type=int, default=120, metavar='N',
                        help='number of epochs to train (default: 120)')
    parser.add_argument('--folds', type=int, default=10, metavar='N',
                        help='number of folds to cross-validate over (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status (default: 5)')
    parser.add_argument('--ssd', action='store_true', default=False,
                        help='specifies to copy the input data to the home directory (default: False)')
    parser.add_argument('--single-slice', action='store_true', default=False,
                        help='specifies to test classifier using only center slice (default: False)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    model = ConvolutionalQCNet(input_shape=(1,) + (image_shape[1],) + (image_shape[2],))

    results_dir, experiment_number = setup_experiment(workdir)

    data_filename = workdir + input_filename
    if args.ssd:
        new_data_filename = '/home/users/adoyle/IBIS/' + input_filename
        try:
            if not os.path.isfile(new_data_filename):
                copyfile(data_filename, new_data_filename)
        except SameFileError:
            print('Data file already exists at ' + new_data_filename)
        data_filename = new_data_filename

    f = h5py.File(data_filename, 'r')
    ibis_indices = list(range(f['MRI'].shape[0]))

    wrong_fails = []

    labels = np.copy(f['qc_label'])

    n_total = len(ibis_indices)

    n_folds = args.folds
    results_shape = (args.folds, args.epochs)

    training_sensitivity, training_specificity, validation_sensitivity, validation_specificity, test_sensitivity, test_specificity, val_aucs = np.zeros(
        results_shape), np.zeros(results_shape), np.zeros(results_shape), np.zeros(results_shape), np.zeros(
        results_shape), np.zeros(results_shape), np.zeros(results_shape)

    best_auc_score, best_sensitivity, best_specificity = np.zeros(n_folds), np.zeros((n_folds, 3)), np.zeros((n_folds, 3))
    best_sens_spec_score = np.zeros((n_folds))

    all_test_probs = np.zeros((n_total, 20, 2))
    all_test_truth = np.zeros((n_total))
    all_val_probs = np.zeros((n_total, 20, 2))
    all_val_truth = np.zeros((n_total))

    if args.cuda:
        model.cuda()

    n_pass = np.sum(labels)
    n_fail = len(ibis_indices) - n_pass

    print('Whole dataset has ' + str(len(ibis_indices)) + ' images (' + str(n_pass) + ' PASS, ' + str(n_fail) + ' FAIL)')
    fail_weight = (n_pass / n_total)
    pass_weight = n_fail / n_total
    print('Setting class weighting to ' + str(fail_weight) + ' for FAIL class and ' + str(
        pass_weight) + ' for PASS class')

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    test_idx, val_idx = 0, 0
    best_epoch_idx = np.empty((n_folds), dtype='uint8')

    skf = StratifiedKFold(n_splits=n_folds)
    for fold_idx, (train_indices, other_indices) in enumerate(skf.split(ibis_indices, labels)):
        fold_num = fold_idx + 1

        current_wrong_fails = []
        model = ConvolutionalQCNet(input_shape=(1,) + (image_shape[1],) + (image_shape[2],))
        if args.cuda:
            model.cuda()

        validation_indices = other_indices[::2]
        test_indices = other_indices[1::2]

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

        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)

        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, threshold=1e-5, patience=10, verbose=True)

        for epoch_idx, epoch in enumerate(range(1, args.epochs + 1)):
            epoch_start = time.time()
            f = h5py.File(workdir + input_filename, 'r')
            train_dataset = QCDataset(f, train_indices, random_slice=True)

            sampler = WeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_sample_weights))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False,
                                                       **kwargs)

            train_truth, train_probabilities = train(epoch)
            train_predictions = np.argmax(train_probabilities, axis=-1)

            val_truth, val_probabilities = test(f, validation_indices)
            val_average_probs = np.mean(val_probabilities, axis=1)
            val_predictions = np.argmax(val_average_probs, axis=-1)

            test_truth, test_probabilities = test(f, test_indices)
            test_average_probs = np.mean(test_probabilities, axis=1)
            test_predictions = np.argmax(test_average_probs, axis=-1)

            # print('probs shape:', test_probabilities.shape, val_probabilities.shape)

            train_auc, val_auc, test_auc = plot_roc(train_truth, train_probabilities, val_truth, val_average_probs,
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

            # print('Calculating sensitivity/specificity...')
            epsilon = 1e-6
            training_sensitivity[fold_idx, epoch_idx] = train_tp / (train_tp + train_fn + epsilon)
            training_specificity[fold_idx, epoch_idx] = train_tn / (train_tn + train_fp + epsilon)

            validation_sensitivity[fold_idx, epoch_idx] = val_tp / (val_tp + val_fn + epsilon)
            validation_specificity[fold_idx, epoch_idx] = val_tn / (val_tn + val_fp + epsilon)

            test_sensitivity[fold_idx, epoch_idx] = test_tp / (test_tp + test_fn + epsilon)
            test_specificity[fold_idx, epoch_idx] = test_tn / (test_tn + test_fp + epsilon)

            val_aucs[fold_idx, epoch_idx] = val_auc

            print('Train sensitivity/specificity:', training_sensitivity[fold_idx, epoch_idx],
                  training_specificity[fold_idx, epoch_idx])
            print('Validation sensitivity/specificity:', validation_sensitivity[fold_idx, epoch_idx],
                  validation_specificity[fold_idx, epoch_idx])
            print('Test sensitivity/specificity:', test_sensitivity[fold_idx, epoch_idx],
                  test_specificity[fold_idx, epoch_idx])

            auc_score = (val_auc / len(validation_indices)) / 2 + (train_auc / len(train_indices)) / 2

            sens_score = 0.1*validation_sensitivity[fold_idx, epoch_idx] + 0.9*training_sensitivity[fold_idx, epoch_idx]
            spec_score = 0.1*validation_specificity[fold_idx, epoch_idx] + 0.9*training_specificity[fold_idx, epoch_idx]
            sens_spec_score = 0.5*sens_score + 0.5*spec_score - 0.1*np.abs(sens_score - spec_score)

            if sens_spec_score > best_sens_spec_score[fold_idx]:
                print('This epoch is the new best model on the train/validation set!')
                best_auc_score[fold_idx] = auc_score
                best_sens_spec_score[fold_idx] = sens_spec_score

                best_epoch_idx[fold_idx] = epoch_idx

                best_sensitivity[fold_idx, 0] = training_sensitivity[fold_idx, epoch_idx]
                best_specificity[fold_idx, 0] = training_specificity[fold_idx, epoch_idx]

                best_sensitivity[fold_idx, 1] = validation_sensitivity[fold_idx, epoch_idx]
                best_specificity[fold_idx, 1] = validation_specificity[fold_idx, epoch_idx]

                best_sensitivity[fold_idx, 2] = test_sensitivity[fold_idx, epoch_idx]
                best_specificity[fold_idx, 2] = test_specificity[fold_idx, epoch_idx]

                torch.save(model.state_dict(), results_dir + 'qc_torch_fold_' + str(fold_num) + '.tch')

                current_wrong_fails = []
                for prediction, truth, idx in zip(test_predictions, test_truth, test_indices):
                    if truth == 0 and prediction == 1:
                        current_wrong_fails.append(f['filename'][idx, ...])

            epoch_elapsed = time.time() - epoch_start
            print('Epoch ' + str(epoch) + ' of fold ' + str(fold_num) + ' took ' + str(epoch_elapsed / 60) + ' minutes')


        wrong_fails += current_wrong_fails
        # test images using best model this fold
        model.load_state_dict(torch.load(results_dir + 'qc_torch_fold_' + str(fold_num) + '.tch'))
        model.eval()

        val_truth, val_probabilities = test(f, validation_indices)
        test_truth, test_probabilities = test(f, test_indices)
        # print('last test this epoch:', test_probabilities)
        # print('prob shape:', test_probabilities.shape)

        all_val_probs[val_idx:val_idx+len(validation_indices), :, :] = val_probabilities
        all_val_truth[val_idx:val_idx+len(validation_indices)] = val_truth
        all_test_probs[test_idx:test_idx+len(test_indices), :, :] = test_probabilities
        all_test_truth[test_idx:test_idx+len(test_indices)] = test_truth

        test_idx += len(test_indices)
        val_idx += len(validation_indices)

        model_with_temperature = ModelWithTemperature(model)
        model_with_temperature = set_temperature(model_with_temperature, f, validation_indices)

        model = ModelWithSoftmax(model_with_temperature)

        model_filename = os.path.join(results_dir, 'calibrated_qc_fold_' + str(fold_num) + '.tch')
        torch.save(model.state_dict(), model_filename)

    plot_sens_spec(training_sensitivity, training_specificity,
                           validation_sensitivity, validation_specificity,
                           test_sensitivity, test_specificity, best_epoch_idx, results_dir)

    plot_confidence(all_test_probs, all_test_truth, results_dir)

    sens_plot = [best_sensitivity[:, 0], best_sensitivity[:, 1], best_sensitivity[:, 2]]
    spec_plot = [best_specificity[:, 0], best_specificity[:, 1], best_specificity[:, 2]]

    print('Sensitivity')
    print('Average:', np.mean(best_sensitivity[:, 0]), np.mean(best_sensitivity[:, 1]), np.mean(best_sensitivity[:, 2]))
    print('Best:', np.max(best_sensitivity[:, 0]), np.max(best_sensitivity[:, 1]), np.max(best_sensitivity[:, 2]))

    print('Specificity')
    print('Average:', np.mean(best_specificity[:, 0]), np.mean(best_specificity[:, 1]), np.mean(best_specificity[:, 2]))
    print('Best:', np.max(best_specificity[:, 0]), np.max(best_specificity[:, 1]), np.max(best_specificity[:, 2]))
    print('(train, val, test)')

    pickle.dump(sens_plot, open(results_dir + 'best_sens.pkl', 'wb'))
    pickle.dump(spec_plot, open(results_dir + 'best_spec.pkl', 'wb'))

    sens_spec_across_folds(sens_plot, spec_plot, results_dir)

    # grad_cam = GradCam(model=model, target_layer_names=['output'], use_cuda=args.cuda)

    dummy_input = Variable(torch.randn(20, 1, image_shape[1], image_shape[2]))

    input_names = ["coronal_slice"]
    output_names = ["pass_fail"]

    model = ConvolutionalQCNet(input_shape=(1,) + (image_shape[1],) + (image_shape[2],))
    model.load_state_dict(torch.load(results_dir + 'qc_torch_fold_1.tch'))
    model.eval()

    torch.onnx.export(model, dummy_input, results_dir + "ibis_qc_net_v1.onnx", verbose=True)

    for fold in range(skf.get_n_splits()):
        make_roc_gif(results_dir, args.epochs, fold + 1)

    print('Fails where classifier made wrong prediction:')
    for wrong_fail in wrong_fails:
        print(wrong_fail)

    pickle.dump(wrong_fails, open(results_dir + 'wrong_fails.pkl', 'wb'))

    time_elapsed = time.time() - start_time
    print('Whole experiment took', time_elapsed / (60*60), 'hours')
    print('This experiment was brought to you by the number:', experiment_number)

