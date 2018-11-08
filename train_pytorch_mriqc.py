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

from qc_pytorch_models import ConvolutionalQCNet, BigConvolutionalQCNet

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

class QCDataset(Dataset):
    def __init__(self, f, all_indices, n_slices=40):
        self.images = f['MRI']
        self.labels = f['qc_label']

        self.n_subjects = len(all_indices)
        self.indices = np.zeros((self.n_subjects))

        self.n_slices = n_slices

        for i, index in enumerate(all_indices):
            self.indices[i] = index

    def __getitem__(self, index):
        good_index = self.indices[index]

        slice_modifier = np.random.randint(-self.n_slices, self.n_slices)

        label = self.labels[good_index]
        image_slice = self.images[good_index, :, image_shape[0] // 2 + slice_modifier, :, :]

        return image_slice, label

    def __len__(self):
        return self.n_subjects


def train(epoch, class_weight=None):
    model.train()

    truth, probabilities = np.zeros((len(train_loader.dataset))), np.zeros((len(train_loader.dataset), 2))
    m = torch.nn.Softmax(dim=-1)
    if not class_weight is None:
        w = torch.FloatTensor(class_weight)
    else:
        w = None

    for batch_idx, (data, target) in enumerate(train_loader):
        n_in_batch = data.shape[0]
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            if not class_weight is None:
                w.cuda()
                w = Variable(w).type(torch.cuda.FloatTensor)
        data, target = Variable(data), Variable(target).type(torch.cuda.LongTensor)
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

        truth[batch_idx * args.batch_size:batch_idx * args.batch_size + n_in_batch] = target.data.cpu().numpy()
        probabilities[batch_idx * args.batch_size:batch_idx * args.batch_size + n_in_batch] = output.data.cpu().numpy()

    return truth, probabilities


def test(f, test_indices, n_slices):
    model.eval()

    truth, probabilities = np.zeros(len(test_indices), dtype='uint8'), np.zeros((len(test_indices), n_slices*2, 2), dtype='float32')
    m = torch.nn.Softmax(dim=-1)

    images = f['MRI']
    labels = f['qc_label']

    data = torch.zeros((n_slices*2, 1, image_shape[1], image_shape[2]), dtype=torch.float32).pin_memory()
    target = torch.zeros((data.shape[0], 1), dtype=torch.int64).pin_memory()

    for i, test_idx in enumerate(test_indices):
        data[:, 0, ...] = torch.FloatTensor(images[test_idx, 0, image_shape[0] // 2 - n_slices : image_shape[0] // 2 + n_slices, ...])
        target[:, 0] = torch.LongTensor([int(labels[test_idx])])

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target).type(torch.cuda.LongTensor)

        output = model(data)
        output = m(output)

        truth[i] = target.data.cpu()[0, 0]
        probabilities[i, :, :] = output.data.cpu().numpy()

    return truth, probabilities


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
        data = torch.FloatTensor(1, 1, n_slices*2, image_shape[1], image_shape[2]).pin_memory()

        for j in range(n_slices*2):
            data[0, 0, j, :, :] = torch.from_numpy(images[val_idx, 0, image_shape[0] // 2 - n_slices + j, :, :])
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
    before_temperature_nll = nll_criterion(logits_var, labels_var).data.cpu().numpy()
    before_temperature_ece = ece_criterion(logits_var, labels_var).data.cpu().numpy()
    print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

    # Next: optimize the temperature w.r.t. NLL
    optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=50)
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
    parser.add_argument('--n-slices', type=int, default=50, metavar='N',
                        help='specifies how many slices to include about the centre for testing (default: 50)')
    parser.add_argument('--gpu', type=int, default=0, metavar='N',
                        help='specifies which GPU to use')
    parser.add_argument('--scheduler', type=bool, default=False, metavar='N',
                        help='whether to enable learning rate scheduling')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print('Using GPU:', str(args.gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    model = densenet.DenseNet(input_shape=(1,) + (image_shape[1],) + (image_shape[2],), growthRate=4, depth=32, reduction=0.5, bottleneck=True, nClasses=2)
    # model = BigConvolutionalQCNet(input_shape=(1,) + (image_shape[1],) + (image_shape[2],))

    print('Parameters:', sum([p.data.nelement() for p in model.parameters()]))

    results_dir, experiment_number = setup_experiment(workdir)

    abide_f = h5py.File(workdir + abide_filename, 'r')
    ds030_f = h5py.File(workdir + ds030_filename, 'r')

    abide_indices = pickle.load(open(workdir + 'abide_indices.pkl', 'rb'))
    ds030_indices = pickle.load(open(workdir + 'ds030_indices.pkl', 'rb'))

    np.random.shuffle(abide_indices)

    labels = np.copy(abide_f['qc_label'])

    n_total = len(abide_indices)

    n_folds = args.folds
    n_slices = args.n_slices

    results_shape = (n_folds, args.epochs)

    training_sensitivity, training_specificity, validation_sensitivity, validation_specificity, test_sensitivity, test_specificity, val_aucs = np.zeros(
        results_shape), np.zeros(results_shape), np.zeros(results_shape), np.zeros(results_shape), np.zeros(
        results_shape), np.zeros(results_shape), np.zeros(results_shape)

    ds030_results = np.zeros((n_folds, 4))

    best_auc_score, best_sensitivity, best_specificity = np.zeros(n_folds), np.zeros((n_folds, 3)), np.zeros((n_folds, 3))
    best_sens_spec_score = np.zeros((n_folds))

    all_test_truth, all_val_truth, all_test_probs, all_val_probs, all_test_probs_cal, all_val_probs_cal = [], [], [], [], [], []

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

    test_idx, val_idx = 0, 0
    best_epoch_idx = np.empty((n_folds), dtype='uint8')

    skf = StratifiedKFold(n_splits=n_folds)
    for fold_idx, (train_val_indices, test_indices) in enumerate(skf.split(abide_indices, labels)):
        fold_num = fold_idx + 1

        # model = ConvolutionalQCNet(input_shape=(1,) + (image_shape[1],) + (image_shape[2],))
        model = densenet.DenseNet(input_shape=(1,) + (image_shape[1],) + (image_shape[2],), growthRate=4, depth=32, reduction=0.5, bottleneck=True, nClasses=2)

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
        optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, dampening=0, weight_decay=0, nesterov=True)
        if args.scheduler:
            scheduler = StepLR(optimizer, args.epochs // 4)

        for epoch_idx, epoch in enumerate(range(1, args.epochs + 1)):
            epoch_start = time.time()

            if args.scheduler:
                scheduler.step()

            abide_f = h5py.File(workdir + 'abide.hdf5', 'r')
            train_dataset = QCDataset(abide_f, train_indices, n_slices=n_slices)

            sampler = WeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_sample_weights))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False,
                                                       **kwargs)

            class_weights = [0.5, 0.5]

            train_truth, train_probabilities = train(epoch, class_weight=None)
            train_predictions = np.argmax(train_probabilities, axis=-1)

            val_truth, val_probabilities = test(abide_f, validation_indices, n_slices)
            val_average_probs = np.mean(val_probabilities, axis=1)
            val_predictions = np.argmax(val_average_probs, axis=-1)

            test_truth, test_probabilities = test(abide_f, test_indices, n_slices)
            test_average_probs = np.mean(test_probabilities, axis=1)
            test_predictions = np.argmax(test_average_probs, axis=-1)

            # print('probs shape:', test_probabilities.shape, val_probabilities.shape)

            train_auc, val_auc, test_auc = plot_roc(train_truth, train_probabilities, val_truth, val_average_probs,
                                                    test_truth, test_average_probs, results_dir, epoch, fold_num)

            train_tn, train_fp, train_fn, train_tp = confusion_matrix(np.asarray(train_truth, dtype='uint8'), np.asarray(train_predictions, dtype='uint8')).ravel()
            print('Training TP:', train_tp, 'TN:', train_tn, 'FP:', train_fp, 'FN:', train_fn)

            val_tn, val_fp, val_fn, val_tp = confusion_matrix(np.asarray(val_truth, dtype='uint8'), np.asarray(val_predictions, dtype='uint8')).ravel()
            print('Validation TP:', val_tp, 'TN:', val_tn, 'FP:', val_fp, 'FN:', val_fn)

            test_tn, test_fp, test_fn, test_tp = confusion_matrix(np.asarray(test_truth, dtype='uint8'), np.asarray(test_predictions, dtype='uint8')).ravel()
            print('Testing TP:', test_tp, 'TN:', test_tn, 'FP:', test_fp, 'FN:', test_fn)

            # print('Calculating sensitivity/specificity...')

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

            auc_score = val_auc

            sens_score = 0.6*validation_sensitivity[fold_idx, epoch_idx] + 0.4*training_sensitivity[fold_idx, epoch_idx]
            spec_score = 0.6*validation_specificity[fold_idx, epoch_idx] + 0.4*training_specificity[fold_idx, epoch_idx]

            sens_spec_score = (sens_score + spec_score) / 2

            if auc_score > best_auc_score[fold_idx]:
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

            epoch_elapsed = time.time() - epoch_start
            print('Epoch ' + str(epoch) + ' of fold ' + str(fold_num) + ' took ' + str(epoch_elapsed / 60) + ' minutes')


        # test images using best model this fold
        model.load_state_dict(torch.load(results_dir + 'qc_torch_fold_' + str(fold_num) + '.tch'))
        model.eval()

        val_truth, val_probabilities = test(abide_f, validation_indices, n_slices)
        test_truth, test_probabilities = test(abide_f, test_indices, n_slices)
        ds030_truth, ds030_probabilities = test(ds030_f, ds030_indices, n_slices)

        print('ds030 truth shape:', ds030_truth.shape)
        print('ds030 probabilities shape:', ds030_probabilities.shape)
        ds030_predictions = np.argmax(np.mean(ds030_probabilities, axis=1), axis=-1)
        print('ds030 predictions shape:', ds030_predictions.shape)

        (ds030_tn, ds030_fp, ds030_fn, ds030_tp) = confusion_matrix(ds030_truth, ds030_predictions).ravel()

        ds030_results[fold_idx, 0] = ds030_tp / (ds030_tp + ds030_fn + epsilon)
        ds030_results[fold_idx, 1] = ds030_tn / (ds030_tn + ds030_fp + epsilon)
        ds030_results[fold_idx, 2] = accuracy_score(ds030_truth, ds030_predictions)
        ds030_results[fold_idx, 3] = roc_auc_score(ds030_truth, ds030_predictions)

        #calibrate model probability on validation set
        model_with_temperature = ModelWithTemperature(model)
        model = set_temperature(model_with_temperature, abide_f, validation_indices, n_slices)

        val_truth, val_probabilities_calibrated = test(abide_f, validation_indices, n_slices)
        test_truth, test_probabilities_calibrated = test(abide_f, test_indices, n_slices)

        for i, val_idx in enumerate(validation_indices):
            all_val_probs.append(val_probabilities[i, ...])
            all_val_truth.append(val_truth[i, ...])
            all_val_probs_cal.append(val_probabilities_calibrated[i, ...])

        for i, test_idx in enumerate(test_indices):
            all_test_probs.append(test_probabilities[i, ...])
            all_test_truth.append(test_truth[i, ...])
            all_test_probs_cal.append(test_probabilities_calibrated[i, ...])

        # all_val_probs[val_idx:val_idx+len(validation_indices), :, :] = val_probabilities
        # all_val_truth[val_idx:val_idx+len(validation_indices)] = val_truth
        # all_test_probs[test_idx:test_idx+len(test_indices), :, :] = test_probabilities
        # all_test_truth[test_idx:test_idx+len(test_indices)] = test_truth

        # print('val_ indices:', val_idx, val_idx + len(validation_indices))
        # print('test indices:', test_idx, test_idx + len(test_indices))

        # all_val_probs_calibrated[val_idx:val_idx + len(validation_indices), :, :] = val_probabilities_calibrated
        # all_test_probs_calibrated[test_idx:test_idx + len(test_indices), :, :] = test_probabilities_calibrated

        model_filename = os.path.join(results_dir, 'calibrated_qc_fold_' + str(fold_num) + '.tch')
        torch.save(model, model_filename)

        test_idx += len(test_indices)
        val_idx += len(validation_indices)

    plot_sens_spec(training_sensitivity, training_specificity,
                           validation_sensitivity, validation_specificity,
                           test_sensitivity, test_specificity, best_epoch_idx, results_dir)

    plot_confidence(np.asarray(all_test_probs, dtype='float32'), np.asarray(all_test_probs_cal, dtype='float32'), np.asarray(all_test_truth, dtype='uint8'), results_dir)

    plot_roc(None, None, np.asarray(all_val_truth, dtype='float32'), np.mean(np.asarray(all_val_probs, dtype='float32'), axis=1), np.asarray(all_test_truth, dtype='float32'), np.mean(np.asarray(all_test_probs, dtype='float32'), axis=1), results_dir, -1, fold_num=-1)
    plot_roc(None, None, np.asarray(all_val_truth, dtype='float32'), np.mean(np.asarray(all_val_probs_cal, dtype='float32'), axis=1), np.asarray(all_test_truth, dtype='float32'), np.mean(np.asarray(all_test_probs_cal, dtype='float32'), axis=1), results_dir, -2, fold_num=-1)

    sens_plot = [best_sensitivity[:, 0], best_sensitivity[:, 1], best_sensitivity[:, 2], ds030_results[:, 0]]
    spec_plot = [best_specificity[:, 0], best_specificity[:, 1], best_specificity[:, 2], ds030_results[:, 1]]

    print('Sensitivity')
    print('Average:', np.mean(best_sensitivity[:, 0]), np.mean(best_sensitivity[:, 1]), np.mean(best_sensitivity[:, 2]))
    print('Best:', np.max(best_sensitivity[:, 0]), np.max(best_sensitivity[:, 1]), np.max(best_sensitivity[:, 2]))

    print('Specificity')
    print('Average:', np.mean(best_specificity[:, 0]), np.mean(best_specificity[:, 1]), np.mean(best_specificity[:, 2]))
    print('Best:', np.max(best_specificity[:, 0]), np.max(best_specificity[:, 1]), np.max(best_specificity[:, 2]))
    print('(train, val, test)')

    print('Test sens/spec:', np.mean(best_sensitivity[:, 2]), np.mean(best_specificity[:, 2]))
    print('ds030:', np.mean(ds030_results[:, 0]), np.mean(ds030_results[:, 1]))
    # pickle.dump(sens_plot, open(results_dir + 'best_sens.pkl', 'wb'))
    # pickle.dump(spec_plot, open(results_dir + 'best_spec.pkl', 'wb'))

    sens_spec_across_folds(sens_plot, spec_plot, ['Training', 'Validation', 'Testing', 'ds030'], results_dir)

    # grad_cam = GradCam(model=model, target_layer_names=['output'], use_cuda=args.cuda)

    # dummy_input = Variable(torch.randn(n_slices*2, 1, image_shape[1], image_shape[2]))
    #
    # input_names = ["coronal_slice"]
    # output_names = ["pass_fail"]
    #
    # model = ConvolutionalQCNet(input_shape=(1,) + (image_shape[1],) + (image_shape[2],))
    # model.load_state_dict(torch.load(results_dir + 'qc_torch_fold_1.tch'))
    # model.eval()

    # torch.onnx.export(model, dummy_input, results_dir + "deepqc.onnx", verbose=False)

    for fold in range(skf.get_n_splits()):
        make_roc_gif(results_dir, args.epochs, fold + 1)

    time_elapsed = time.time() - start_time
    print('Whole experiment took', time_elapsed / (60*60), 'hours')
    print('This experiment was brought to you by the number:', experiment_number)