import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

import imageio, os

import torch
from torch.autograd import Variable, Function

import numpy as np

from sklearn.calibration import calibration_curve


def make_roc_gif(results_dir, epochs, fold_num=1):
    images = []

    epoch_range = range(1, epochs+1)
    for epoch in epoch_range:
        filename = results_dir + 'ROC_fold_' + str(fold_num) + '_epoch_' + str(epoch).zfill(2) + '.png'
        img = plt.imread(filename)
        images.append(img)

    imageio.mimsave(results_dir + 'ROC_fold_' + str(fold_num) + '.gif', images)


def plot_roc(ground_truth, probabilities, segment_labels, results_dir, epoch_num=-1, fold_num=-1, title=None, filename=None):
    plt.figure(figsize=(8, 8))

    colors = ['gold', 'darkorange', 'peru', 'hotpink', 'purple', 'red', 'darkred']

    lw = 2
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    aucs = []
    for i, (truth, probs, label) in enumerate(zip(ground_truth, probabilities, segment_labels)):
        print(truth.shape, probs.shape, label)
        roc = roc_auc_score(truth, probs[:, 1], 'weighted')
        fpr, tpr, _ = roc_curve(truth, probs[:, 1])
        plt.plot(fpr, tpr, color=colors[i], lw=lw, label=label + ' ROC (AUC %0.2f)' % roc)

        aucs.append(roc)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)

    if not title is None:
        plt.title(title)

    if not filename is None:
        save_filename = filename

    if epoch_num >= 0:
        plt.title('ROC Epoch:' + str(epoch_num).zfill(3), fontsize=24)
        save_filename ='ROC_fold_' + str(fold_num) + '_epoch_' + str(epoch_num).zfill(2) + '.png'

    plt.legend(loc="lower right", shadow=True, fontsize=20)

    plt.savefig(results_dir + save_filename, bbox_inches='tight')
    plt.close()

    return aucs


def plot_sens_spec(senses, specs, curve_labels, best_epoch_idx, results_dir, title):
    f, (sens_ax, spec_ax) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))

    n_folds = senses[0].shape[0]
    n_epochs = senses[0].shape[-1]
    epoch_number = np.asarray(range(n_epochs))

    colours = ['pink', 'blue', 'lightblue', 'green', 'lightgreen', 'hotpink', 'purple']

    lw = 1

    for fold_num in range(n_folds):
        for i, (sens, spec, label, colour) in enumerate(zip(senses, specs, curve_labels, colours)):

            # test and ds030 are probably sparsely populated
            sens = sens[fold_num, :][np.nonzero(sens[fold_num, :])]
            spec = spec[fold_num, :][np.nonzero(spec[fold_num, :])]
            sparse_epochs = epoch_number[np.nonzero(sens[fold_num, :])]

            if not fold_num == 0:
                label = None
                marker_label = None
            else:
                marker_label = 'selected model'

            sens_ax.plot(sparse_epochs, sens, color=colour, lw=lw, label=label)
            spec_ax.plot(sparse_epochs, spec, color=colour, linestyle=':', lw=lw, label=label)


        validation_sens = senses[1]
        validation_spec = specs[1]

        best_epoch = best_epoch_idx[fold_num]

        sens_ax.plot(best_epoch, validation_sens[fold_num, best_epoch], color='k', marker='o', markerfacecolor='None', label=marker_label)
        spec_ax.plot(best_epoch, validation_spec[fold_num, best_epoch], color='k', marker='o', markerfacecolor='None', label=marker_label)

    sens_ax.set_xlim([0, n_epochs])
    spec_ax.set_xlim([0, n_epochs])

    spec_ax.set_xlabel('Epoch #', fontsize=20)
    sens_ax.set_ylabel('Sensitivity', fontsize=20)
    spec_ax.set_ylabel('Specificity', fontsize=20)

    sens_ax.legend(shadow=True, fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))
    spec_ax.legend(shadow=True, fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(results_dir + title + '.png', bbox_inches='tight')
    plt.close()


def sens_spec_across_folds(sens_to_plot, spec_to_plot, labels, results_dir, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 4))

    bplot1 = ax1.boxplot(sens_to_plot, patch_artist=True)
    bplot2 = ax2.boxplot(spec_to_plot, patch_artist=True)

    colors = ['pink', 'lightblue', 'lightgreen', 'peachpuff']
    for bplot in (bplot1, bplot2):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    ax1.set_ylim([0, 1])
    ax2.set_ylim([0, 1])

    ax1.set_xticklabels(labels, fontsize=16)
    ax2.set_xticklabels(labels, fontsize=16)

    ax1.set_title('Sensitivity', fontsize=24)
    ax2.set_title('Specificity', fontsize=24)

    ax1.grid()
    ax2.grid()
    plt.savefig(results_dir + filename + '.png', dpi=500)
    plt.close()


def plot_confidence(probabilities, probabilities_calibrated, truth, results_dir):
    print('probs range:', np.min(probabilities), np.max(probabilities))
    print('calibrated probs range:', np.min(probabilities_calibrated), np.max(probabilities_calibrated))
    # print(probabilities.shape, probabilities_calibrated.shape)

    n_subjects = probabilities.shape[0]
    n_slices = probabilities.shape[1]

    print('We have', n_subjects, 'subjects and are testing', n_slices, 'slices')

    pass_confidence, fail_confidence = [], []
    tp_confidence, tn_confidence, fp_confidence, fn_confidence = [], [], [], []

    f, (passfail_ax, confidence_ax, confusion_ax) = plt.subplots(1, 3, figsize=(10, 4), gridspec_kw = {'width_ratios':[1, 3, 3]})

    y_prob = np.mean(probabilities[:, :, 1], axis=1)
    y_conf = []

    for i in range(n_subjects):
        confidence = 0
        for j in range(n_slices):
            if probabilities[i, j, 1] > 0.5:
                confidence += 1 / n_slices

        if y_prob[i] < 0.5:
            y_conf.append(1 - confidence)
        else:
            y_conf.append(confidence)

        if i%20 == 0:
            print('Truth, Pass Prob, Fail Prob, Conf:', truth[i], y_prob[i], np.mean(probabilities[i, :, 0]), y_conf[i])

        if truth[i] > 0.5:
            pass_confidence.append(y_conf[i])
            if y_prob[i] > 0.5:
                tp_confidence.append(y_conf[i])
            else:
                fn_confidence.append(y_conf[i])
        else:
            fail_confidence.append(y_conf[i])
            if y_prob[i] > 0.5:
                fp_confidence.append(y_conf[i])
            else:
                tn_confidence.append(y_conf[i])

    n_pass = np.sum(truth)
    n_fail = len(truth) - n_pass

    print('n_pass, n_fail', n_pass, n_fail)

    passfail_ax.bar([1], [int(n_fail)], width=0.85, tick_label=['FAIL'], color='darkred')
    passfail_ax.bar([2], [int(n_pass)], width=0.85, tick_label=['PASS'], color='darkgreen')

    plt.sca(confidence_ax)
    plt.xticks([0, 1, 2, 3], ['', 'FAIL', 'PASS', ''], fontsize=16)

    # passfail_ax.set_xlabel('QC Label')

    bins = np.linspace(0, 1, num=n_slices+1, endpoint=True)

    pass_hist, bin_edges = np.histogram(pass_confidence, bins)
    fail_hist, bin_edges = np.histogram(fail_confidence, bins)

    tp_hist, bin_edges = np.histogram(tp_confidence, bins)
    fn_hist, bin_edges = np.histogram(fn_confidence, bins)
    fp_hist, bin_edges = np.histogram(fp_confidence, bins)
    tn_hist, bin_edges = np.histogram(tn_confidence, bins)

    width = 0.02

    b1 = confidence_ax.bar(bin_edges[:-1]+0.025, pass_hist, width, color='darkgreen')
    b2 = confidence_ax.bar(bin_edges[:-1]+0.05, fail_hist, width, color='darkred')

    # b3 = confusion_ax.bar(bin_edges[:-1]+0.025, tp_hist, width/2, color='green')
    # b4 = confusion_ax.bar(bin_edges[:-1]+0.05, tn_hist, width/2, color='red')
    b5 = confusion_ax.bar(bin_edges[:-1]+0.025, fn_hist, width, color='purple')
    b6 = confusion_ax.bar(bin_edges[:-1]+0.05, fp_hist, width, color='darkorange')

    passfail_ax.set_ylabel('# Images', fontsize=20)
    confidence_ax.set_xlabel('Confidence', fontsize=20)
    confusion_ax.set_xlabel('Confidence', fontsize=20)
    # confusion_ax.set_ylabel('# Images', fontsize=20)

    # passfail_ax.set_xlim([0.95, 2.05])
    confidence_ax.set_xlim([-0.05, 1.05])
    confusion_ax.set_xlim([-0.05, 1.05])

    bins_for_display = np.linspace(0, 1, num=4+1, endpoint=True)
    bins_display = []
    for i, bin in enumerate(bins_for_display):
        bins_display.append("%.2f" % round(bin, 2))

    plt.sca(confidence_ax)
    plt.xticks(bins_for_display, bins_display)

    plt.sca(confusion_ax)
    plt.xticks(bins_for_display, bins_display)

    # confidence_ax.set_xticklabels(['%s' % float('%.2g' % bin_edge) for bin_edge in bins[:-1]])
    # confusion_ax.set_xticklabels(['%s' % float('%.2g' % bin_edge) for bin_edge in bins[:-1]])

    lgd = plt.legend([b1, b2, b5, b6], ['Pass', 'Fail', 'False Negative', 'False Positive'], shadow=True, fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(results_dir + 'confidence.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=500)
    plt.close()

    f, (calib_ax) = plt.subplots(1, 1, sharey=True, figsize=(8, 4))

    # print('ground truth shape', truth.shape)
    # print(truth)
    y_prob = np.mean(probabilities[:, :, 1], axis=1)
    # print('probs shape', y_prob.shape)
    # print(y_prob)
    y_calib = np.mean(probabilities_calibrated[:, :, 1], axis=1)
    # print('calib probs shape', y_calib.shape)
    # print(y_calib)

    fraction_of_positives, mean_predicted_value = calibration_curve(truth, y_prob, n_bins=10)
    fraction_of_positives_calibrated, mean_predicted_value_calibrated = calibration_curve(truth, y_calib, n_bins=10)

    calib_ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    calib_ax.plot(mean_predicted_value, fraction_of_positives, "s-", label='Uncalibrated')
    calib_ax.plot(mean_predicted_value_calibrated, fraction_of_positives_calibrated, "s-", label='Temperature Scaled')

    calib_ax.set_ylabel('Accuracy', fontsize=20)
    calib_ax.set_xlabel('Confidence', fontsize=20)

    lgd = calib_ax.legend(shadow=True, fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(results_dir + 'reliability.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=500)



# code below here from https://github.com/jacobgil/pytorch-grad-cam/blob/master/grad-cam.py
class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermediate targeted layers.
	3. Gradients from intermediate targeted layers. """
	def __init__(self, model, target_layers):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output  = self.feature_extractor(x)
		output = output.view(output.size(0), -1)
		output = self.model.classifier(output)
		return target_activations, output

class GradCam:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names)

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):
		if self.cuda:
			features, output = self.extractor(input.cuda())
		else:
			features, output = self.extractor(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.features.zero_grad()
		self.model.classifier.zero_grad()
		one_hot.backward(retain_variables=True)

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

		target = features[-1]
		target = target.cpu().data.numpy()[0, :]

		weights = np.mean(grads_val, axis = (2, 3))[0, :]
		cam = np.ones(target.shape[1 : ], dtype = np.float32)

		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		# cam = cv2.resize(cam, (224, 224))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam

class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
	def __init__(self, model, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		# replace ReLU with GuidedBackpropReLU
		for idx, module in self.model.features._modules.items():
			if module.__class__.__name__ == 'ReLU':
				self.model.features._modules[idx] = GuidedBackpropReLU()

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):
		if self.cuda:
			output = self.forward(input.cuda())
		else:
			output = self.forward(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		# self.model.features.zero_grad()
		# self.model.classifier.zero_grad()
		one_hot.backward(retain_variables=True)

		output = input.grad.cpu().data.numpy()
		output = output[0,:,:,:]

		return output
