import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

import imageio, os

import torch
from torch.autograd import Variable, Function

import numpy as np
from scipy.stats import entropy
from sklearn.neighbors.kde import KernelDensity


def make_roc_gif(results_dir, epochs, fold_num=1):
    images = []

    epoch_range = range(1, epochs+1)
    for epoch in epoch_range:
        filename = results_dir + 'ROC_fold_' + str(fold_num) + '_epoch_' + str(epoch).zfill(2) + '.png'
        img = plt.imread(filename)
        images.append(img)

    imageio.mimsave(results_dir + 'ROC_fold_' + str(fold_num) + '.gif', images)


def plot_roc(train_truth, train_probs, val_truth, val_probs, test_truth, test_probs, results_dir, epoch_num, fold_num=-1):
    plt.figure(figsize=(8, 8))

    lw = 2
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    train_roc_auc, val_roc_auc, test_roc_auc = 0, 0, 0

    try:
        train_roc_auc = roc_auc_score(train_truth, train_probs[:, 1], 'weighted')
        train_fpr, train_tpr, _ = roc_curve(train_truth, train_probs[:, 1])
        plt.plot(train_fpr, train_tpr, color='darkorange', lw=lw, label='Train ROC (area = %0.2f)' % train_roc_auc)
    except:
        print('Couldnt plot training')

    try:
        val_roc_auc = roc_auc_score(val_truth, val_probs[:, 1], 'weighted')
        val_fpr, val_tpr, _ = roc_curve(val_truth, val_probs[:, 1])
        plt.plot(val_fpr, val_tpr, color='red', lw=lw, label='Val ROC (area = %0.2f)' % val_roc_auc)
    except Exception as e:
        print(e)
        print('Couldnt plot validation')

    try:
        test_roc_auc = roc_auc_score(test_truth, test_probs[:, 1], 'weighted')
        test_fpr, test_tpr, _ = roc_curve(test_truth, test_probs[:, 1])
        plt.plot(test_fpr, test_tpr, color='darkred', lw=lw, label='Test ROC (area = %0.2f)' % test_roc_auc)
    except:
        print('Couldnt plot test')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title('ROC Epoch:' + str(epoch_num).zfill(3), fontsize=24)
    plt.legend(loc="lower right", shadow=True, fontsize=20)

    plt.savefig(results_dir + 'ROC_fold_' + str(fold_num) + '_epoch_' + str(epoch_num).zfill(2) + '.png', bbox_inches='tight')
    plt.close()

    return train_roc_auc, val_roc_auc, test_roc_auc


def plot_sens_spec(train_sens, train_spec, val_sens, val_spec, test_sens, test_spec, results_dir, fold_num=-1):
    plt.figure(figsize=(8, 8))

    epoch_number = range(len(train_sens))

    lw = 2

    if not train_sens is None:
        plt.plot(epoch_number, train_sens, color='darkorange', linestyle=':', lw=lw, label='Train Sensitivity')
    if not train_spec is None:
        plt.plot(epoch_number, train_spec, color='gold', linestyle=':', lw=lw, label='Train Specificity')

    if not val_sens is None:
        plt.plot(epoch_number, val_sens, color='darkred', linestyle='--', lw=lw, label='Validation Sensitivity')
    if not val_spec is None:
        plt.plot(epoch_number, val_spec, color='salmon', linestyle='--', lw=lw, label='Validation Specificity')

    if not test_sens is None:
        plt.plot(epoch_number, test_sens, color='darkblue', lw=lw, label='Test Sensitivity')
    if not test_spec is None:
        plt.plot(epoch_number, test_spec, color='mediumblue', lw=lw, label='Test Specificity')

    plt.xlabel('Epoch #', fontsize=20)
    plt.ylabel('Metric Value', fontsize=20)

    plt.legend(shadow=True, fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(results_dir + 'results_fold_' + str(fold_num), bbox_inches='tight')
    plt.close()


def sens_spec_across_folds(sens_to_plot, spec_to_plot, results_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))

    bplot1 = ax1.boxplot(sens_to_plot, patch_artist=True)
    bplot2 = ax2.boxplot(spec_to_plot, patch_artist=True)

    colors = ['pink', 'lightblue', 'lightgreen']
    for bplot in (bplot1, bplot2):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    ax1.set_ylim([0, 1])
    ax2.set_ylim([0, 1])

    ax1.set_xticklabels(['Train', 'Validation', 'Test'], fontsize=20)
    ax2.set_xticklabels(['Train', 'Validation', 'Test'], fontsize=20)

    ax1.set_title('Sensitivity', fontsize=24)
    ax2.set_title('Specificity', fontsize=24)

    ax1.grid()
    ax2.grid()
    plt.savefig(results_dir + 'sensitivity_specificity_all_folds.png', dpi=500)
    plt.close()


def plot_entropy(probabilities, truth, results_dir):
    probabilities = np.exp(probabilities) # probs are actually log probs

    pass_entropies, fail_entropies = [], []
    tp_entropies, fp_entropies, tn_entropies, fn_entropies = [], [], [], []

    for i, y_true in enumerate(truth):
        y_prob = np.mean(probabilities[i, :, 1])
        if y_prob < 0.5:
            y_predicted = 0
        else:
            y_predicted = 1

        # print('pass probs:', probabilities[i, :, 1])

        pass_probs = probabilities[i, :, 1]
        fail_probs = probabilities[i, :, 0]

        pass_probs = pass_probs[pass_probs != 0]
        fail_probs = fail_probs[fail_probs != 0]

        H_pass = entropy(pass_probs) / pass_probs.shape[0]
        H_fail = entropy(fail_probs) / fail_probs.shape[0]

        # print('entropy:', H_pass, H_fail)

        if y_true == 1:
            pass_entropies.append(H_pass)
            if y_predicted == 1:
                tp_entropies.append(H_pass)
            else:
                fn_entropies.append(H_pass)
        else:
            fail_entropies.append(H_fail)
            if y_predicted == 1:
                fp_entropies.append(H_fail)
            else:
                tn_entropies.append(H_fail)


    prob_space = np.reshape(np.linspace(0, 1, 200), (-1, 1))

    bw = 0.00001

    kde_pass = KernelDensity(kernel='gaussian', bandwidth=bw).fit(np.reshape(np.asarray(pass_entropies), (-1, 1)))
    kde_fail = KernelDensity(kernel='gaussian', bandwidth=bw).fit(np.reshape(np.asarray(fail_entropies), (-1, 1)))

    kde_tp = KernelDensity(kernel='gaussian', bandwidth=bw).fit(np.reshape(np.asarray(tp_entropies), (-1, 1)))
    kde_tn = KernelDensity(kernel='gaussian', bandwidth=bw).fit(np.reshape(np.asarray(tn_entropies), (-1, 1)))
    kde_fp = KernelDensity(kernel='gaussian', bandwidth=bw).fit(np.reshape(np.asarray(fp_entropies), (-1, 1)))
    kde_fn = KernelDensity(kernel='gaussian', bandwidth=bw).fit(np.reshape(np.asarray(fn_entropies), (-1, 1)))

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))

    ax1.plot(prob_space, kde_pass.score_samples(prob_space), color='g', label='Pass')
    ax1.plot(prob_space, kde_fail.score_samples(prob_space), color='r', label='Fail')

    ax2.plot(prob_space, kde_tp.score_samples(prob_space), color='b', label='TP')
    ax2.plot(prob_space, kde_fn.score_samples(prob_space), color='r', label='FN')
    ax2.plot(prob_space, kde_fp.score_samples(prob_space), color='orange', label='FP')
    ax2.plot(prob_space, kde_tn.score_samples(prob_space), color='g', label='TN')

    ax1.legend(loc='upper right', shadow=True)
    ax2.legend(loc='upper right', shadow=True)

    plt.savefig(results_dir + 'prediction_entropies.png', dpi=500)
    plt.close()


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
