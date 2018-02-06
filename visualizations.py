import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

import imageio, os

import torch
from torch.autograd import Variable, Function

import numpy as np


def make_roc_gif(results_dir, epochs, fold_num=1):
    images = []

    epoch_range = range(1, epochs+1)
    for epoch in epoch_range:
        filename = results_dir + 'ROC_fold_' + str(fold_num) + '_epoch_' + str(epoch) + '.png'
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
    plt.title('ROC Epoch:' + str(epoch_num), fontsize=24)
    plt.legend(loc="lower right", shadow=True, fontsize=20)

    plt.savefig(results_dir + 'ROC_fold_' + str(fold_num) + '_epoch_' + str(epoch_num) +  '.png', bbox_inches='tight')
    plt.close()

    return train_roc_auc, val_roc_auc, test_roc_auc

def plot_sens_spec(train_sens, train_spec, val_sens, val_spec, test_sens, test_spec, results_dir, fold_num=-1):
    plt.figure(figsize=(8, 8))

    epoch_number = range(len(train_sens))

    lw = 2

    if not train_sens is None:
        plt.plot(epoch_number, train_sens, color='darkorange', lw=lw, label='Train Sensitivity')
    if not train_spec is None:
        plt.plot(epoch_number, train_spec, color='gold', lw=lw, label='Train Specificity')

    if not val_sens is None:
        plt.plot(epoch_number, val_sens, color='darkred', lw=lw, label='Validation Sensitivity')
    if not val_spec is None:
        plt.plot(epoch_number, val_spec, color='salmon', lw=lw, label='Validation Specificity')

    if not test_sens is None:
        plt.plot(epoch_number, test_sens, color='darkblue', lw=lw, label='Test Sensitivity')
    if not test_spec is None:
        plt.plot(epoch_number, test_spec, color='mediumblue', lw=lw, label='Test Specificity')

    plt.legend(shadow=True, fontsize=20)
    plt.savefig(results_dir + 'results_fold_' + str(fold_num), bbox_inches='tight')
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
		self.feature_extractor = FeatureExtractor(self.model.conv6_bn, target_layers)

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
