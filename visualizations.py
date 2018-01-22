import torch

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



def plot_roc(probs, truth, results_dir, fold_num):
    plt.figure(figsize=(8,8))

    epoch_num = range(truth)

    plt.close()
    # plt.plot(epoch_num, hist.history['acc'], label='Training Accuracy')
    # plt.plot(epoch_num, hist.history['val_acc'], label="Validation Accuracy")
    plt.plot(epoch_num, self.train_sens, label='Train Sensitivity')
    plt.plot(epoch_num, self.train_spec, label='Train Specificity')
    plt.plot(epoch_num, self.val_sens, label='Validation Sensitivity')
    plt.plot(epoch_num, self.val_spec, label='Val Specificity')

    plt.legend(shadow=True)
    plt.xlabel("Training Epoch Number")
    plt.ylabel("Metric Value")
    plt.savefig(results_dir + 'training_metrics.png', bbox_inches='tight')
    plt.close()