import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score


def plot_roc(truth, probs, results_dir, epoch_num):
    plt.figure(figsize=(8, 8))

    lw = 2
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    roc_auc = roc_auc_score(truth, probs[:, 1], 'weighted')
    fpr, tpr, _ = roc_curve(truth, probs[:, 1])

    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC (area = %0.2f)' % roc_auc)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    # plt.title('Receiver operating characteristic example', fontsize=24)
    plt.legend(loc="lower right", shadow=True, fontsize=20)

    plt.savefig(results_dir + '_epoch_' + str(epoch_num) + '_roc.png', bbox_inches='tight')
    plt.close()

