import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score


def plot_roc(truth, probs, results_dir, epoch_num, fold_num=-1):
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

    plt.savefig(results_dir + '_epoch_' + str(epoch_num) + '_fold_' + fold_num + '_roc.png', bbox_inches='tight')
    plt.close()

def plot_sens_spec(train_sens, train_spec, val_sens, val_spec, test_sens, test_spec, results_dir, fold_num=-1):
    plt.figure(figsize=(8, 8))

    epoch_number = range(len(train_sens))

    lw = 2

    if not train_sens is None:
        plt.plot(epoch_number, train_sens, color='darkgoldenrod', lw=lw//2, label='Sensitivity (train)')
    if not val_sens is None:
        plt.plot(epoch_number, val_sens, color='darkorange', lw=lw//2, label='Sensitivity (val)')
    if not test_sens is None:
        plt.plot(epoch_number, test_sens, color='orange', lw=lw, label='Sensitivity (test)')

    if not train_spec is None:
        plt.plot(epoch_number, train_spec, color='darkblue', lw=lw//2, label='Specificity (train)')
    if not val_spec is None:
        plt.plot(epoch_number, val_spec, color='darkslateblue', lw=lw//2, label='Specificity (val)')
    if not test_spec is None:
        plt.plot(epoch_number, test_spec, color='blue', lw=lw, label='Specificity (test)')

    plt.legend(shadow=True)
    plt.savefig(results_dir + '_fold_' + str(fold_num), bbox_inches='tight')
    plt.close()

