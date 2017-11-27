import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# the following funciton is adapted from...
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# def plot_confusion_matrix(cm, classes, fname,
#                           normalize=False, print_txt=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     :param confusion matrix (cm) and class labels (classes) need to be array-like
#     :param confusion matrix (cm) should have rows representing true labels
#     """
#
#     cm = np.array(cm)
#     classes = np.array(classes)
#
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#     if print_txt:
#         if normalize:
#             print("Normalized confusion matrix")
#         else:
#             print('Confusion matrix, without normalization')
#         print(cm)
#
#     plt.figure()
#     plt.clf()
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.savefig(fname, bbox_inches='tight', pad_inches=0)


def plot_confusion_matrix(cm, classes, fname, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = np.array(cm)
    classes = np.array(classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.round(cm, decimals=2)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    fig, ax = plt.subplots()
    ax = sns.heatmap(df_cm, annot=True, cmap=cmap)
    ax.set_title(title)
    ax.figure.savefig(fname)


def plot_heat_map(i, j, i_mat, j_mat, i_j_mat, fname, cmap=plt.cm.jet):
    """
    :param heat_matrix must be a 2D array-like object
    """
    df_i_mat = pd.DataFrame(np.log(i_mat))
    df_j_mat = pd.DataFrame(np.log(j_mat))
    df_i_j_mat = pd.DataFrame(np.log(i_j_mat)/4)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))
    ax1 = sns.heatmap(df_i_mat, annot=False, xticklabels=False, yticklabels=False, cmap=cmap, ax=ax1)
    ax1.set_title("Likelihood Map of Digit " + str(i))
    ax2 = sns.heatmap(df_j_mat, annot=False, xticklabels=False, yticklabels=False, cmap=cmap, ax=ax2)
    ax2.set_title("Likelihood Map of Digit " + str(j))
    ax3 = sns.heatmap(df_i_j_mat, annot=False, xticklabels=False, yticklabels=False, cmap=cmap, ax=ax3)
    ax3.set_title("Log Odds Ratio of Pair (" + str(i) + "," + str(j) + ")")
    fig.savefig(fname)

# parent_dir = Path(__file__).resolve().parents[1]
# fname = str(parent_dir) + '/report/img/test.jpg'

