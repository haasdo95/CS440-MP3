import sys, itertools, warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# the following funciton is adapted from...
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes, fname, 
                          normalize=False, print_txt=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    :param confusion matrix (cm) and class labels (classes) need to be array-like
    :param confusion matrix (cm) should have rows representing true labels
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if print_txt:
	    if normalize:
	        print("Normalized confusion matrix")
	    else:
	        print('Confusion matrix, without normalization')
	    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)


def plot_heat_map(heat_matrix, fname, title, cmap="hot"):
    """
    :param heat_matrix must be a 2D array-like object
    """
    ax = plt.axes()
    ax = sns.heatmap(np.array(heat_matrix), xticklabels=False, yticklabels=False)
    ax.set_title(title)
    ax.figure.savefig(fname)

# parent_dir = Path(__file__).resolve().parents[1]
# fname = str(parent_dir) + '/report/img/test.jpg'

