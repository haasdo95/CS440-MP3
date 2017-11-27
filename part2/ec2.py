from splitData import run
import utils
import os, warnings
from visualization.viz import *

cond_prob = run()
yes_path = './txt_yesno/yes_test'
no_path = './txt_yesno/no_test'
yesFiles = os.scandir(yes_path)
noFiles = os.scandir(no_path)

test = []
for i in yesFiles:
    test.append((utils.getData(i.path,interval=0,smooth=False)[0],'yes'))
for i in noFiles:
    test.append((utils.getData(i.path,interval=0,smooth=False)[0],'no'))

mat = utils.getAccuracy(cond_prob,test,25,10)
print(mat)
accu = 0
for i in mat:
    accu+=mat[i][i]
print('TOTAL ACCURACY:',accu)
conf_mat = list([[val for _, val in val_dict.items()] for _, val_dict in mat.items()])
parent_dir = os.path.dirname(os.path.dirname(__file__))
# know open issue when using savefig()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    title = 'Confusion Matrix for Hebrew Words (Y/N) with Unsegmented Data'
    file_name = str(parent_dir) + '/report/img/Hebrew_Words_Unsegmented_Conf_Mat.png'
    plot_confusion_matrix(cm=conf_mat, classes=list([key.upper() for key in mat.keys()]),
                          fname=file_name, normalize=True, title=title)
    print("CONF MAT GENERATED: ", title)