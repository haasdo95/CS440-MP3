from splitData import run
import utils
import os

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