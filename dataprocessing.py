from string import *
import numpy as np
import pandas as pd
from scipy import stats, integrate
import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, log_loss

infile = open("full.txt", "r")
outfile = open("out.txt", "w")

count = 0
zero = []
_zero = []
one = []
_one = []

y_test = []
y_pred = []

def draw(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic(ROC)')
    plt.legend(loc="lower right")
    plt.savefig('kkkkk.png', format='png')

if __name__ == "__main__":
    while True:
        line = infile.readline()
        if len(line) == 0:
            break
        count += 1

        pos1 = line.find("(")
        pos = line.find(",")
        pos2 = line.find(")")

        probability = eval(line[pos1+1:pos])
        label = eval(line[pos+1:pos2])
        y_test.append(label)
        y_pred.append(probability)

        if label == 0:
            zero.append([count, probability])
            _zero.append(probability)
        else:
            one.append([count, probability])
            _one.append(probability)

    # x = np.array(_one)
    # y = np.array(_zero)
    # sns.kdeplot(x, shade=True, legend=True, color="r")
    # sns.kdeplot(y, shade=True, legend=True, color="b")
    # plt.show()

    print(zero, file=outfile)
    print("\n")
    print(one, file=outfile)

    print('\n[Analysis Module]', file=outfile)

    roc_auc = roc_auc_score(y_test, y_pred)
    print("AUC:", roc_auc)
    print("AUC:", roc_auc, file=outfile)

    logloss = log_loss(y_test, y_pred)
    print("LogLoss:", logloss)
    print("LogLoss:", logloss, file=outfile)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    draw(fpr, tpr, roc_auc)

    # for i in y_pred:
    #     i = (i > 0.5)
    # print("accuracy_score:", accuracy_score(y_test, y_pred))
    # print("accuracy_score:", accuracy_score(y_test, y_pred), file=outfile)
    # print("f1_score:", f1_score(y_test, y_pred, average='weighted'))
    # print("f1_score:", f1_score(y_test, y_pred, average='weighted'), file=outfile)

