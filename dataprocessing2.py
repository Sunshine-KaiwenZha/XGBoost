import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, log_loss

def draw(fpr, tpr, roc_auc, _fpr, _tpr, _roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='blue',
             lw=lw, label='ROC Curve of Linear Model (AUC = %0.3f)' % roc_auc)
    plt.plot(_fpr, _tpr, color='gold',
             lw=lw, label='ROC Curve of XGBoost Model (AUC = %0.3f)' % _roc_auc)
    plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic(ROC)')
    plt.legend(loc="lower right")
    plt.savefig('v2_pic.png', format='png')

def readData1():
    xy = pd.read_csv('./logitbad.csv', delimiter=',', dtype=np.float32)
    xy = xy.values
    prob = xy[:, 0:1].tolist()
    label = xy[:, 1:2].tolist()
    # print(prob)
    # print(label)
    return prob, label

def readData2():
    infile = open("isoverdue_preprocess.txt", "r")
    prob = []
    label = []
    while True:
        line = infile.readline()
        if len(line) == 0:
            break

        pos1 = line.find("(")
        pos = line.find(",")
        pos2 = line.find(")")

        probability = eval(line[pos1 + 1:pos])
        lab = eval(line[pos + 1:pos2])

        prob.append(probability)
        label.append(lab)

    return prob, label

if __name__ == "__main__":
    prob, label = readData1()
    _prob, _label = readData2()

    roc_auc = 0.668
    fpr, tpr, thresholds = roc_curve(label, prob)

    _roc_auc = roc_auc_score(_label, _prob)
    _fpr, _tpr, _thresholds = roc_curve(_label, _prob)

    draw(fpr, tpr, roc_auc, _fpr, _tpr, _roc_auc)
