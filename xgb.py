import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, log_loss

outfile = open("v2.txt", "w")

def readData():
    xy = pd.read_csv('./data.csv', delimiter=',', dtype=np.float32)
    xy = xy.values
    X = xy[:, 3:29]
    Y = xy[:, 0:1]
    return X, Y

def init(X, Y):
    assert X.shape[0] == Y.shape[0], 'shape not match'
    num_all = X.shape[0]
    num_train = int(0.9 * num_all)
    num_test = num_all - num_train
    # shuffle
    mask = np.random.permutation(num_all)
    X = X[mask]
    Y = Y[mask]
    # training data
    mask_train = range(num_train)
    X_train = X[mask_train]
    Y_train = Y[mask_train]
    # testing data
    mask_test = range(num_train, num_all)
    X_test = X[mask_test]
    Y_test = Y[mask_test]
    print('[Data Scale]', file=outfile)
    print('All data shape: ', X.shape, file=outfile)
    print('Train data shape: ', X_train.shape, file=outfile)
    print('Train label shape: ', Y_train.shape, file=outfile)
    print('Test data shape: ', X_test.shape, file=outfile)
    print('Test label shape: ', Y_test.shape, file=outfile)
    return X_train, Y_train, X_test, Y_test

def draw(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('v2_pic.png', format='png')

def work():
    X, Y = readData()
    X_train, Y_train, X_test, Y_test = init(X, Y)
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)

    # param = {
    #     'objective': 'binary:logistic',
    #     'min_child_weight': 100,
    #     'eta': 0.02,
    #     'colsample_bytree': 0.7,
    #     'max_depth': 12,
    #     'subsample': 0.7,
    #     'alpha': 1,
    #     'gamma': 1,
    #     'silent': 1,
    #     'verbose_eval': True,
    #     'seed': 12
    # }

    param = {'objective': 'binary:logistic', 'max_depth': 10}

    num_round = 100
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_round, evallist)
    preds = bst.predict(dtest)

    print('\n[Test Module] Format:(probability, label)', file=outfile)
    for i in range(preds.shape[0]):
        print('Test case {}: {}'.format(i+1, (preds[i], int(Y_test[i][0]))), file=outfile)
        # print(preds[i], int(Y_test[i][0]), file=_outfile)

    print('\n[Analysis Module]', file=outfile)

    # feature_importance
    xgb.plot_importance(bst)
    plt.savefig('feat_pic.png', format='png')

    roc_auc = roc_auc_score(Y_test, preds)
    print("AUC:", roc_auc)
    print("AUC:", roc_auc, file=outfile)

    logloss = log_loss(Y_test, preds)
    print("LogLoss:", logloss)
    print("LogLoss:", logloss, file=outfile)

    fpr, tpr, thresholds = roc_curve(Y_test, preds)
    draw(fpr, tpr, roc_auc)

    preds = preds > 0.5
    print("accuracy_score:", accuracy_score(Y_test, preds))
    print("accuracy_score:", accuracy_score(Y_test, preds), file=outfile)
    print("f1_score:", f1_score(Y_test, preds, average='weighted'))
    print("f1_score:", f1_score(Y_test, preds, average='weighted'), file=outfile)

if __name__ == "__main__":
    work()