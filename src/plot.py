
import argparse

import pickle

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import matthews_corrcoef

import scikitplot as skplt

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description='Malconv-keras classifier')
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--csv', type=str, default='../saved/result.csv')
parser.add_argument('--plot-title', type=str, default='')

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    #plt.xticks(ticks=np.arange(cm.shape[1]),
    #           labels = classes)
    
    #ax.set_yticklabels(classes)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels= classes, yticklabels= classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label',

           )
    ax.set_ylim(len(classes)-0.5, -0.5)
    #plt.yticks(ticks=np.arange(cm.shape[0]), labels=classes)
    #plt.yticks(ticks=np.arange(cm.shape[1]+1) - 0.5,
    #           labels =[''] + classes)

    # plt.xlabel("Predicted label")
    # plt.ylabel("True label")

    # plt.title(title)


    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    return plt

def get_predictions_by_class(y_true,y_pred,by_y_true=True):
    if by_y_true:
        i = 0
    else:
        i = 1
    labels = np.unique(y_true)
    df = np.stack( (y_true,y_pred), axis=1 )
    for label in labels:
        x = df[df.T[i] == label]
        yield label, x.T[0] == label, x.T[1] == label

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

def calculate_precision_recall(y_true,y_pred):
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i, true, pred in get_predictions_by_class(y_true,y_pred,by_y_true=False):
        precision[i] = precision_score(true,pred)
    
    for i, true, pred in get_predictions_by_class(y_true,y_pred,by_y_true=True):
        recall[i] = recall_score(true,pred)


    # A "micro-average": quantifying score on all classes jointly
    #precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(),
    #    y_pred.ravel())
    #average_precision["micro"] = average_precision_score(y_true, y_pred,
    #
    #                                                 average="micro")
    return precision,recall, 0

def plot_metrics(history):
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='test')
    plt.legend()
    # plt accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history['acc'], label='train')
    plt.plot(history['val_acc'], label='test')
    plt.legend()
    plt.show()

def get_plot_directory(plot_title):
    if plot_title:
        try:
            os.mkdir("../plots/{}".format(plot_title))
        except (OSError):
            pass
        return "../plots/{}/".format(plot_title)
    else:
        return "../plots/"

def plot(y_true,y_pred,y_proba,plot_title):
    #classes = ['China', 'Russia', 'North-Korea', 'USA', 'Pakistan'] 
    #classes = ["APT-{}".format(i+1) for i in np.unique(y_true)]
    classes = ["Country {}".format(i) for i in range(5)]
    #classes = ['China', 'North-Korea']
    location =get_plot_directory(plot_title)

    # cnf_matrix = confusion_matrix(y_true, pred,labels=range(5))
    np.set_printoptions(precision=2)

    # BROKEN
    # norm = skplt.metrics.plot_confusion_matrix(y_true,y_pred,
    #                             #    classes=classes,
    #                                normalize=True,
    #                                title = plot_title + " (normalized)")
    # norm.set_ylim(len(classes)-0.5, -0.5)
    # plt.savefig(location + "normalized.pdf")
    norm = plot_confusion_matrix(y_true,y_pred, 
                                 normalize=True, 
                                 classes=classes, 
                                 title=plot_title + " (normalized)")
    
    norm.savefig(location + "normalized.pdf")
    plt.clf()

    # BROKEN
    # skplt.metrics.plot_confusion_matrix(y_true,y_pred,
    #                             #    classes=classes,
    #                                normalize=False,
    #                                title = plot_title + " (normalized)")
    # plt.savefig(location + "normalized.pdf")

    not_norm = plot_confusion_matrix(y_true,y_pred, 
                                     normalize=False, 
                                     classes=classes, 
                                     title=plot_title)

    not_norm.savefig(location + "not_normalized.pdf")

    #print(np.round(pred))
    acc = accuracy_score(y_true,np.round(y_pred))
    print(acc)

    print(classification_report(y_true, y_pred, digits=3))

    (TP, FP, TN, FN) = perf_measure(y_true, y_pred)
    precision, recall, average_precision = calculate_precision_recall(y_true, y_pred)
    print("recall ", recall)
    print("precision ", precision)

    kappa = cohen_kappa_score(y_true, y_pred)
    print("Cohen's Kappa statistic ", kappa)


    matthew = matthews_corrcoef(y_true,y_pred)
    print("Matthews correlation coeffiecient ", matthew)

    # Multi-class ROC
    #pred = np.eye(len(classes))[y_pred]
    skplt.metrics.plot_roc(y_true,y_proba,plot_micro=False,plot_macro=True,figsize=None)
    plt.savefig(location + "roc.pdf")

    skplt.metrics.plot_precision_recall(y_true,y_proba)
    plt.savefig(location + "precision_recall_curve.pdf")

    # Binary ROC
    # fpr, tpr, threshold = roc_curve(y_true,y_pred)
    # roc_auc = auc(fpr,tpr)
    # plt.clf()
    # plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr,tpr,'b', label = 'AUC = {0:.2f}'.format(roc_auc))
    # plt.legend(loc = 'lower right')
    # plt.plot([0,1], [0,1], 'r--')
    # plt.xlim([0,1])
    # plt.ylim([0,1])
    # plt.xlabel('True Positive Rate')
    # plt.ylabel('False Positive Rate')
    # plt.savefig(location + "roc.pdf")
    #plt.show()

    with open('../saved/history.pkl', 'rb') as f:
            history = pickle.load(f)
            print(history.keys())
            plot_metrics(history)

    with open(location + "scores.txt", 'w') as f:
        f.write("prediction dataset size: {}\n".format(len(y_true)))
        f.write("accuracy: {}\n".format(acc))
        f.write("recall: {}\n".format(recall))
        f.write("precision: {}\n".format(precision))
        f.write("kappa: {}\n".format(kappa))
        f.write("matthew corcoef: {}\n".format(matthew))
        # f.write("ROC\ntpr: {}\nfpr: {}\nAUC: {}".format(tpr,fpr,roc_auc))
        f.write(classification_report(y_true, y_pred, digits=3))

if __name__ == '__main__':
    args = parser.parse_args()
    
    # read data
    df = pd.read_csv(args.csv)

    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    y_proba = df.drop(['fn_list', 'y_true', 'y_pred'], axis=1).values

    plot(y_true,y_pred,y_proba,args.plot_title)

