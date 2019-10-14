
import argparse

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report

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
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im)
    # We want to show all ticks...
    plt.xticks(ticks=np.arange(cm.shape[1]),
               labels = classes)
    
    #ax.set_yticklabels([''] + classes)
    plt.yticks(ticks=np.arange(cm.shape[1]+1) - 0.5,
               labels =[''] + classes)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.title(title)


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
    plt.tight_layout()

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

def get_plot_directory(plot_title):
    if plot_title:
        try:
            os.mkdir("../plots/{}".format(plot_title))
        except (OSError):
            pass
        return "../plots/{}/".format(plot_title)
    else:
        return "../plots/"

def plot(y_true,y_pred,plot_title):
    #classes = ['China', 'Russia', 'North-Korea', 'USA', 'Pakistan'] 
    classes = ["APT-{}".format(i+1) for i in np.unique(y_true)]
    #classes = ["Country {}".format(i) for i in range(5)]
    location =get_plot_directory(plot_title)

    # cnf_matrix = confusion_matrix(y_true, pred,labels=range(5))
    np.set_printoptions(precision=2)

    norm = plot_confusion_matrix(y_true,y_pred, 
                                 normalize=True, 
                                 classes=classes, 
                                 title=plot_title + " (normalized)")
    norm.savefig(location + "normalized.pdf")
    plt.clf()

    not_norm = plot_confusion_matrix(y_true,y_pred, 
                                     normalize=False, 
                                     classes=classes, 
                                     title=plot_title)

    
    
    #norm.show()

    not_norm.savefig(location + "not_normalized.pdf")
    #not_norm.show()
    #print(np.round(pred))
    acc = accuracy_score(y_true,np.round(y_pred))
    print(acc)

    print(classification_report(y_true, y_pred, digits=3))

    (TP, FP, TN, FN) = perf_measure(y_true, y_pred)
    precision, recall, average_precision = calculate_precision_recall(y_true, y_pred)
    print("recall ", recall)
    print("precision ", precision)

    with open(location + "scores.txt", 'w') as f:
        f.write("prediction dataset size: {}\n".format(len(y_true)))
        f.write("accuracy: {}\n".format(acc))
        f.write("recall: {}\n".format(recall))
        f.write("precision: {}\n".format(precision))
        f.write(classification_report(y_true, y_pred, digits=3))

if __name__ == '__main__':
    args = parser.parse_args()
    
    # read data
    df = pd.read_csv(args.csv, header=None)

    y_true = df[1].values
    y_pred = df[2].values

    plot(y_true,y_pred,args.plot_title)

