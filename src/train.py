#!/bin/python3

from os.path import join
import argparse
import pickle
import warnings
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.models import load_model
from keras import optimizers
import math

from keras.utils.np_utils import to_categorical
import numpy as np

from sklearn.metrics import matthews_corrcoef

import keras.backend as K
from keras import losses

import utils
from utils import mcc
from malconv import Malconv
from preprocess import preprocess

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef

import predict

#TEMP
from collections import Counter

from sklearn.utils import class_weight
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Malconv-keras classifier training')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--limit', type=float, default=0., help="limit gpy memory percentage")
parser.add_argument('--max_len', type=int, default=3594, help="model input legnth")
parser.add_argument('--win_size', type=int, default=500)
parser.add_argument('--val_size', type=float, default=0.1, help="validation percentage")
parser.add_argument('--save_path', type=str, default='../saved/', help='Directory to save model and log')
parser.add_argument('--model_path', type=str, default='../saved/malconv.h5', help="model to resume")
parser.add_argument('--save_best', action='store_true', help="Save model with best validation accuracy")
parser.add_argument('--resume', action='store_true')
parser.add_argument('--under_sample', action='store_true', help="Under-sample dataset")
parser.add_argument('--kfold', type=int, default=0, help="Use k fold cross validation")
parser.add_argument('csv', type=str)



def train(model, data, label, max_len=200000, batch_size=64, verbose=True, epochs=100, save_path='../saved/', save_best=True):
    
    #Learning rate schedulers
    def step_decay(epoch):
        initial_lrate = 0.01
        drop = 0.5
        epochs_drop = 5.0
        lrate = initial_lrate * math.pow(drop,  
                math.floor((1+epoch)/epochs_drop))
        return lrate

    # callbacks
    ear = EarlyStopping(monitor='val_loss', patience=4)
    mcp = ModelCheckpoint(join(save_path, 'malconv.h5'), 
                          monitor="val_acc", 
                          save_best_only=save_best, 
                          save_weights_only=False)
    lrs = LearningRateScheduler(schedule=step_decay)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                 patience = 3, min_lr=0.00001)

    # y = to_categorical(label, num_classes=n_classes)
    y_argmax = np.argmax(label,axis=1)
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y_argmax),
                                                      y_argmax)

    x_data, len_list = preprocess(data,max_len=max_len)
    # x_train_data = preprocess(x_train, max_len=6000)
    # x_test_data = preprocess(x_test, max_len=6000)

    history = model.fit(
        x=x_data,
        y=label,
        steps_per_epoch=len(x_data)//batch_size + 1,
        epochs=epochs, 
        verbose=verbose, 
        callbacks=[ear, mcp,reduce_lr],
        validation_split=args.val_size,
        validation_steps=int(len(x_data)*args.val_size//batch_size + 1),
        class_weight=class_weights,
        
    )

    return history

def plot_history(history):
    # plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.savefig("../saved/loss.pdf")
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.savefig("../saved/acc.pdf")
    plt.show()

def generate_model(max_len, win_size, out_size):
    model = Malconv(max_len, win_size, out_size=out_size)
    # default:
    # adam = optimizers.adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

    #adam = optimizers.adam(lr=0.01, beta_1=0.9, beta_2=0.999, amsgrad=True)
    sgd = optimizers.SGD(learning_rate=0.01, momentum=0.9)
    # model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc',mcc])
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])
    #model.compile(loss=losses.CategoricalCrossentropy(from_logits=True), optimizer=adam, metrics=['acc'])
    #model.compile(loss='kullback_leibler_divergence', optimizer=adam, metrics=['acc',mcc])

    return model

if __name__ == '__main__':
    args = parser.parse_args()
    
    # limit gpu memory
    if args.limit > 0:
        utils.limit_gpu_memory(args.limit)
    
    # prepare data
    # preprocess is handled in utils.data_generator
    df = pd.read_csv(args.csv, header=None)
    data, label = df[0].values, df[1].values

    n_classes = len(np.unique(label))
    max_len = len(data)

    # Prepare model
    if args.resume:
        model = load_model(args.model_path)
    else:
        model = generate_model(max_len, args.win_size, out_size=n_classes)

    
    # Under-sample
    if args.under_sample:
        print("Under-sampling dataset")
        rus = RandomUnderSampler(random_state=0)
        data, label = rus.fit_resample(data.reshape(-1, 1), label)
        data = data.ravel()

        df_predict = pd.DataFrame(data=[x for x in df.values if not x[0] in data])
        predict_data, predict_label = zip(*[ (x[0],x[1]) for x in df.values if not x[0] in data])
        predict_data = np.array(predict_data)
        predict_label = np.array(predict_label)

    
    if args.kfold:
        seed = 1
        kfold = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=seed)
        cvscores = []
        for train_data, test in kfold.split(data, label):
            #x_train, x_test, y_train, y_test = data[train], data[test], label[train], label[test]
            x_train, x_test, y_train, y_test = utils.train_test_split(data[train_data], label[train_data], args.val_size)
            #y_train = y_train.astype(float)
            print('Train on %d data, test on %d data' % (len(x_train), len(x_test)))
            model = generate_model(max_len, args.win_size, out_size=n_classes)
            history = train(model, x_train, y_train, max_len, args.batch_size, args.verbose, args.epochs, args.save_path, args.save_best)
            
            # predict/evaluate
            pred, pred_proba = predict.predict(model,data[test],label[test])
            # pred2, pred_proba2 = predict.predict(predict_data, predict_label)

            # pred = np.concatenate(pred1,pred2)
            # pred_proba = np.concatenate(pred1,pred2)
            filepath = "../saved/result{}.csv".format(len(cvscores))
            predict.write_to_csv(filepath,data[test],label[test],pred, pred_proba)

            with open(join(args.save_path, 'history.pkl'), 'wb') as f:
                pickle.dump(history.history, f)

            mcc = matthews_corrcoef(label[test],pred)
            print("%s: %.2f" % ("mcc", mcc))
            if len(cvscores) == 0 or mcc  > np.max(cvscores):
                best_history = history
            cvscores.append(mcc)
        print(cvscores)
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        history = best_history
    else: 
        x_train, x_test, y_train, y_test = utils.train_test_split(data, label, args.val_size)
        #y_train = y_train.astype(float)
        print('Train on %d data, test on %d data' % (len(x_train), len(x_test)))
        history = train(model, x_train, y_train, max_len, args.batch_size, args.verbose, args.epochs, args.save_path, args.save_best)
    
    if args.under_sample:
        pred, pred_proba = predict.predict(model, predict_data, predict_label, args.batch_size, args.verbose)

        result_path = "../saved/result.csv"
        #print(df[1].values)
        df_predict['predict score'] = pred
        df_predict[0] = [i.split('/')[-1] for i in predict_data] # os.path.basename
        df_predict.to_csv(result_path, header=None, index=False)
        print('Results writen in', result_path)
    
    # Should be in plot.py
    plot_history(history)

    with open(join(args.save_path, 'history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)

