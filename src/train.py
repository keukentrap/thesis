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

import numpy as np

import utils
from malconv import Malconv
from preprocess import preprocess

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold

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
parser.add_argument('--max_len', type=int, default=200000, help="model input legnth")
parser.add_argument('--win_size', type=int, default=500)
parser.add_argument('--val_size', type=float, default=0.1, help="validation percentage")
parser.add_argument('--save_path', type=str, default='../saved/', help='Directory to save model and log')
parser.add_argument('--model_path', type=str, default='../saved/malconv.h5', help="model to resume")
parser.add_argument('--save_best', action='store_true', help="Save model with best validation accuracy")
parser.add_argument('--resume', action='store_true')
parser.add_argument('--under_sample', action='store_true', help="Under-sample dataset")
parser.add_argument('--k_fold', type=int, default=0, help="Use k fold cross validation")
parser.add_argument('csv', type=str)



def train(model, max_len=200000, batch_size=64, verbose=True, epochs=100, save_path='../saved/', save_best=True):
    
    #Learning rate schedulers
    def step_decay(epoch):
        initial_lrate = 0.001
        drop = 0.5
        epochs_drop = 5.0
        lrate = initial_lrate * math.pow(drop,  
                math.floor((1+epoch)/epochs_drop))
        return lrate

    # callbacks
    ear = EarlyStopping(monitor='val_acc', patience=7)
    mcp = ModelCheckpoint(join(save_path, 'malconv.h5'), 
                          monitor="val_acc", 
                          save_best_only=save_best, 
                          save_weights_only=False)
    #lrs = LearningRateScheduler(schedule=step_decay)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                 patience = 4, min_lr=0.00001)

    y_argmax = np.argmax(y_train,axis=1)
    print(y_argmax)
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y_argmax),
                                                      y_argmax)

    history = model.fit_generator(
        utils.data_generator(x_train, y_train, max_len, batch_size, shuffle=True),
        steps_per_epoch=len(x_train)//batch_size + 1,
        epochs=epochs, 
        verbose=verbose, 
        callbacks=[ear, mcp, reduce_lr],
        validation_data=utils.data_generator(x_test, y_test, max_len, batch_size),
        validation_steps=len(x_test)//batch_size + 1,
        class_weight=class_weights
    )

    return history

def plot_history(history):
    # plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()
    
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
    
    # prepare model
    if args.resume:
        model = load_model(args.model_path)
    else:
        model = Malconv(args.max_len, args.win_size, out_size=n_classes)
        # default:
        # adam = optimizers.adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

        adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        # model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
        # model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
        model.compile(loss='kullback_leibler_divergence', optimizer=adam, metrics=['acc'])
        
    
    

    
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

    
    if args.k_fold:
        seed = 0
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        cvscores = []
        for train_data, test in kfold.split(data, label):
            #x_train, x_test, y_train, y_test = data[train], data[test], label[train], label[test]
            x_train, x_test, y_train, y_test = utils.train_test_split(data[train_data], label[train_data], args.val_size)
            print('Train on %d data, test on %d data' % (len(x_train), len(x_test)))
            history = train(model, args.max_len, args.batch_size, args.verbose, args.epochs, args.save_path, args.save_best)
            
            # predict/evaluate
            scores = model.evaluate(data[test], label[test], verbose=0)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            if scores[1] * 100 > np.max(csvscores):
                best_history = history.copy()
            cvscores.append(scores[1] * 100)
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        history = best_history
    else: 
        x_train, x_test, y_train, y_test = utils.train_test_split(data, label, args.val_size)
        print('Train on %d data, test on %d data' % (len(x_train), len(x_test)))
        history = train(model, args.max_len, args.batch_size, args.verbose, args.epochs, args.save_path, args.save_best)
    
    if args.under_sample:
        pred = predict.predict(model, predict_data, predict_label, args.batch_size, args.verbose)
        pred = pred.argmax(1)

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

