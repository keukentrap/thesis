import argparse
import numpy as np
import pandas as pd
from keras.models import load_model

from keras.utils.np_utils import to_categorical

import keras.backend as K

import numpy as np

import utils
from utils import mcc
from preprocess import preprocess

parser = argparse.ArgumentParser(description='Malconv-keras classifier')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--limit', type=float, default=0.)
parser.add_argument('--model_path', type=str, default='../saved/malconv.h5')
parser.add_argument('--result_path', type=str, default='../saved/result.csv')
parser.add_argument('csv', type=str)

def predict(model, fn_list, label, batch_size=64, verbose=1):
    
    max_len = model.input.shape[1]
    pred_proba = model.predict_generator(
        utils.data_generator(fn_list, label, max_len, batch_size, shuffle=False),
        steps=len(fn_list)//batch_size + 1,
        verbose=verbose
        )
    pred = pred_proba.argmax(1)
    return pred, pred_proba

def write_to_csv(filepath, fn_list, y_true, y_pred, y_pred_proba):
    df = pd.DataFrame(columns=['fn_list','y_true','y_pred'])
    df['fn_list'] = fn_list
    df['y_true'] = y_true
    df['y_pred'] = y_pred
    df = df.join(pd.DataFrame(y_pred_proba),rsuffix="_proba")
    df.to_csv(filepath, header=True, index=False)
    print('Results writen in', filepath)

if __name__ == '__main__':
    args = parser.parse_args()
    
    # limit gpu memory
    if args.limit > 0:
        utils.limit_gpu_memory(args.limit)
    
    # load model
    model = load_model(args.model_path, custom_objects={'mcc':mcc})
    
    # read data
    df = pd.read_csv(args.csv, header=None)
    df.columns = ['fn_list', 'y_true']
    fn_list = df['fn_list'].values
    y_true  = df['y_true'].values
    #label = np.zeros((fn_list.shape))

    pred, pred_proba = predict(model, fn_list, y_true, args.batch_size, args.verbose)

    write_to_csv(args.result_path,fn_list,y_true,pred,pred_proba)

