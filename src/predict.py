import argparse
import numpy as np
import pandas as pd
from keras.models import load_model

from keras.utils.np_utils import to_categorical

import keras.backend as K

def mcc(y_true, y_pred):
    ''' Matthews correlation coefficient
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(1 - y_neg * y_pred_pos)
    fn = K.sum(1 - y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

import numpy as np

import utils
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
    pred = model.predict_generator(
        utils.data_generator(fn_list, label, max_len, batch_size, shuffle=False),
        steps=len(fn_list)//batch_size + 1,
        verbose=verbose
        )
    return pred

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
    label = np.zeros((fn_list.shape))

    pred_proba = predict(model, fn_list, label, args.batch_size, args.verbose)
    pred = pred_proba.argmax(1)

    #pred = np.eye(len(classes))[y_pred]
    import scikitplot as skplt
    import matplotlib.pyplot as plt

    
    y_true = df['y_true'].values
    skplt.metrics.plot_roc(y_true,pred_proba)
    plt.savefig("/tmp/roc.pdf")

    #pred_proba = model.predict_proba(model, fn_list, label, args.batch_size, args.verbose)
    #pred_proba = pred_proba.argmax(1)

    #print(df[1].values)
    df['y_pred'] = pred
    #df['predcit probability'] = pred_proba
    df['fn_list'] = [i.split('/')[-1] for i in fn_list] # os.path.basename
    df = df.join(pd.DataFrame(pred_proba),rsuffix="_proba")
    df.to_csv(args.result_path, header=True, index=False)
    print('Results writen in', args.result_path)

