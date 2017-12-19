from keras.models import load_model
from keras import backend as K
from keras.layers import Dense
from keras.models import Model
from models.alexnet import decaf
from keras import metrics
import numpy as np
import os
from os.path import join
import argparse
from progressbar import ProgressBar
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import json
from datetime import datetime
from keras.preprocessing.image import load_img,img_to_array
from utils.utils import imagenet_preprocess_input,get_dico,wp_preprocess_input,invert_dico
from keras import activations
from vis.utils import utils

DEFAULT_MODEL_PATH='models/default/model.h5'
DEFAULT_BAGGING=True
DEFAULT_PREPROCESSING='imagenet'

def main():
    PATH = os.path.dirname(__file__)
    RESULT_FILE_PATH  = join(PATH,'../models/results.csv')
    K.set_image_data_format('channels_last')

    parser = argparse.ArgumentParser(description='Description')

    parser.add_argument('-t', action="store", default='acc', dest='type', help='Type of evaluation [pred|acc]')
    parser.add_argument('--isdecaf', action="store_true", dest='isdecaf',
                        help='if the model is a decaf6 type')
    parser.add_argument('-k', action="store", default='1,3,5', type=str, dest='k', help='top-k number')
    parser.add_argument('--data_path', action="store",
                        default=join(PATH, '../data/wikipaintings_small/wikipaintings_test'), dest='data_path',
                        help='Path of the data (image or train folder)')
    parser.add_argument('--model_path', action="store", dest='model_path', default=DEFAULT_MODEL_PATH,
                        help='Path of the h5 model file')
    parser.add_argument('-j', action="store_true", dest='json', help='Output prediction as json')
    parser.add_argument('-s', action="store_true", dest='save', help='Save accuracy in results file')
    parser.add_argument('-b', action="store_true", dest='b', default=DEFAULT_BAGGING, help='Sets bagging')
    parser.add_argument('-p', action="store", dest='preprocessing', default=DEFAULT_PREPROCESSING,
                        help='Type of preprocessing : [imagenet|wp]')


    args = parser.parse_args()

    eval_type = args.type

    model_path = args.model_path
    data_path = args.data_path
    isdecaf = args.isdecaf
    k = (str(args.k)).split(",")
    k = [int(val) for val in k]
    print(k)

    if eval_type == 'acc':
        preds = get_top_multi_acc(model_path, data_path,top_k=k,bagging=args.b,is_decaf6=isdecaf,preprocessing=args.preprocessing)
        for val,pred in zip(k,preds):
            print('\nTop-{} accuracy : {}%'.format(val,pred*100))

        if args.save and k==[1,3,5]:
            model_name = model_path.split('/')[-2]
            print(model_name)
            with open(RESULT_FILE_PATH,'a') as f:
                   f.write('\n'+model_name+";"+str(preds[0])+";"+str(preds[1])+";"+str(preds[2])+';'+str(datetime.now())+';'+str(args.b)+';'+args.preprocessing)

    elif eval_type == 'pred':
        k = k[0]
        model = init(model_path, isdecaf)
        pred,pcts = get_pred(model, data_path, is_decaf6=isdecaf, top_k=k,bagging=args.b,preprocessing=args.preprocessing)
        print(pcts)
        if args.json:
            result = { 'pred' : pred, 'k' : k }
            print(json.dumps(result))
        else:
            print("Top-{} prediction : {}".format(k, pred))
    else:
        print('Error in arguments. Please try with -h')






def get_y_pred(model_path, test_data_path, is_decaf6=False,top_k=1,bagging = False,preprocessing=None):

    model = init(model_path, is_decaf6=is_decaf6)

    target_size =(224,224)
    if is_decaf6:
        target_size = (227,227)

    dico = get_dico()
    y_true = []
    y_pred = []
    s = 0
    for t in list(os.walk(test_data_path)):
        s += len(t[2])
    style_names = os.listdir(test_data_path)
    print('Calculating predictions...')
    bar = ProgressBar(max_value=s)
    i = 0
    for style_name in style_names:
        style_path = join(test_data_path, style_name)
        img_names = os.listdir(style_path)
        label = dico.get(style_name)
        for img_name in img_names:
            img = load_img(join(style_path, img_name),target_size=target_size)
            x = img_to_array(img)
            if bagging:
                pred = _bagging_predict(x,model,preprocessing)
            else :
                x = _preprocess_input(x,preprocessing)
                pred = model.predict(x[np.newaxis,...])
            args_sorted = np.argsort(pred)[0][::-1]
            y_true.append(label)
            y_pred.append([a for a in args_sorted[:top_k]])
            i += 1
            bar.update(i)
    return np.asarray(y_pred), y_true



def _bagging_predict(x,model,preprocessing=None):
    x_flip = np.copy(x)
    x_flip = np.fliplr(x_flip)
    x = _preprocess_input(x,preprocessing)
    x_flip = _preprocess_input(x_flip,preprocessing)
    pred = model.predict(x[np.newaxis,...])
    pred_flip = model.predict(x_flip[np.newaxis,...])
    avg = np.mean(np.array([pred,pred_flip]), axis=0 )
    return avg

def _preprocess_input(x,preprocessing=None):
    if preprocessing == 'imagenet':
        return imagenet_preprocess_input(x)
    elif preprocessing == 'wp':
        return wp_preprocess_input(x)
    else:
        return x


def init(model_path, is_decaf6=False):
    if is_decaf6:
        K.set_image_data_format('channels_first')
        base_model = decaf()
        predictions = Dense(25, activation='softmax')(base_model.output)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights(model_path, by_name=True)
    else:
        model = load_model(model_path)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy', metrics.top_k_categorical_accuracy])    
    return model

def get_pred(model, image_path, is_decaf6=False, top_k=1,bagging=DEFAULT_BAGGING,preprocessing=DEFAULT_PREPROCESSING):
    target_size = (224, 224)
    if is_decaf6:
        target_size = (227, 227)
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    if bagging:
        pred = _bagging_predict(x, model,preprocessing=preprocessing)
    else:
        x = _preprocess_input(x,preprocessing=preprocessing)
        pred = model.predict(x[np.newaxis, ...])
    dico = get_dico()
    inv_dico = invert_dico(dico)
    args_sorted = np.argsort(pred)[0][::-1]
    preds = [inv_dico.get(a) for a in args_sorted[:top_k]]
    pcts = [pred[0][a] for a in args_sorted[:top_k]]
    return preds,pcts


def get_top_multi_acc(model_path, test_data_path, is_decaf6=False,top_k=[1,3,5],bagging=False,preprocessing=None):
    y_pred, y = get_y_pred(model_path, test_data_path, is_decaf6, max(top_k),bagging=bagging,preprocessing=preprocessing)
    scores = []
    for k in top_k:
        score = 0
        for pred, val in zip(y_pred[:,:k], y):
            if val in pred:
                score += 1
        scores.append(score / len(y))
    return scores


def plot_confusion_matrix(labels,preds):
    conf_arr = confusion_matrix(labels, preds)

    dico = get_dico()

    new_conf_arr = []
    for row in conf_arr:
        new_conf_arr.append(row / sum(row))

    plt.matshow(new_conf_arr)
    plt.yticks(range(25), dico.keys())
    plt.xticks(range(25), dico.keys(), rotation=90)
    plt.colorbar()
    plt.show()

def get_per_class_accuracy(labels,preds):
    names = []
    accs = []
    dico = get_dico()
    inv_dico = invert_dico(dico)
    for value in set(labels):
        s = 0
        n = 0
        for i in range(len(labels)):
            if (labels[i] == value):
                n = n + 1
                if (preds[i] == value):
                    s = s + 1
        names.append(inv_dico.get(value))
        accs.append(s / n * 100,)
    return accs,names

if __name__ == '__main__':

    main()
