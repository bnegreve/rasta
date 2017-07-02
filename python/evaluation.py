from keras.models import load_model
from keras import backend as K
from keras.layers import Dense
from keras.models import Model
from models.alexnet import decaf
from keras import metrics
import numpy as np
from PIL import Image as image
import os
from os.path import join
import argparse
from progressbar import ProgressBar
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import json

def main():
    PATH = os.path.dirname(__file__)

    K.set_image_data_format('channels_last')

    parser = argparse.ArgumentParser(description='Description')

    parser.add_argument('-t', action="store", default='acc', dest='type', help='Type of evaluation [pred|acc]')
    parser.add_argument('--isdecaf', action="store", default=False, type=bool, dest='isdecaf',
                        help='if the model is a decaf6 type')
    parser.add_argument('-k', action="store", default=1, type=str, dest='k', help='top-k number')
    parser.add_argument('--data_path', action="store",
                        default=join(PATH, '../data/wikipaintings_10/wikipaintings_test'), dest='data_path',
                        help='Path of the data (image or train folder)')
    parser.add_argument('--model_path', action="store", default=None, dest='model_path',
                        help='Path of the h5 model file')
    parser.add_argument('-j', action="store_true", dest='json', help='Output prediction as json')

    args = parser.parse_args()

    eval_type = args.type

    model_path = args.model_path
    data_path = args.data_path
    isdecaf = args.isdecaf
    k = (str(args.k)).split(",")
    k = [int(val) for val in k]
    print(k)
    if model_path == None:
        print('Please run providing argument --model_path')
    else:
        if eval_type == 'acc':
            preds = get_top_multi_acc(model_path, data_path,top_k=k)
            for val,pred in zip(k,preds):
                print('\nTop-{} accuracy : {}%'.format(val,pred*100))
        elif eval_type == 'pred':
            k = k[0]
            pred = get_pred(model_path, data_path, is_decaf6=isdecaf, top_k=k)
            if args.json:
                result = { 'pred' : pred, 'k' : k }
                print(json.dumps(result))
            else:
                print("Top-{} prediction : {}".format(k, pred))
        else:
            print('Error in arguments. Please try with -h')


def get_dico():
    classes = []
    PATH = os.path.dirname(__file__)
    directory = join(PATH,'../data/wikipaintings_10/wikipaintings_train')
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)
    class_indices = dict(zip(classes, range(len(classes))))
    return class_indices


def get_test_accuracy(model_path, test_data_path, is_decaf6=False,top_k=1):
    y_pred, y = get_y_pred(model_path, test_data_path, is_decaf6,top_k=top_k)
    score = 0
    for pred, val in zip(y_pred, y):
        if val in pred:
            score += 1
    return score / len(y)


def get_y_pred(model_path, test_data_path, is_decaf6=False,top_k=1):

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
    dico = get_dico()
    y = []
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
            img = image.open(join(style_path, img_name))
            img = img.resize((227, 227))
            img_np = np.asarray(img, dtype='uint8')
            img_np = np.divide(img_np, 255)
            x = img_np[..., np.newaxis]
            x = x.transpose(3, 0, 1,2)
            if is_decaf6:
                x = x.transpose(0,3,2,1)
            pred = model.predict(x)
            args_sorted = np.argsort(pred)[0][::-1]
            y.append(label)
            y_pred.append([a for a in args_sorted[:top_k]])
            i += 1
            bar.update(i)
    return np.asarray(y_pred), y


def get_pred(model_path, image_path, is_decaf6=False, top_k=1):
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
    img = image.open(image_path)
    img = img.resize((227, 227))
    img_np = np.asarray(img, dtype='uint8')
    img_np = np.divide(img_np, 255)
    x = img_np[..., np.newaxis]
    x = x.transpose(3,0, 1,2)
    if is_decaf6:
       x = x.transpose(0, 3, 2, 1)
    pred = model.predict(x)
    dico = get_dico()
    inv_dico = {v: k for k, v in dico.items()}
    args_sorted = np.argsort(pred)[0][::-1]
    return [inv_dico.get(a) for a in args_sorted[:top_k]]


def get_top_multi_acc(model_path, test_data_path, is_decaf6=False,top_k=[1,3,5]):
    y_pred, y = get_y_pred(model_path, test_data_path, is_decaf6, max(top_k))
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
    inv_dico = {v: k for k, v in dico.items()}
    for value in set(labels):
        s = 0
        n = 0
        for i in range(len(labels)):
            if (labels[i] == value):
                n = n + 1
                if (preds[i] == value):
                    s = s + 1
        names.append(inv_dico.get(value))
        accs.append(s / n * 100, )

if __name__ == '__main__':
    main()