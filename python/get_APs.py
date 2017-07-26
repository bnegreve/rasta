from keras.models import load_model,Model
from keras.layers import Dense
import keras.backend as K
from models.alexnet import decaf
from keras import metrics
from utils.utils import get_dico
import argparse
import os
from progressbar import ProgressBar
from os.path import join
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from sklearn.metrics import average_precision_score



def get_scores_labels(model, test_data_path, is_decaf=False):
    target_size = (224, 224)
    if is_decaf:
        target_size = (227, 227)
    dico = get_dico()
    y = []
    scores = []
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
            img = load_img(join(style_path, img_name), target_size=target_size)
            x = img_to_array(img)

            pred = model.predict(x[np.newaxis, ...])
            y.append(label)
            scores.append(pred[0])
            i += 1
            bar.update(i)

    index_shuf = range(len(y))
    y = [y[i] for i in index_shuf]
    scores = [scores[i] for i in index_shuf]
    return np.asarray(scores),y


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Description')

    parser.add_argument('--isdecaf', action="store_true", dest='isdecaf',help='if the model is a decaf6 type')
    parser.add_argument('-k', action="store", default='1,3,5', type=str, dest='k', help='top-k number')
    parser.add_argument('--data_path', action="store",required=True, dest='data_path',help='Path of the data (image or train folder)')
    parser.add_argument('--model_path', action="store", dest='model_path',help='Path of the h5 model file',required=True)
    #parser.add_argument('-p', action="store", dest='preprocessing', help='Type of preprocessing : [imagenet|wp]')

    args = parser.parse_args()





    if args.isdecaf:
        K.set_image_data_format('channels_first')
        base_model = decaf()
        predictions = Dense(25, activation='softmax')(base_model.output)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights(args.model_path, by_name=True)
    else :
        model=load_model(args.model_path)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy', metrics.top_k_categorical_accuracy])

    scores,labels = get_scores_labels(model,args.data_path,is_decaf=args.isdecaf)


    n_per_classes = 87
    new_scores = []
    new_labels = []
    for classe in set(labels):
        n_class = 0
        for i,label in enumerate(labels):
            if label==classe and n_class <n_per_classes:
                n_class +=1
                new_scores.append(scores[i,:])
                new_labels.append(label)
    print(len(new_scores),len(new_labels))




    dico = get_dico()
    inv_dico = {v: k for k, v in dico.items()}
    APs = []


    new_scores = np.asarray(new_scores)
    for classe in set(labels):
        temp_labels = []
        for i,label in enumerate(new_labels):
            if label==classe:
                temp_labels.append(1)
            else:
                temp_labels.append(0)
        temp_scores = new_scores[:, classe]
        score = average_precision_score(np.asarray(temp_labels), temp_scores)
        APs.append(score)
        print(np.sum(np.asarray(temp_labels)),len(temp_labels),len(temp_scores))
        print(inv_dico.get(classe),' : ',score)
    print('MEAN : ',np.mean(APs))