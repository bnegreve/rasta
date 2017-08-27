from sklearn.metrics import confusion_matrix
from utils.utils import get_dico
import matplotlib.pyplot as plt
import keras.backend as K
import argparse
from evaluation import get_y_pred
import pickle

def main():
    K.set_image_data_format('channels_last')
    parser = argparse.ArgumentParser(description='Description')

    parser.add_argument('--isdecaf', action="store_true", dest='isdecaf',
                        help='if the model is a decaf6 type')
    parser.add_argument('-k', action="store", default='1,3,5', type=str, dest='k', help='top-k number')
    parser.add_argument('--data_path', action="store",required=True, dest='data_path', help='Path of the data (image or train folder)')
    parser.add_argument('--model_path', action="store", dest='model_path',
                        help='Path of the h5 model file',required=True)
    parser.add_argument('-b', action="store_true", dest='b', help='Sets bagging')
    parser.add_argument('-p', action="store", dest='preprocessing', help='Type of preprocessing : [imagenet|wp]')


    args = parser.parse_args()


    model_path = args.model_path
    data_path = args.data_path
    isdecaf = args.isdecaf
    preds,labels = get_y_pred(model_path, data_path, is_decaf6=isdecaf,top_k=1,bagging = args.b,preprocessing=args.preprocessing)

    with open('labels.pick','wb') as f:
        pickle.dump(labels,f)
    with open('preds.pick', 'wb') as f:
        pickle.dump(preds, f)


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
    plt.savefig('./conf_mat.eps',format='eps', dpi=1000,bbox_inches='tight')

if __name__== '__main__':
    main()