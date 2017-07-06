from keras.models import load_model
from keras import backend as K
import sys
from keras.preprocessing.image import img_to_array
from evaluation import get_dico
from keras.layers.convolutional import _Conv
import argparse
import os
from os.path import join
import numpy as np
from matplotlib import pyplot as plt
from vis.visualization import visualize_cam,overlay,visualize_saliency
from vis.utils import utils
from keras import activations



class HeatMapper(object):

    def __init__(self,model_path,img_path):
        K.set_image_data_format('channels_last')
        self.model_path = model_path
        self.img_path = img_path
        self.model = load_model(model_path)
        self.inv_dico = {v: k for k, v in get_dico().items()}

        self.seed_img = utils.load_img(img_path, target_size=(224, 224))
        x = np.expand_dims(img_to_array(self.seed_img), axis=0)
        img_np = np.asarray(self.seed_img, dtype='uint8')
        img_np = np.divide(img_np, 255)
        x = img_np[..., np.newaxis]
        x = x.transpose(3, 0, 1, 2)
        self.x = x

    def summary(self):
        return self.model.summary

    def predictions(self):
        return

    def predict_class(self):
        return

    def plot_convs_heatmap(self):
        layer_idx = -1

        self.model.layers[layer_idx].activation = activations.linear
        model = utils.apply_modifications(self.model)
        names = []
        for layer in self.model.layers:
            if isinstance(layer, _Conv):
                names.append(layer.name)

        pred_class =  np.argmax(self.model.predict(self.x))
        fig = plt.figure()

        for i in range(len(names))[-5:]:
            name = names[i]

            print('Calculating heatmap for ', name)
            penult_layer_idx = utils.find_layer_idx(model, name)
            heatmap = visualize_cam(model, layer_idx, filter_indices=[pred_class], seed_input=self.seed_img,penultimate_layer_idx=penult_layer_idx, backprop_modifier=None)
            sp = fig.add_subplot(6, 9, i + 1)
            sp.title.set_text(name)
            sp.imshow(overlay(self.seed_img, heatmap))
        plt.show()

    def plot_last_heatmap(self):
        layer_idx = -1
        pred_class =  np.argmax(self.model.predict(self.x))
        self.model.layers[layer_idx].activation = activations.linear
        model = utils.apply_modifications(self.model)
        names =[]
        for layer in self.model.layers:
            if isinstance(layer, _Conv):
                names.append(layer.name)
        penult_layer_idx = utils.find_layer_idx(model, names[-1])
        print(names[-1])
        heatmap = visualize_cam(model, layer_idx, filter_indices=[pred_class], seed_input=self.seed_img,penultimate_layer_idx=penult_layer_idx, backprop_modifier=None)
        plt.imshow(overlay(self.seed_img, heatmap))
        plt.show()

if __name__ == '__main__':
    PATH = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description='Tool to plot the heatmap.')
    parser.add_argument('--data_path', action="store",default=join(PATH, '../data/wikipaintings_10/wikipaintings_test/Cubism/albert-gleizes_on-a-sailboat.jpg'), dest='data_path',help='Path of the image')
    parser.add_argument('--model_path', action="store", default=None, dest='model_path',help='Path of the h5 model file',required=True)
    args = parser.parse_args()

    hm = HeatMapper(args.model_path, args.data_path)
    hm.plot_last_heatmap()