from keras import backend as K
from keras.models import load_model
from keras.applications import ResNet50,imagenet_utils
from evaluation import get_dico
from keras.layers.convolutional import _Conv
import argparse
import os
from os.path import join
import numpy as np
from vis.visualization import visualize_cam,overlay,visualize_saliency,visualize_activation
from vis.utils import utils
from keras import activations
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from vis.visualization import get_num_filters
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt




class HeatMapper(object):

    def __init__(self,model_path,img_path):
        K.set_image_data_format('channels_last')
        self.model_path = model_path
        self.img_path = img_path
        self.model = load_model(model_path)
        self.inv_dico = {v: k for k, v in get_dico().items()}

        self.seed_img = utils.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(self.seed_img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        self.x = x

    def summary(self):
        return self.model.summary()

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

        for i in range(len(names)):
            name = names[i]

            print('Calculating heatmap for ', name)
            penult_layer_idx = utils.find_layer_idx(model, name)
            heatmap = visualize_cam(model, layer_idx, filter_indices=[pred_class], seed_input=self.seed_img,penultimate_layer_idx=penult_layer_idx, backprop_modifier=None)
            sp = fig.add_subplot(6, 9, i + 1)
            sp.set_title(name,fontsize=7)
            sp.imshow(overlay(self.seed_img, heatmap))
            sp.get_xaxis().set_visible(False)
            sp.get_yaxis().set_visible(False)

        plt.show()

    def plot_last_heatmap(self):
        layer_idx = -1
        pred_class =  np.argmax(self.model.predict(self.x))
        print(pred_class)
        print(imagenet_utils.decode_predictions(self.model.predict(self.x)))

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

    def plot_activation(self):
        layer_idx=-1
        self.model.layers[layer_idx].activation = activations.linear
        model = utils.apply_modifications(self.model)
        for pred_class in range(25)[11:12]:
            print(self.inv_dico.get(pred_class))
            actmap = visualize_activation(model,layer_idx,filter_indices=pred_class)
            plt.imsave('img.jpg',actmap)
            #sp = fig.add_subplot(5, 5, pred_class + 1)
            #sp.imshow(actmap)
            #sp.set_title(self.inv_dico.get(pred_class),fontsize=7)
            #sp.get_xaxis().set_visible(False)
            #sp.get_yaxis().set_visible(False)

    def plot_conv_weights(self):
        PATH = os.path.dirname(__file__)
        cls = _get_conv_layers(self.model)
        i=0
        for layer in cls :
            layer_name = layer.name
            print("{}/{} : {}".format(i,len(cls),layer_name))
            layer_idx = utils.find_layer_idx(self.model,layer_name)

            filters = np.arange(get_num_filters(self.model.layers[layer_idx]))
            vis_images = []
            for idx in filters[:64]:
                img = visualize_activation(self.model, layer_idx, filter_indices=idx)
                # Utility to overlay text on image.
                img = utils.draw_text(img, 'Filter {}'.format(idx))
                vis_images.append(img)
            # Generate stitched image palette with 8 cols.
            stitched = utils.stitch_images(vis_images, cols=8)
            plt.axis('off')
            plt.imsave(join(PATH,'heatmaps/'+layer_name+'.jpg'),stitched)
            i+=1

def _get_conv_layers(model):
    res = []
    for layer in model.layers:
        if  isinstance(layer,_Conv):
            res.append(layer)
    return res

if __name__ == '__main__':
    PATH = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description='Tool to plot the heatmap.')
    parser.add_argument('--data_path', action="store",default=join(PATH, '../data/wikipaintings_10/wikipaintings_test/Cubism/albert-gleizes_on-a-sailboat.jpg'), dest='data_path',help='Path of the image')
    parser.add_argument('--model_path', action="store", default=None, dest='model_path',help='Path of the h5 model file',required=True)
    args = parser.parse_args()

    hm = HeatMapper(args.model_path, args.data_path)
    hm.plot_conv_weights()