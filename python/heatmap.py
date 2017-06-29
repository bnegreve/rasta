from keras.models import load_model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import sys
from keras.layers import Dense
from keras.models import Model
from models.alexnet import decaf6
import cv2
import numpy as np
from scipy.misc import imresize


import numpy as np
from matplotlib import pyplot as plt

from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input

from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_saliency



def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

K.set_image_data_format('channels_first')

model_path = sys.argv[1]
img_path = sys.argv[2]


base_model =decaf6()
predictions = Dense(25, activation='softmax',name='predictions')(base_model.output)
model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights(model_path)


print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'predictions'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

# Images corresponding to tiger, penguin, dumbbell, speedboat, spider

heatmaps = []
image=cv2.imread(img_path)
seed_img = np.transpose(imresize(image,(227,227)))
seed_img = seed_img[:,np.newaxis]
seed_img
pred_class = np.argmax(model.predict(seed_img))

    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
heatmap = visualize_saliency(model, layer_idx, [pred_class], seed_img)
heatmaps.append(heatmap)

plt.axis('off')
plt.imshow(utils.stitch_images(heatmaps))
plt.title('Saliency map')
plt.show()