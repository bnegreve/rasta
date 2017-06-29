from models.alexnet import decaf
from models.processing import train_model_from_directory
from keras.layers import Dense
from keras.models import Model
from keras.layers.pooling import GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras import backend as K
import os
from os.path import join
import argparse

PATH = os.path.dirname(__file__)


parser = argparse.ArgumentParser(description='Description')

parser.add_argument('-m', action="store", default='decaf6',dest='model_name',help='Name of the model [decaf|decaf6|decaf7|resnet|resnetfull|resnetfullempty]')
parser.add_argument('-b', action="store", default=32, type=int,dest='batch_size',help='Size of the batch.')
parser.add_argument('-e', action="store",default=10,type=int,dest='epochs',help='Number of epochs')
parser.add_argument('-f', action="store", default=False, type=bool,dest='horizontal_flip',help='Set horizontal flip or not [True|False]')
parser.add_argument('--train_path', action="store", default=join(PATH, '../data/wikipaintings_10/wikipaintings_train'),dest='training_path',help='Path of the training data directory')
parser.add_argument('--val_path', action="store", default=join(PATH, '../data/wikipaintings_10/wikipaintings_val'),dest='validation_path',help='Path of the validation data directory')


args = parser.parse_args()

model_name = args.model_name
batch_size = args.batch_size
epochs = args.epochs
flip = args.horizontal_flip
TRAINING_PATH = args.training_path
VAL_PATH = args.validation_path

K.set_image_data_format('channels_first')
size = (227,227)

WEIGHTS_PATH = join(PATH, 'models/weights/alexnet_weights.h5')

if model_name =='decaf5':
    WEIGHTS_PATH = join(PATH,'models/weights/alexnet_weights.h5')
    base_model = decaf(WEIGHTS_PATH,5)
    predictions = Dense(25, activation='softmax')(base_model.output)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

elif model_name =='decaf6':
    base_model = decaf(WEIGHTS_PATH,6)
    predictions = Dense(25, activation='softmax')(base_model.output)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False


elif model_name =='decaf7':
    base_model = decaf(WEIGHTS_PATH,7)
    predictions = Dense(25, activation='softmax')(base_model.output)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

elif model_name =='resnet':
    K.set_image_data_format('channels_last')
    base_model = ResNet50(include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(25, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    size = (224,224)

elif model_name == 'resnetfull':
    K.set_image_data_format('channels_last')
    base_model = ResNet50(include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(25, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    list_trainable = ['dense_1','bn5c_branch2c','res5c_branch2c','bn5c_branch2b','res5c_branch2b','bn5c_branch2a','res5c_branch2a']
    for layer in base_model.layers:
        if layer.name in list_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    size = (224,224)

elif model_name == 'resnetfullempty':
    K.set_image_data_format('channels_last')
    base_model = ResNet50(include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(25, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = True
    size = (224,224)

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
train_model_from_directory(TRAINING_PATH,model,model_name=model_name,target_size=size,validation_path=VAL_PATH,epochs = epochs,batch_size = batch_size,horizontal_flip=flip)
