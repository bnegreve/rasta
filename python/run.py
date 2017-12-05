from models.alexnet import decaf,alexnet
from models.processing import train_model_from_directory
from models.custom_resnets import *
from models.inceptionV4 import inception_v4
from keras.layers import Dense
from keras.models import Model
from keras import backend as K
import os
from os.path import join
import argparse

PATH = os.path.dirname(__file__)

# PARSING ARGUMENTS

parser = argparse.ArgumentParser(description='Description')

parser.add_argument('-m', action="store", default='resnet',dest='model_name',help='Name of the model [alexnet_empty|decaf6|resnet|inception|inceptionv4|resnet2|empty_resnet|resnet_dropout|resnet_18|resnet_34|resnet_101|resnet_152|custom_resnet')
parser.add_argument('-b', action="store", default=32, type=int,dest='batch_size',help='Size of the batch.')
parser.add_argument('-e', action="store",default=10,type=int,dest='epochs',help='Number of epochs')
parser.add_argument('-f', action="store", default=False, type=bool,dest='horizontal_flip',help='Set horizontal flip or not [True|False]')
parser.add_argument('-n', action="store", default=0, type=int,dest='n_layers_trainable',help='Set the number of last trainable layers')
parser.add_argument('-d', action="store", default=0, type=float,dest='dropout_rate',help='Set the dropout_rate')

parser.add_argument('-p', action="store",dest='preprocessing',help='Set imagenet preprocessing or not')

parser.add_argument('--distortions', action="store", type=float,dest='disto',default=0.,help='Activate distortions or not')

parser.add_argument('--train_path', action="store", default=join(PATH, '../data/wikipaintings_10/wikipaintings_train'),dest='training_path',help='Path of the training data directory')
parser.add_argument('--val_path', action="store", default=join(PATH, '../data/wikipaintings_10/wikipaintings_val'),dest='validation_path',help='Path of the validation data directory')



args = parser.parse_args()

model_name = args.model_name
batch_size = args.batch_size
epochs = args.epochs
flip = args.horizontal_flip
TRAINING_PATH = args.training_path
VAL_PATH = args.validation_path
n_layers_trainable = args.n_layers_trainable
dropout_rate = args.dropout_rate

params = vars(args)

# BUILDING MODEL


if model_name =='alexnet_empty':
    K.set_image_data_format('channels_first')
    size = (227, 227)
    model = alexnet(weights=None)
    for layer in model.layers:
        layer.trainable = True

elif model_name =='decaf6':
    K.set_image_data_format('channels_first')
    size = (227, 227)
    base_model = decaf()
    predictions = Dense(25, activation='softmax')(base_model.output)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

elif model_name =='resnet':
    K.set_image_data_format('channels_last')
    size = (224,224)

    base_model = resnet_trained(n_layers_trainable)
    predictions = Dense(25, activation='softmax')(base_model.output)
    model = Model(inputs=base_model.input, outputs=predictions)

elif model_name =='inception':
    K.set_image_data_format('channels_last')
    size = (224,224)

    base_model = inception(n_layers_trainable)
    predictions = Dense(25, activation='softmax')(base_model.output)
    model = Model(inputs=base_model.input, outputs=predictions)

elif model_name =='inceptionv4':
    K.set_image_data_format('channels_last')
    size = (299,299)
    model = inception_v4()


elif model_name=='resnet2':
    K.set_image_data_format('channels_last')
    size=(224,224)
    base_model = resnet_trained_2(n_layers_trainable)
    predictions = Dense(25, activation='softmax')(base_model.output)
    model = Model(inputs=base_model.input, outputs=predictions)

elif model_name =='empty_resnet':
    K.set_image_data_format('channels_last')
    size = (224,224)

    base_model = empty_resnet()
    predictions = Dense(25, activation='softmax')(base_model.output)
    model = Model(inputs=base_model.input, outputs=predictions)

elif model_name=='resnet_dropout':
    K.set_image_data_format('channels_last')
    size = (224, 224)
    base_model = resnet_dropout(dp_rate=dropout_rate,n_retrain_layers=n_layers_trainable)
    predictions = Dense(25, activation='softmax')(base_model.output)
    model = Model(inputs=base_model.input, outputs=predictions)

elif model_name=='resnet_18':
    size = (224, 224)
    K.set_image_data_format('channels_last')
    model =  resnet18()

elif model_name=='resnet_34':
    size = (224, 224)
    K.set_image_data_format('channels_last')
    model =  resnet34()

elif model_name=='resnet_101':
    size = (224, 224)
    K.set_image_data_format('channels_last')
    model =  resnet101()

elif model_name=='resnet_152':
    size = (224, 224)
    K.set_image_data_format('channels_last')
    model =  resnet152()
elif model_name == 'custom_resnet':
    size = (224, 224)
    K.set_image_data_format('channels_last')
    model = custom_resnet(dp_rate=dropout_rate)
else:
    print("The model name doesn't exist")

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
train_model_from_directory(TRAINING_PATH,model,model_name=model_name,target_size=size,validation_path=VAL_PATH,epochs = epochs,batch_size = batch_size,horizontal_flip=flip,params=params,preprocessing=args.preprocessing,distortions=args.disto)
