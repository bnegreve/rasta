from keras.layers.convolutional import Conv2D
from models.alexnet import Alexnet,decaf
from models.processing import train_model_from_directory
from keras.layers import Dense
from keras.models import Model,Sequential
from keras.layers.pooling import GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras import backend as K
import os
import sys
from os.path import join



PATH = os.path.dirname(__file__)

vals = sys.argv

id = vals[1]
batch_size = int(vals[2]) 
epochs = int(vals[3])
steps = int(vals[4])
val_steps = int(vals[5])
flip = (vals[6]=='True')


K.set_image_data_format('channels_first')
size = (227,227)
TRAINING_PATH = '/home/alecoutre/rasta/data/wikipaintings_train'
VAL_PATH = '/home/alecoutre/rasta/data/wikipaintings_val'
WEIGHTS_PATH = join(PATH,'models/weights/alexnet_weights.h5')




if id=='decaf5':
    base_model = decaf(WEIGHTS_PATH,5)
    predictions = Dense(25, activation='softmax')(base_model.output)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

elif id=='decaf6':
    base_model = decaf(WEIGHTS_PATH,6)
    predictions = Dense(25, activation='softmax')(base_model.output)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False


elif id=='decaf7':
    base_model = decaf(WEIGHTS_PATH,7)
    predictions = Dense(25, activation='softmax')(base_model.output)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

elif id=='resnet':
    K.set_image_data_format('channels_last')
    base_model = ResNet50(include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(25, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    size = (224,224)

elif id == 'other':
    model = Sequential()
    model.add(Conv2D(128,(3,3)))
    model.add(Dense(1000, input_shape=(227,227), activation='relu', name='hidden_layer_1'))
    model.add(Dense(50, activation='relu', name='hidden_layer_2'))
    model.add(Dense(25, activation='softmax', name='predictions'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
train_model_from_directory(TRAINING_PATH,model,model_name=id,target_size=size,validation_path=VAL_PATH,epochs = epochs,batch_size = batch_size,steps_per_epoch=steps,validation_steps=val_steps,horizontal_flip=flip)
