import pickle

from keras.layers import Flatten, Dense, Dropout,Input,merge,Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential,Model

import h5py

from utils import load_data as ld


def get_model(weights_path=None):
  
    model = Sequential()

    #Layer 1
    model.add(Convolution2D(96, 11, 11, subsample=(4, 4),input_shape=(32,32,3), activation='relu',name='conv_1', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Layer 2
    model.add(Convolution2D(256, (5, 5), activation='relu',  padding='same',name='conv_2_1'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Layer 3
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu',  padding='same',name='conv_3'))

    #Layer 4
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(384, (3, 3), activation='relu',  padding='same',name='conv_4_1'))

    #Layer 5
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same',name='conv_5_1'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Layer 6
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', init='glorot_normal',name='dense_1'))
    model.add(Dropout(0.5))

    #Layer 7
    model.add(Dense(4096, activation='relu', init='glorot_normal',name='dense_2'))
    model.add(Dropout(0.5))

    #Layer 8
    model.add(Dense(12, activation='softmax', init='glorot_normal',name='dense_3'))
    


    return model


def splittensor(axis=1, ratio_split=1, id_split=0,**kwargs):
    def f(X):
        div = X.shape[axis] // ratio_split

        if axis == 0:
            output =  X[id_split*div:(id_split+1)*div,:,:,:]
        elif axis == 1:
            output =  X[:, id_split*div:(id_split+1)*div, :, :]
        elif axis == 2:
            output = X[:,:,id_split*div:(id_split+1)*div,:]
        elif axis == 3:
            output = X[:,:,:,id_split*div:(id_split+1)*div]
        else:
            raise ValueError("This axis is not possible")

        return output

def get_model2(nb_classes =12):
      # code adapted from https://github.com/heuritech/convnets-keras

    inputs = Input(shape=(32,32,3))


    conv_1 = Conv2D(96, (11, 11), strides=(4, 4), activation='relu',name='conv_1', kernel_initializer='he_normal')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = merge([Conv2D(128, (5, 5), activation="relu", kernel_initializer='he_normal', name='conv_2_' + str(i + 1))(splittensor(ratio_split=2, id_split=i)(conv_2)) for i in range(2)], mode='concat', concat_axis=1, name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, 3, 3, activation='relu', name='conv_3', init='he_normal')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = merge([Conv2D(192, 3, 3, activation="relu", init='he_normal', name='conv_4_' + str(i + 1))(splittensor(ratio_split=2, id_split=i)(conv_4)) for i in range(2)], mode='concat', concat_axis=1, name="conv_4")

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = merge([Conv2D(128, 3, 3, activation="relu", init='he_normal', name='conv_5_' + str(i + 1))(splittensor(ratio_split=2, id_split=i)(conv_5)) for i in range(2)], mode='concat', concat_axis=1, name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)

    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1', init='he_normal')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2', init='he_normal')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(nb_classes, name='dense_3_new', init='he_normal')(dense_3)

    prediction = Activation("softmax", name="softmax")(dense_3)

    alexnet = Model(input=inputs, output=prediction)

    return alexnet

if __name__ == '__main__':
    with open('../dataset.pick','rb') as f:
        x = dataset = pickle.load(f)
    with open('../labels.pick', 'rb') as f:
        y = pickle.load(f)

    print('Loading model')
    model = get_model2()
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    #print('Running model')
    #model.fit(x, y, batch_size=32, epochs=1, validation_split=0.1)
    x_test = x
    y_test = y
    model.load_weights('weights/alexnet_weights.h5', by_name=True)
    score = model.evaluate(x_test, y_test, batch_size=32)