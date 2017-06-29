from keras.layers import Flatten, Dense, Dropout,Input,merge,Activation,Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import backend as K
import os,sys




def Alexnet(weights=None,nb_classes = 12,training_mode='full'):
    trainable_conv = False
    trainable_dense_1 = False
    trainable_dense_2 = False
    trainable_dense_3 = False
    if training_mode=='partial_1':
        trainable_dense_3 = True
    elif training_mode=='partial_2':
        trainable_dense_3 = True
        trainable_dense_2 = True

    elif training_mode=='partial_3':
        trainable_dense_3 = True
        trainable_dense_2 = True
        trainable_dense_1 = True
    elif training_mode=='full':
        trainable_dense_3 = True
        trainable_dense_2 = True
        trainable_dense_1 = True
        trainable_conv = True
    else:
        print('Bad training_mode argument. Should be : [ full | partial_1 | partial_2 | partial_3 ]')

    inputs = Input(shape=(3,227,227))


    conv_1 = Conv2D(96, (11, 11), strides=(4, 4), activation='relu',name='conv_1', kernel_initializer='he_normal',trainable=trainable_conv)(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = _crosschannelnormalization(name="convpool_1")(conv_2)

    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = merge([Conv2D(128, (5, 5), activation="relu", kernel_initializer='he_normal', name='conv_2_' + str(i + 1),trainable=trainable_conv)(_splittensor(ratio_split=2, id_split=i)(conv_2)) for i in range(2)], mode='concat', concat_axis=1, name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = _crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, 3, 3, activation='relu', name='conv_3', init='he_normal',trainable=trainable_conv)(conv_3)

    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = merge([Conv2D(192, 3, 3, activation="relu", init='he_normal', name='conv_4_' + str(i + 1),trainable=trainable_conv)(_splittensor(ratio_split=2, id_split=i)(conv_3)) for i in range(2)], mode='concat', concat_axis=1, name="conv_4")

    conv_4 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = merge([Conv2D(128, 3, 3, activation="relu", init='he_normal', name='conv_5_' + str(i + 1),trainable=trainable_conv)(_splittensor(ratio_split=2, id_split=i)(conv_4)) for i in range(2)], mode='concat', concat_axis=1, name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)

    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1', init='he_normal',trainable=trainable_dense_1)(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2', init='he_normal',trainable=trainable_dense_2)(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(nb_classes, name='dense_3_new', init='he_normal',trainable=trainable_dense_3)(dense_3)

    prediction = Activation("softmax", name="softmax")(dense_3)

    alexnet = Model(input=inputs, output=prediction)

    if weights!=None:
        alexnet.load_weights(weights, by_name=True)
    
    return alexnet


def decaf(weights=None,rank=6):
    inputs = Input(shape=(3,227,227))

    conv_1 = Conv2D(96, (11, 11), strides=(4, 4), activation='relu',name='conv_1', kernel_initializer='he_normal',trainable=False)(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = _crosschannelnormalization(name="convpool_1")(conv_2)

    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = Concatenate(axis=1,name='conv_2')([Conv2D(128, (5, 5), activation="relu", kernel_initializer='he_normal', name='conv_2_' + str(i + 1),trainable=False)(_splittensor(ratio_split=2, id_split=i)(conv_2)) for i in range(2)])

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = _crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, (3, 3), activation='relu', name='conv_3', kernel_initializer='he_normal',trainable=False)(conv_3)

    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = Concatenate(axis=1,name='conv_4')([Conv2D(192, (3, 3), activation="relu", kernel_initializer='he_normal', name='conv_4_' + str(i + 1),trainable=False)(_splittensor(ratio_split=2, id_split=i)(conv_3)) for i in range(2)])

    conv_4 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = Concatenate(axis=1,name='conv_5')([Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal', name='conv_5_' + str(i + 1),trainable=False)(_splittensor(ratio_split=2, id_split=i)(conv_4)) for i in range(2)])

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)
    dense_1 = Flatten(name="flatten")(dense_1)

    
    if rank!=5 :
        dense_1 = Dense(4096, kernel_initializer='he_normal',trainable=False)(dense_1)
        if rank == 7 :
            dense_1 =  Dense(4096, kernel_initializer='he_normal',trainable=False)(dense_1)
            
    last = dense_1
    prediction = Activation("relu", name="relu")(last)

    alexnet = Model(input=inputs, output=prediction)
    if weights!=None:
        alexnet.load_weights(weights, by_name=True)
    
    return alexnet    


def _splittensor(axis=1, ratio_split=1, id_split=0):
    def split_function(X):
        div = int(X.shape[axis] // ratio_split)

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

    return Lambda(split_function)

def _crosschannelnormalization(alpha = 1e-4, k=2, beta=0.75, n=5,**kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet
    """
    def f(X):
        b, ch, r, c = X.shape
        ch = int(ch)
        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0,2,3,1)), ((0,0),(half,half)))
        extra_channels = K.permute_dimensions(extra_channels, (0,3,1,2))
        scale = k
        for i in range(n):
            scale += alpha * extra_channels[:,i:i+ch,:,:]
        scale = scale ** beta
        return X / scale

    return Lambda(f)

if __name__ == '__main__':
    K.set_image_data_format('channels_first')
    model = decaf6()
    with open('./summary.txt', 'w') as f:
        orig_stdout = sys.stdout
        sys.stdout = f
        print(model.summary())
        sys.stdout = orig_stdout
        f.close()
