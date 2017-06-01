from keras.layers import Flatten, Dense, Dropout,Input,merge,Activation,Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential,Model
from keras import backend as K
from sklearn.model_selection import train_test_split
import h5py


def my_get_model():
  
    model = Sequential()

    #Layer 1
    model.add(Conv2D(96, 11, 11, subsample=(4, 4),input_shape=(32,32,3), activation='relu',name='conv_1', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Layer 2
    model.add(Conv2D(256, (5, 5), activation='relu',  padding='same',name='conv_2_1'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Layer 3
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu',  padding='same',name='conv_3'))

    #Layer 4
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(384, (3, 3), activation='relu',  padding='same',name='conv_4_1'))

    #Layer 5
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same',name='conv_5_1'))
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

def splittensor(axis=1, ratio_split=1, id_split=0):
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

def get_model2(nb_classes=12):




    inputs = Input(shape=(3,256,256))


    conv_1 = Conv2D(96, (11, 11), strides=(4, 4), activation='relu',name='conv_1', kernel_initializer='he_normal')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    #conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)

    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = merge([Conv2D(128, (5, 5), activation="relu", kernel_initializer='he_normal', name='conv_2_' + str(i + 1))(splittensor(ratio_split=2, id_split=i)(conv_2)) for i in range(2)], mode='concat', concat_axis=1, name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    #conv_3 = crosschannelnormalization()(conv_3)
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

def get_model(nb_classes=12,training_mode='full'):
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
    else:
        print('Bad training_mode argument. Should be : [ full | partial_1 | partial_2 | partial_3 ]')

    inputs = Input(shape=(3,256,256))


    conv_1 = Conv2D(96, (11, 11), strides=(4, 4), activation='relu',name='conv_1', kernel_initializer='he_normal',trainable=trainable_conv)(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    #conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)

    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = merge([Conv2D(128, (5, 5), activation="relu", kernel_initializer='he_normal', name='conv_2_' + str(i + 1),trainable=trainable_conv)(splittensor(ratio_split=2, id_split=i)(conv_2)) for i in range(2)], mode='concat', concat_axis=1, name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    #conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, 3, 3, activation='relu', name='conv_3', init='he_normal',trainable=trainable_conv)(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = merge([Conv2D(192, 3, 3, activation="relu", init='he_normal', name='conv_4_' + str(i + 1),trainable=trainable_conv)(splittensor(ratio_split=2, id_split=i)(conv_4)) for i in range(2)], mode='concat', concat_axis=1, name="conv_4")

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = merge([Conv2D(128, 3, 3, activation="relu", init='he_normal', name='conv_5_' + str(i + 1),trainable=trainable_conv)(splittensor(ratio_split=2, id_split=i)(conv_5)) for i in range(2)], mode='concat', concat_axis=1, name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)

    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1', init='he_normal',trainable=trainable_dense_1)(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2', init='he_normal',trainable=trainable_dense_2)(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(nb_classes, name='dense_3_new', init='he_normal',trainable=trainable_dense_3)(dense_3)

    prediction = Activation("softmax", name="softmax")(dense_3)

    alexnet = Model(input=inputs, output=prediction)

    alexnet.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

    return alexnet

def crosschannelnormalization(alpha = 1e-4, k=2, beta=0.75, n=5,**kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet
    """
    def f(X):
        b, ch, r, c = X.shape
        ch = int(ch)
        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0,2,3,1)), ((0,half),(0,half)))
        extra_channels = K.permute_dimensions(extra_channels, (0,3,1,2))
        scale = k
        for i in range(n):
            scale += alpha * extra_channels[:,i:i+ch,:,:]
        scale = scale ** beta
        return X / scale

    return Lambda(f)


def load_pandora():
    print('Loading data...')
    with h5py.File('../datasets/pandora.h5') as f:
        x = f['dataset'][:]
        y = f['labels'][:]
    return x,y


def set_imageNet_weights(model):
    model.load_weights('weights/alexnet_weights.h5', by_name=True)
    return model

if __name__ == '__main__':
    x,y = load_panda()

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)

    model = get_model()
    model = set_imageNet_weights(model)

    model.fit(X_train, y_train, batch_size=32, epochs=1, validation_split=0.2)

    score = model.evaluate(X_test, y_test, batch_size=32)

    print('The test score is : ',score)