from keras.applications.resnet50 import ResNet50
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.layers import Dropout
from keras import backend as K
from resnet_build import ResnetBuilder



def resnet_trained(n_retrain_layers = 0):
    K.set_image_data_format('channels_last')
    base_model = ResNet50(include_top=False)
    features = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=features)
    split_value = len(base_model.layers)+1-n_retrain_layers
    for layer in model.layers[:split_value]:
        layer.trainable=False
    for layer in model.layers[split_value:]:
        layer.trainable = True
    return model


def empty_resnet():
    K.set_image_data_format('channels_last')
    base_model = ResNet50(weights=None,include_top=False)
    features = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=features)
    return model

def dropout_resnet(dropout_rate=0.5):
    K.set_image_data_format('channels_last')
    base_model = ResNet50(weights=None,include_top=False)
    features = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=features)
    dropout = Dropout(dropout_rate)(model.output)
    model = Model(inputs=model.input,outputs=dropout)
    return model


def resnet18():
    return ResnetBuilder.build_resnet_18((3,224,224),25)
def resnet34():
    return ResnetBuilder.build_resnet_34((3,224,224),25)
def resnet101():
    return ResnetBuilder.build_resnet_101((3,224,224),25)
def resnet152():
    return ResnetBuilder.build_resnet_152((3,224,224),25)

def main():
    model = resnet18()
    print(model.summary())


if __name__ == '__main__':
    main()