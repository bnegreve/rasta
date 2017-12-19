import os
from os.path import join


#PREPROCESSING FUNCTIONS

def imagenet_preprocess_input(x):
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]
    # Zero-center by mean pixel
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    return x

def wp_preprocess_input(x):
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]

    x[:,:,0] -=  133.104
    x[:,:,0] -=  119.973
    x[:,:,0] -=  104.432

    return x

def custom_preprocess_input(x):
    x = x[:, :, ::-1]
    # Zero-center by mean pixel
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    x *= 1./255


#DICO FUNCTIONS

def get_dico():
    classes = []
    PATH = os.path.dirname(__file__)
    directory = join(PATH,'../../data/wikipaintings_small/wikipaintings_train')
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)
    class_indices = dict(zip(classes, range(len(classes))))
    return class_indices

def invert_dico(dico):
    return {v: k for k, v in dico.items()}
