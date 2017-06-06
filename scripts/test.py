from nets.alexnet import Alexnet
from nets.processing import train_model_from_directory
import os
from os.path import join



PATH = os.path.dirname(__file__)
model = Alexnet(join(PATH,'nets/weights/alexnet_weights.h5'),nb_classes=6,training_mode='full')

train_model_from_directory(join(PATH,'../data/pandora') ,model,epochs=1,steps_per_epoch=1,target_size=(227,227))