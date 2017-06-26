from keras.models import load_model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.models import Model
from models.alexnet import decaf
from keras import metrics
import numpy as np
from PIL import Image as image
import matplotlib.image as img
from scipy.misc import imresize
from matplotlib import pyplot as plt
import os
import sys


def get_test_accuracy(batch_size = 32,is_decaf6 = False):
	K.set_image_data_format('channels_first')

	model_path = sys.argv[1]
	test_data_path = sys.argv[2]

	if is_decaf6:
		base_model = decaf()
		predictions = Dense(25, activation='softmax')(base_model.output)
		model = Model(inputs=base_model.input, outputs=predictions)
		model.load_weights(model_path)
	else:
		model = load_model(model_path)

	model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

	test_datagen = ImageDataGenerator(rescale=1./255)

	test_generator = test_datagen.flow_from_directory(
		test_data_path,
		target_size=(227, 227),
		batch_size = batch_size,
		class_mode='categorical')

	return model.evaluate_generator(test_generator,8200//32)

def get_y_pred(batch_size = 32,is_decaf6 = False)
	K.set_image_data_format('channels_first')

	model_path = sys.argv[1]
	test_data_path = sys.argv[2]


	if is_decaf6:
		base_model =decaf()
		predictions = Dense(25, activation='softmax')(base_model.output)
		model = Model(inputs=base_model.input, outputs=predictions)
		model.load_weights(model_path,by_name=True)
	else:
		model = load_model(model_path)

	model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy',metrics.top_k_categorical_accuracy])

	test_datagen = ImageDataGenerator(rescale=1./255)

	test_generator = test_datagen.flow_from_directory(
		test_data_path,
		target_size=(227, 227),
		batch_size=1,
		class_mode='categorical')

	dico = test_generator.class_indices
	print(dico)

	y = []
	ap = []
	y_pred = []
	style_names = os.listdir(test_data_path)
	for style_name in style_names:
	    style_path = test_data_path+style_name
	    img_names = os.listdir(style_path)
	    size = len(img_names)
	    label = dico.get(style_name)
	    print("Class is : ",label)

	    for img_name in img_names:
		img = image.open(style_path+"/"+img_name)
		img = img.resize((227,227))
		img_np = np.asarray( img, dtype='uint8')
		img_np = np.divide(img_np,255)
		x = img_np[...,np.newaxis]
		x = x.transpose(3,0,1,2)
		pred = model.predict(x)
		predict = np.argmax(pred)
		y.append(label)
		y_pred.append(predict)

	return y_pred,y
