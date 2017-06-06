from keras.utils import plot_model
import os,datetime
from os.path import join
import sys
import pickle
import pydot
from keras.preprocessing.image import ImageDataGenerator


def train_model(X_train,y_train,model,saving=True,batch_size = 32, epochs = 10, validation_split = 0.2):
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    if saving:
        _save_model_data(model,history)
    return model

def train_model_from_directory(directory_path,model,saving=True,target_size =(256,256) ,batch_size = 32 ,horizontal_flip = True,epochs=10,steps_per_epoch=100):
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip = horizontal_flip)
    train_generator = train_datagen.flow_from_directory(directory_path, target_size = target_size, batch_size = batch_size, class_mode='categorical')
    history  = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs = epochs)
    if saving:
        _save_model_data(model,history)
    return model

def _save_model_data(model,history):
    now = datetime.datetime.now()
    model_name = 'alexnet_' + str(now.year) + '_' + str(now.month) + '_' + str(now.day) + ':' + str(now.hour) + str(now.minute) + str(now.second)
    SAVINGS_DIR = join(os.path.dirname(__file__),'../savings')
    os.makedirs(join(SAVINGS_DIR, model_name))
    _model_to_image(model, 'model',join(SAVINGS_DIR, model_name))
    with open(join(SAVINGS_DIR,model_name,'summary.txt'), 'w') as f:
        orig_stdout = sys.stdout
        sys.stdout = f
        print(model.summary())
        sys.stdout = orig_stdout
        f.close()
    with open(join(SAVINGS_DIR,model_name,'history.pck'), 'wb') as f:
        pickle.dump(history.history, f)
        f.close()
    with open(join(SAVINGS_DIR,model_name,'model.json'), 'w') as f:
        f.write(model.to_json())
    model.save_weights(join(SAVINGS_DIR,model_name,'my_model_weights.h5'))

def _model_to_image(model,name,folderpath):
    pydot.find_graphviz = lambda: True
    plot_model(model, to_file=join(folderpath,name))