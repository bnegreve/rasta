from keras.models import load_model
import os,datetime
from os.path import join
import sys
import pickle
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras.preprocessing.image import load_img
from utils.utils import imagenet_preprocess_input,wp_preprocess_input,custom_preprocess_input

PATH = os.path.dirname(__file__)
SAVINGS_DIR = join(PATH,'../../savings')

def train_model_from_directory(directory_path,model,model_name ='model',target_size =(256,256) ,batch_size = 64 ,horizontal_flip = False,epochs=30,steps_per_epoch=None,validation_path=None,validation_steps=None,params=None,preprocessing=None,distortions=0.):

    # Naming and creating folder
    now = datetime.datetime.now()
    model_name = model_name+'_' + str(now.year) + '_' + str(now.month) + '_' + str(now.day) + '-' + str(now.hour) +':'+ str(now.minute) +':'+ str(now.second)

    model_name_temp = model_name
    i=0
    while os.path.exists(join(SAVINGS_DIR,model_name_temp)):
        model_name_temp=model_name+'('+str(i)+')'
        i+=1
    MODEL_DIR = join(SAVINGS_DIR,model_name_temp)
    os.makedirs(MODEL_DIR)


    # Calculate the number of steps, in order to use all the set for one epoch

    n_files = count_files(directory_path)
    n_val_files = count_files(validation_path)
    if steps_per_epoch==None:
        steps_per_epoch = n_files//batch_size
    if validation_steps==None:
        validation_steps = n_val_files//batch_size

    _presaving(model,MODEL_DIR,params)

    preprocessing_fc =None
    if preprocessing=='imagenet':
        preprocessing_fc = imagenet_preprocess_input
    elif preprocessing=='wp':
        preprocessing_fc = wp_preprocess_input
    elif preprocessing =='custom':
        preprocessing_fc = custom_preprocess_input


    # Training
    train_datagen = ImageDataGenerator(horizontal_flip = horizontal_flip,preprocessing_function=preprocessing_fc,rotation_range=90*distortions,width_shift_range=distortions,height_shift_range=distortions,zoom_range=distortions)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocessing_fc)
    train_generator = train_datagen.flow_from_directory(directory_path, target_size = target_size, batch_size = batch_size, class_mode='categorical')
    tbCallBack = TensorBoard(log_dir=MODEL_DIR, histogram_freq=0, write_graph=True, write_images=True)
    if validation_path!=None:
        checkpoint = ModelCheckpoint(join(MODEL_DIR,'model.h5'), monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        validation_generator = test_datagen.flow_from_directory(validation_path,target_size=target_size,batch_size=batch_size,class_mode='categorical')
        history  = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs = epochs,callbacks = [tbCallBack,checkpoint],validation_data=validation_generator,validation_steps=validation_steps)
    else:
        history  = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs = epochs,callbacks = [tbCallBack])

    _postsaving(model,history,MODEL_DIR)

    return model

def continue_training(model_path,directory_path,saving=True,target_size =(256,256) ,batch_size = 64 ,horizontal_flip = False,epochs=30,steps_per_epoch=1000,validation_path=None,validation_steps=110):
    model = load_model(join(model_path,'best_model.h5'))    

    train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip = False)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(directory_path, target_size = target_size, batch_size = batch_size, class_mode='categorical')
    tbCallBack = TensorBoard(log_dir=model_path, histogram_freq=0, write_graph=True, write_images=True)       
    if validation_path!=None:
        validation_generator = test_datagen.flow_from_directory(validation_path,target_size=target_size,batch_size=batch_size,class_mode='categorical')
        history  = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs = epochs,callbacks = [tbCallBack],validation_data=validation_generator,validation_steps=validation_steps)
    else:
        history  = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs = epochs,callbacks = [tbCallBack])

    if saving:
        _postsaving(model,history,model_path)
    return model

def _presaving(model,model_dir,params):
    with open(join(model_dir,'summary.txt'), 'w') as f:
        orig_stdout = sys.stdout
        sys.stdout = f
        print(model.summary())
        sys.stdout = orig_stdout
        f.close()
    with open(join(model_dir,'model.json'), 'w') as f:
        f.write(model.to_json())
    with open(join(model_dir,'params.txt'), 'w') as f:
        f.write(str(params))

def _postsaving(model,history,model_dir):
    model.save_weights(join(model_dir,'my_model_weights.h5'))
    model.save(join(model_dir,'final_model.h5'))
    with open(join(model_dir,'history.pck'), 'wb') as f:
        pickle.dump(history.history, f)
        f.close()


def count_files(folder):
    s = 0
    for t in list(os.walk(folder)):
        s += len(t[2])
    return s


if __name__=='__main__':
    DATA_PATH = join(PATH,'../../data/wikipaintings/wikipaintings_train')
    s = np.array([0.,0.,0.])
    t=0
    for folder in os.listdir(DATA_PATH):
        print("processing ",folder)
        for file in os.listdir(join(DATA_PATH,folder)):
            x = load_img(join(DATA_PATH,folder,file),target_size=(224,224))
            s += np.mean(x,axis=(0,1))
            t +=1
    mean = s/t
    print(mean)