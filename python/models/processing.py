from keras.utils import plot_model
import os,datetime
from os.path import join
import sys
import pickle
import pydot
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard,EarlyStopping


PATH = os.path.dirname(__file__)
SAVINGS_DIR = join(PATH,'../../savings')

def train_model_from_directory(directory_path,model,model_name ='model',saving=True,target_size =(256,256) ,batch_size = 64 ,horizontal_flip = False,epochs=30,steps_per_epoch=1000,validation_path=None,validation_steps=110):
    now = datetime.datetime.now()
    model_name = model_name+'_' + str(now.year) + '_' + str(now.month) + '_' + str(now.day) + '-' + str(now.hour) +':'+ str(now.minute) +':'+ str(now.second)
    MODEL_DIR = join(SAVINGS_DIR, model_name)
    os.makedirs(MODEL_DIR)
    
    if saving:
        params={"batch size":batch_size ,"horizontal flip":horizontal_flip,"epochs":epochs,"steps per epoch":steps_per_epoch,"validation steps":validation_steps}
        _presaving(model,MODEL_DIR,params)

    train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip = horizontal_flip)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(directory_path, target_size = target_size, batch_size = batch_size, class_mode='categorical')
    tbCallBack = TensorBoard(log_dir=MODEL_DIR, histogram_freq=0, write_graph=True, write_images=True)       
    if validation_path!=None:
        #earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
        validation_generator = test_datagen.flow_from_directory(validation_path,target_size=target_size,batch_size=batch_size,class_mode='categorical')
        history  = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs = epochs,callbacks = [tbCallBack],validation_data=validation_generator,validation_steps=validation_steps)
    else:
        history  = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs = epochs,callbacks = [tbCallBack])

    if saving:
        _postsaving(model,history,MODEL_DIR)
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
    model.save(join(model_dir,'model.h5'))
    with open(join(model_dir,'history.pck'), 'wb') as f:
        pickle.dump(history.history, f)
        f.close()
