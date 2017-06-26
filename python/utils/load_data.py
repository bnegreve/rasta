import matplotlib.image as img
import os
import pandas as pd
from PIL import Image
from io import BytesIO
import numpy as np
import requests
import h5py
from scipy.misc import imresize
import zipfile
import sys
from random import shuffle
from shutil import copyfile,move
from os.path import join


DIR_PATH = os.path.dirname(os.path.realpath(__file__))

#----------------------------GENERAL FUNCTIONS---------------------------

def df_as_images_labels(df):

    print('Cleaning data...')
    df = clean_data(df)

    dataset = df['image'].tolist()
    labels = df['label'].tolist()

    print('Encoding data...')
    dataset = encode_dataset(dataset)
    labels,dic = encode_labels(labels)

    return dataset, labels, dic


def encode_dataset(my_list):
    resized_list = resize_images(my_list)
    return np.stack(resized_list, axis=0)


def encode_labels(labels):
    dic = dict(enumerate(set(labels)))

    size = 12  # len(dic)
    inv_dic = {v: k for k, v in dic.items()}
    new_labels = []
    for label in labels:
        new_labels.append(one_hot_vector_encoding(inv_dic.get(label), size))
    return np.stack(new_labels), dic


def one_hot_vector_encoding(label, num_class):
    res = np.zeros(num_class, dtype='int')
    res[label] += 1
    return res

def resize_images(list_images):
    #TODO : better resizing
    new_list = []
    for image in list_images:
            new_list.append(np.transpose(imresize(image,(227,227,3))))
    return new_list

def clean_data(df):
    idx_delete = []
    for i,row in df.iterrows():
        image = row['image']
        if len(image.shape)!=3 or image.shape[2]!=3:
            idx_delete.append(i)
    df = df.drop(df.index[idx_delete])
    return df

#----------------------------PANDORA FUNCTIONS---------------------------

def serialize_pandora():
    df = load_df_pandora()
    images, labels, dic = df_as_images_labels(df)
    del df
    print('Serializing data')
    with h5py.File('../datasets/pandora.h5', 'w') as f:
        f.create_dataset('images', data=images)
        f.create_dataset('labels', data=labels)



def load_df_pandora():
    dataPath = '../../data/pandora/'
    styles = os.listdir(dataPath)
    dataset = []
    labels = []
    artists = []
    image_names = []
    print('Loading data')
    for style in styles:
        print('Loading style ',style,'...')
        style_content = os.listdir(dataPath+style)
        for item in style_content:
            path = dataPath+style+'/'+item
            if os.path.isfile(path):
                try:
                    dataset.append(img.imread(path))
                    artists.append('unknown')
                    labels.append(style)
                    image_names.append(item)
                except OSError:
                    print('Couldn\'t load ' + item)

            if os.path.isdir(path):
                artist_content = os.listdir(path)
                for file in artist_content:
                    try:
                        dataset.append(img.imread(path+'/'+file))
                        artists.append(item)
                        labels.append(style)
                        image_names.append(file)
                    except OSError:
                        print('Couldn\'t load ' + file)

    df = pd.DataFrame()
    df['image_name'] = image_names
    df['image'] = dataset
    df['label'] = labels
    df['artist'] = artists
    return df

def download_pandora():
    dataPath = '../../data/pandora/'
    print('Downloading data...')
    request = requests.get("http://imag.pub.ro/pandora/Download/Pandora_V1.zip",stream=True)
    print('Unziping data...')
    zip_ref = zipfile.ZipFile(BytesIO(request.content))
    zip_ref.extractall(dataPath)
    zip_ref.close()
    return

def load_pandora():
    print('Loading data...')
    file_path = os.path.dirname(os.path.realpath(__file__))
    with h5py.File(file_path+'/../datasets/pandora.h5') as f:
        x = f['images'][:]
        y = f['labels'][:]
    return x,y


#----------------------------WIKIPAINTINGS FUNCTIONS---------------------------


def download_wikipaintings():
    dataPath = '../../data/wikipaintings/'
    data = pd.read_csv('../datasets/wiki_paintings.csv')
    main_styles = ['Art Informel', 'Magic Realism', 'Abstract Art', 'Pop Art', 'Ukiyo-e', 'Mannerism (Late Renaissance)', 'Color Field Painting', 'Minimalism', 'High Renaissance', 'Early Renaissance', 'Cubism', 'Rococo', 'Abstract Expressionism', 'Naïve Art (Primitivism)', 'Northern Renaissance', 'Neoclassicism', 'Baroque', 'Symbolism', 'Art Nouveau (Modern)', 'Surrealism', 'Expressionism', 'Post-Impressionism', 'Romanticism', 'Realism', 'Impressionism']
    data = data[data['style'].isin(main_styles)]
    size = len(data)
    print(size)
    n_downloaded = 0

    for index, row in data.iterrows():
        style = row['style']
        if not os.path.exists(dataPath+style):
            os.makedirs(dataPath+style)
        _download_image(dataPath+style+'/'+row['image_id'],row['image_url'])
        done = int(50 * n_downloaded / size)
        sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
        sys.stdout.flush()

        n_downloaded +=1
    return data
        

def _download_image(file_name,url):
    try:
        img = Image.open(requests.get(url,stream=True).raw).convert('RGB')
        img.save(file_name+'.jpg','JPEG')
    except OSError:
        print('Erreur downloading image ',file_name)

def serialize_wikipaintings():
    df = load_df_wikipaintings()
    images, labels, dic = df_as_images_labels(df)
    del df
    print('Serializing data')
    with h5py.File('../datasets/wikipaintings.h5', 'w') as f:
        f.create_dataset('images', data=images)
        f.create_dataset('labels', data=labels)

def load_df_wikipaintings():
    dataPath = '../../data/wikipaintings/'
    styles = os.listdir(dataPath)
    dataset = []
    labels = []
    image_names = []
    print('Loading data')
    for style in styles:
        print('Loading style ',style,'...')
        style_content = os.listdir(dataPath+style)
        for item in style_content:
            path = dataPath+style+'/'+item
            try:
                dataset.append(img.imread(path))
                labels.append(style)
                image_names.append(item)
            except OSError:
                print('Couldn\'t load ' + item)
    df = pd.DataFrame()
    df['image_name'] = image_names
    df['image'] = dataset
    df['label'] = labels
    return df


def split_test_training(ratio_test=0.1):
    STYLES_PATH = join(DIR_PATH,'../../data/wikipaintings')
    TRAIN_PATH = join(DIR_PATH,'../../data/wikipaintings_train')
    TEST_PATH = join(DIR_PATH,'../../data/wikipaintings_test')
    os.mkdir(TRAIN_PATH)
    os.mkdir(TEST_PATH)
    styles = os.listdir(STYLES_PATH)
    for style in styles:
        os.mkdir(join(TRAIN_PATH,style))
        os.mkdir(join(TEST_PATH,style))
        list_style = os.listdir(join(STYLES_PATH,style))
        n = len(list_style)
        shuffle(list_style)
        split_value = int(n*ratio_test) 
        print('Splitting ',style,' : ')
        list_test = list_style[:split_value]
        list_train = list_style[split_value:]
        print(len(list_train))
        print(len(list_test))
        for f in list_train:
            SRC_PATH = join(STYLES_PATH,style,f)
            DEST_PATH = join(TRAIN_PATH,style,f)
            copyfile(SRC_PATH,DEST_PATH)
        for f in list_test:
            SRC_PATH = join(STYLES_PATH,style,f)
            DEST_PATH = join(TEST_PATH,style,f)
            copyfile(SRC_PATH,DEST_PATH)
def split_val_training(ratio_val=0.1):
    TRAIN_PATH = join(DIR_PATH,'../../data/wikipaintings_train')
    VAL_PATH = join(DIR_PATH,'../../data/wikipaintings_val')
    os.mkdir(VAL_PATH)
    styles = os.listdir(TRAIN_PATH)
    for style in styles:
        os.mkdir(join(VAL_PATH,style))
        list_style = os.listdir(join(TRAIN_PATH,style))
        n = len(list_style)
        shuffle(list_style)
        split_value = int(n*ratio_val) 
        print('Splitting ',style,' : ')
        list_val = list_style[:split_value]
        print(len(list_val))
        for f in list_val:
            SRC_PATH = join(TRAIN_PATH,style,f)
            DEST_PATH = join(VAL_PATH,style,f)
            move(SRC_PATH,DEST_PATH)

if __name__ == "__main__":
    split_val_training()