import matplotlib.image as img
import os
import pandas as pd
from PIL import Image
import requests
import numpy as np
import pickle

#----------------------------COMMUN FUNCTIONS---------------------------

def df_as_dataset_labels(df):

    df = clean_data(df)

    dataset = df['image'].tolist()
    labels = df['label'].tolist()


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
    new_list = []
    for image in list_images:
            new_list.append(image[:32,:32,:])
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



def load_pandora():
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




#----------------------------WIKIPAINTINGS FUNCTIONS---------------------------

def load_wiki_paintings():
    dataPath = '../data/wiki_paintings'
    styles = os.listdir(dataPath)

    dataset = []
    labels = []
    print('Loading data')
    for style in styles:
        print('Loading style ',style,'...')
        size = load_style(dataPath+style,dataset)  
        print(size," images found") 
        labels.extend([style for i in range(size)])
    return dataset,labels



def download_wikipaintings():
    dataPath = '../data/wiki_paintings/'
    data = pd.read_csv(dataPath+'wiki_paintings.csv')
    for index, row in data.iterrows():
        style = row['style']
        if not os.path.exists(dataPath+style):
            os.makedirs(dataPath+style)
        download_image(dataPath+style+'/'+row['image_id'],row['image_url'])

        

def download_image(file_name,url):
    try:
        img = Image.open(requests.get(url,stream=True).raw).convert('RGB')
        img.save(file_name+'jpeg','JPEG')
    except OSError:
        print('Erreur downloading image ',file_name)








if __name__ == "__main__":
    df = load_pandora()
    dataset,labels,dic = df_as_dataset_labels(df)

    with open('../dataset.pick','wb') as f:
        pickle.dump(dataset,f)
    with open('../labels.pick', 'wb') as f:
        pickle.dump(labels, f)
    with open('../dic.pick', 'wb') as f:
        pickle.dump(dic, f)
    #dataset,labels = get_pandora_dataset_labels()
    #n = len(labels)
    #randomize = np.arange(len(dataset))
    #np.random.shuffle(randomize)
    #dataset = dataset[randomize,:,:,:]
    #labels = labels[randomize]

    #dataset = dataset[:n//20]
    #labels = labels[:n//20]
    #pickle.dump(dataset,open('dataset.pck','wb'))
    #pickle.dump(labels,open('labels.pck','wb'))