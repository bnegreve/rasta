import matplotlib.pyplot as plt
import pickle
from utils.load_data import *

def plot_history(history_path):
    with open(history_path,'rb') as f:
        history = pickle.load(f)
        print(history.get('val_acc'))

    plt.plot(history.get('acc'),label='Test accuracy')
    plt.plot(history.get('val_acc'),label='Validation accuracy')
    plt.legend(loc=2, borderaxespad=0.)
    plt.show()

def bar_dataset(df):
    styles = df['style'].values()
    D = {}
    for style in styles:
        D.update({style:len(df[df['style']==style])})
    plt.bar(range(len(D)), D.values(), align='center')
    plt.xticks(range(len(D)), D.keys(), rotation=80)
    plt.show()


if __name__ == '__main__':
    df = load_df_pandora()
    bar_dataset(df)