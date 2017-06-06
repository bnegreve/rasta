import matplotlib.pyplot as plt
import pickle

def plot_history(history_path):
    with open(history_path,'rb') as f:
        history = pickle.load(f)
        print(history.get('val_acc'))

    plt.plot(history.get('acc'),label='Test accuracy')
    plt.plot(history.get('val_acc'),label='Validation accuracy')
    plt.legend(loc=2, borderaxespad=0.)
    plt.show()



if __name__ == '__main__':
    plot_history('/home/alecoutre/STAGE/rasta-project/scripts/savings/alexnet_2017_6_6:10299/history.pck')
