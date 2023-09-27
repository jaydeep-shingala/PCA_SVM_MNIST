import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # load the data
    train_df = pd.read_csv('data/mnist_train.csv')
    test_df = pd.read_csv('data/mnist_test.csv')

    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values

    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    return X_train, X_test, y_train, y_test


def normalize(X_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
    
    colmin = 0
    rowmin = 255
    X_train = [2*(x-colmin)/(rowmin-colmin)-1 for x in X_train]
    X_test = [2*(x-colmin)/(rowmin-colmin)-1 for x in X_test]
    return X_train, X_test
    # raise NotImplementedError


def plot_metrics(metrics) -> None:
    # plot and save the results
    numofk = len(metrics)
    ks = []
    accuracy = []
    precision= []
    recall = []
    f1score = []
    for i in range(0, numofk):
        ks.append(metrics[i][0])
        accuracy.append(metrics[i][1])
        precision.append(metrics[i][2])
        recall.append(metrics[i][3])
        f1score.append(metrics[i][4])

       
    plt.plot(ks, accuracy)
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.title('different k values vs. accuracy ')
    plt.savefig("k_accuracy.jpg")
    plt.clf()
    
    plt.plot(ks, precision)
    plt.xlabel('k')
    plt.ylabel('precision')
    plt.title('different k values vs. precision ')
    plt.savefig("k_precision.jpg")
    plt.clf()
    
    plt.plot(ks, recall)
    plt.xlabel('k')
    plt.ylabel('recall')
    plt.title('different k values vs. recall ')
    plt.savefig("k_recall.jpg")
    plt.clf()
    
    plt.plot(ks, f1score)
    plt.xlabel('k')
    plt.ylabel('f1-score')
    plt.title('different k values vs. f1-score ')
    plt.savefig("k_f1-score.jpg")
    plt.clf()
    # raise NotImplementedError