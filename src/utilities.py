'''
most of the utilities function are coded for torch tensor
'''

import numpy as np
import torch

def PCA(data, k=2):
    # preprocess the data
    X = data
    X_mean = torch.mean(X,0)
    X = X - X_mean.expand_as(X)
     # svd
    U,S,V = torch.svd(torch.t(X))
    return torch.mm(X,U[:,:k])

def one_hot(y):
    '''
    return the one hot version of the vector of classes y
    '''
    y_vect = np.zeros((len(y), 7))
    for i in range(len(y)):
        y_vect[i, int(y[i].item())] = 1

    return(torch.tensor(y_vect).type(torch.FloatTensor))


def split_train_test(features,target, percentage_train = 0.8, normalization = True):
    '''
    split randomly a dataset into test and train

    output: train_input, train_target, test_input, test_target
    '''
    train_size = int(0.8 * len(features))

    labels_total = list(range(len(features)))
    np.random.shuffle(labels_total)
    label_training = labels_total[0:int(0.8*len(labels_total))]
    label_test = labels_total[train_size:]

    train_input = features[label_training]
    test_input = features[label_test]

    train_target = target[label_training]
    test_target = target[label_test]



    #normalization
    if normalization:
        mu,std = train_input.mean(0), train_input.std(0)
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)

    return train_input, train_target, test_input, test_target
