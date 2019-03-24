import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


sns.set()

import utilities

################################################################################
# Neural network structures (Relu activation function)
################################################################################
class Net(nn.Module):
    '''
    Fully connected neural network with 3 hidden layers
    '''
    def __init__(self, nb_features = 27):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(nb_features, 200)
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200,200)
        self.fc4 = nn.Linear(200, 7)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class convNet(nn.Module):
    '''
    convolutional neural network with three hidden layer, two convolutional and last one linear
    '''
    def __init__(self):
        super(convNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 20, kernel_size=4)
        self.fc1 = nn.Linear(20*8*8, 21)
        self.fc2 = nn.Linear(21, 7)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=3))
        x = F.relu(self.fc1(x.view(-1, 20*8*8)))
        x = self.fc2(x)
        return x

################################################################################
# evaluating the accuracy of the network prediction
################################################################################
def compute_accuracy(x,y):
    '''
    compute the accuracy of the prediction x = [p1,p2,..,p7]
    different probability for each classes and the true target vector y
    '''
    error = 0
    for elt,res in zip(x,y):
        if torch.argmax(elt).item() != res.item():
            error+=1
    return 100-error*100/y.size(0)

def compute_accuracy_per_class(x,y):
    error = np.repeat(0,7)
    sizes = np.repeat(1,7)

    for elt,res in zip(x,y):
        sizes[res.item()]+=1
        if torch.argmax(elt).item() != res.item():
            error[res.item()]+=1

    return (1-error/sizes)*100


################################################################################
# training the networks
################################################################################
def train_model(model, train_input, train_target, test_input, test_target, optimizer, criterion, batch_size, epochs, rate_print, verbose = True, chrono = False, plots = False, save = False, title = "") :
    '''
    train a neural network with the usual parameters
    '''
    Loss = []
    accuracy = []
    accuracy_train = []
    accuracy_test_classes = []

    if chrono:
        t_start = time.time()

    for e in range(epochs):
        #use batches to improve speed
        for b in range(0,train_input.size(0),batch_size):
            output = model(train_input[b:b+batch_size])
            loss = criterion(output,train_target[b:b+batch_size])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        if e % rate_print == 0:
            Loss.append(criterion(model(train_input),train_target))
            accuracy.append(compute_accuracy(model(test_input),test_target))
            accuracy_train.append(compute_accuracy(model(train_input),train_target))
            accuracy_test_classes.append(compute_accuracy_per_class(model(test_input),test_target))
            if verbose:
                print("epoch {:3}, loss training: {:4.3f}, train accuracy {:4.3f},  test accuracy {:4.3f}".format(e,Loss[-1],accuracy_train[-1],accuracy[-1]))

    #plots

    if chrono:
        t_end = time.time()
        time_total = t_end-t_start
        print("\n")
        print("Time to train: {:4.4f} seconds".format(time_total))
        print("\n")

    if plots:
        #plot the loss and general accuracy against time
        plt.subplot(2,1,1)
        x = rate_print*np.arange(len(Loss))
        plt.plot(x,Loss)
        plt.title("training loss vs epochs")
        plt.subplot(2,1,2)
        plt.plot(x,accuracy, label = "test")
        plt.plot(x,accuracy_train, label = "train")
        plt.title("accuracy vs epochs")
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig('results/plots/neural_net_'+ title +'.png',dpi = 500)
        plt.show()

        #plot the individual accuracy for each class
        test = np.asanyarray(accuracy_test_classes)
        emotions_english = ["anxiety",'disgust','happy','boredom','anger','sadness','neutral']
        x = rate_print*np.arange(len(test[:,0]))
        for i in range(7):
            plt.plot(x,test[:,i],label = "accuracy for {}".format(emotions_english[i]))
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig('results/plots/neural_net_'+ title +'_accuracy_per_class.png',dpi = 500)
        plt.show()

def cross_validation_nn(net, X, y, k_fold = 5, epochs = 60, seed = 0, buffer_path = "buffer", confusion_matrix = False):
    '''
    perform cross validation for a neural network
    buffer_path: path to a folder where to store data for a short amount of time in order to perform cross validation with the same weight initialization for each fold

    if confusion_matrix is True, compute the confusion matrix on average on the folds and return a dataframe with the values, plus the vectors y_true, y_pred concatenated trough the folds
    to plot ut using sklearn function
    '''

    #build k indices for k-fold
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = np.array([indices[k * interval: (k + 1) * interval] for k in range(k_fold)])

    scores = []
    scores_classe = []

    if confusion_matrix:
        #prepare empty recipient to compute the confusion matrix
        dic = {0:[],1:[],2:[],3:[],4:[],5:[],6:[]}
        dic_length = {0:0,1:0,2:0,3:0,4:0,5:0,6:0}
        matrix = np.zeros((7,7))

    #save the state of the model so that we can "reinitialize it"
    torch.save(net.state_dict(), buffer_path +"/cross_val.pt")


    for k in range(k_fold):
        print("{}th fold training".format(k+1))

        #reinitialize the weight of the model
        net.load_state_dict(torch.load(buffer_path +"/cross_val.pt"))
        net.eval()



        #separate the data
        te_indice = k_indices[k]
        tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
        tr_indice = tr_indice.reshape(-1)
        y_te = y[te_indice]
        y_tr = y[tr_indice]
        x_te = X[te_indice]
        x_tr = X[tr_indice]

        #parameters to train the neural network
        batch_size = int(y_tr.size(0)/10)
        optimizer = torch.optim.Adam(net.parameters(),lr = 1e-4)
        criterion = nn.CrossEntropyLoss()

        #actual training
        train_model(net, x_tr, y_tr, x_te, y_te, optimizer, criterion, batch_size, epochs, rate_print = 10, verbose = False, plots = True)

        #evaluation:
        output = net(x_te)
        scores.append(compute_accuracy(output,y_te))
        scores_classe.append(compute_accuracy_per_class(output,y_te))

        if confusion_matrix:
            #could do it without for loop, but I think it is clearer
            for i, goal in enumerate(y_te):
                dic[goal.item()].append(torch.argmax(output[i]).item())
                dic_length[goal.item()] += 1


    #average
    score_mean = np.mean(np.array(scores))
    scores_classe_mean = np.mean(np.array(scores_classe),axis = 0)

    if confusion_matrix:
        #actually compute the confusion matrix
        emotion = ['anxiety', 'disgust', 'happy', 'boredom', 'anger', 'sadness', 'neutral']
        for i in range(7):
            for j in range(7):
                matrix[i,j]= np.round(100*len(np.where(np.asarray(dic[i]) == j)[0])/dic_length[i],decimals = 2)
        confusion_matrix = pd.DataFrame(matrix,index = emotion,columns=emotion)
        confusion_matrix.to_csv("results/confusion_matrix.csv")

        return score_mean,scores_classe_mean, confusion_matrix

    return score_mean,scores_classe_mean
