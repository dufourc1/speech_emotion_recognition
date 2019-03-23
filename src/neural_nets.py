import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import time
import matplotlib.pyplot as plt
import seaborn as sns


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
def train_model(model, train_input, train_target, test_input, test_target, optimizer, criterion, batch_size, epochs, rate_print, chrono = False, plots = False, save = False, title = "") :
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
