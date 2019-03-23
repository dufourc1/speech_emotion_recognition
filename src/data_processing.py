'''
File needed to load the data, compute the features and create a big panda dataframe so that all information are stored in it once

The architecture of the folders should be the following:
     .
    ├── data                              # Data files
        └── wav                           # folder with the audio file
    ├── src                               # Source files
        └── data_processing.py
    ├── results
    └── models
'''

from __future__ import print_function
import os


#libraries
import numpy as np
import pandas as pd
import python_speech_features
import librosa
import librosa.display
import torch

#src file
import utilities

#constant from the data
sampling_rate = 1600


def load_data(path = None):
    '''
    load the data from the audio files

    input:
        path to the folder containing the audio files (.wav extension)

    output:
        a panda dataframe with the following columns: ['id','actor','text','emotion'] with emotion being coded from 0 to seven
                                                        with the following mapping: {'A': 0, 'E': 1, 'F':2, 'L':3, 'W':4, 'T':5, 'N':6}
        an array to map the emotion in english to the number encodage above
    '''

    if path == None:
        path = '../data/wav'

    files = os.listdir(path)

    sample = []
    for ID in files:
        sample.append([path+'/'+ID,ID[0:2],ID[2:5],ID[5]])
    samples = pd.DataFrame(sample,columns=['id','actor','text','emotion'])

    #Changing the encodage of the emotion and keeping track of it in a dictionnary
    emotions = ["A",'E','F','L','W','T','N']
    emotions_english = ["anxiety",'disgust','happy','boredom','anger','sadness','neutral']
    mapping = {'A': 0, 'E': 1, 'F':2, 'L':3, 'W':4, 'T':5, 'N':6}
    samples = samples.replace({'emotion':mapping})

    return samples,emotions_english

def import_wave_wrapper(df , modify = True):
    '''
    Wrapper for the function librosa.load(), so it's callable on a data-set

    input:
        dataframe with a column ["id"] of path to the wave files

    output:
        if modify: modify the dataframe by adding to it the sound data extracted from the file, and its length
        else: return the two above objects
    '''

    #create empty receptacle
    Y = []
    SR = []
    l = []

    # faster thant df.apply(librosa.load)  for some reason
    for i in range(len(df)):
        y,sr = librosa.load(df["id"][i],sr=None)
        Y.append(y)
        l.append(len(y))

    if modify:
        df["sound"] = Y
        df["length"] = l
    else:
        return Y,l

def silence_size(y,size = 2, threshold = 0.01):
    '''
    input:
        np.array representing a time series sound

    output:
        return the ratio:length_silence/length_speaking

    length silence is defined arbitrarly by the formula max(abs(y[i-size:i+size])) < threshold

    could use a bit of tuning and amelioration, but due to the time constraint ...
    '''
    coord = []
    size = 5
    for i in range(size,len(y)-size):
        if np.max(abs(y[i-size:i+size])) < threshold:
            coord.append(i)

    return len(coord)

def ratio(df,threshold = 0.02, modify = True):
    '''
    input:
        dataframe with a sound data loaded as a numpy array in column: ["sound"]

    modify the dataframe by adding a ratio between length of spoken vs silence in the audio file computed by the function silence_size()
    '''

    ratios = []

    for i in range(len(df)):
        sound = df["sound"][i]
        s = silence_size(sound)
        l = len(sound)
        ratio = s/(l-s)
        ratios.append(ratio)

    if modify:
        df["ratio"] = ratios
    else:
        return ratios

def extract_feature(df,f, name, modify = True):
    '''
    Wrapper to extract meaningful summary of the data

    input:
        df with a column ["sound"] containing the time series representation of the sound
        f is the function to apply to the data (from the librosa.features library)
        name is a string so that if modify == True, we can save the statistics with an appropriate name

    output:
        if modify: add mean, sd, median, max and min of f(sound) to df
        else: return those quantities

    '''

    mean = []
    sd = []
    median = []
    maxi = []
    mini = []

    for i in range(len(df)):
        #extract the data
        inter = f(df["sound"][i],1600)

        #compute the actual statistics
        mean.append(np.mean(inter))
        sd.append(np.std(inter))
        median.append(np.median(inter))
        maxi.append(np.max(inter))
        mini.append(np.min(inter))

    if modify:
        df[name +"_mean"] = mean
        df[name +"_sd"] = sd
        df[name +"_median"] = median
        df[name +"_max"] = maxi
        df[name + "_min"] = mini
    else: return mean,sd,median,maxi,mini

def put_sound_to_mean_lenght(df,final_length = None, verbose = False):
    '''
    try to eliminate the difference in the input size by cutting the samples which are too large and padding with zeros the
    ones that are too small

    input:
        df: dataframe with the sound data in the column ["sound"]
        final_length: final length of the audio data. If none, df should have a column ['length'] and the cut would be the mean of the not too extremes sound
        verbose: if true print the number of troncatured and padded sound samples

    output:
        numpy nd array containing the modified sound samples
    '''

    final_length = int(np.mean(df["length"][df["length"]<99447]))
    sound_modified = []

    raccourci = 0
    rallonge = 0

    for a in df["sound"]:

        if len(a)> final_length:
            #cut the signal if it's too long
            inter = a[:final_length]
            raccourci +=1

        elif len(a) < final_length:
            #pad the signals with 0
            inter = np.zeros(final_length)
            inter[:len(a)] = a
            rallonge +=1

        else:
            inter = a

        sound_modified.append(inter)

    if verbose:
        print("troncatured: {}, padded: {}, total: {}".format(raccourci,rallonge,len(samples["sound"])))

    return sound_modified

def graphic_representation_audio(df):
    '''
    take the sound samples and return an image like representation based on decibel input and mfcc coefficients

    input:
        df: dataframe with the sound data in the column ["sound"] and length in ["length"]
    output:
        torch tensor of size torch.Size([535, 2, 86, 86])

    '''
    #make all the sound samples the same size
    source = put_sound_to_mean_lenght(df)

    #size are choosen such that both of the following quantities have the same size
    #were choosen from visual inspection, heuristic approach
    size = 86

    #empty receptacle
    spectrogram = []
    db_rep = []
    reduced_data = torch.zeros(len(source),size,size)
    data_complete = torch.zeros(len(source),2,size,size)

    for y in source:
        #compute the quantities from the data

        spectrogram.append(librosa.feature.melspectrogram(y,1600)[0:size,:])
        db_rep.append(librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref = np.max))

    #tranform to be able to use torch library
    db = torch.tensor(db_rep).type(torch.FloatTensor)
    data_spec = torch.tensor(spectrogram).type(torch.FloatTensor)

    #use PCA to compress the data so both quantities have the same size
    for i,elt in enumerate(db):
        elt_inter = torch.transpose(elt,0,1)
        reduced_data[i] = torch.transpose(utilities.PCA(elt_inter,size),0,1)

    #combine the two
    #should be made better, but no time

    for i,elt1 in enumerate(data_spec):
        elt2 = reduced_data[i]
        inter = torch.zeros(2,86,86)
        inter[0] = elt1
        inter[1] = elt2
        data_complete[i] = inter


    return data_complete



if __name__ == '__main__':
    print("data processing have function to be imported")
