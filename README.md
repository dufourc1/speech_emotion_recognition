

# Speech emotion recognition

Use case for a summer internship at Visium, Speech emotion recognition based on the Berlin database Emodb

The purpose of this project is to find a model to properly classify the emotion of the speaker given sound samples, so we can do it later live.

## Libraries used
We used the following libraries for this project, with Python 3.6.5


 Computational:

    numpy (as np)
    sklearn (scikit-learn)
    torch
    librosa
    python_speech_features
    pandas
    pandas_profiling

Graphical:

    seaborn (as sns) (version 0.9.0)
    matplotlib (as plt)


## Prerequisites


To install some of the libraries mentioned before, please use the following command (Linux):

    pip install librosa
    pip install python_speech_features
    pip install pandas-profiling

The folder structure has to be the following:

    .
    ├── data                              # Data files, in .csv
        └── wav
    ├── src                               # Source files
        ├── neural_networks.py
        ├── data_processing.py
        └── utilities.py
    ├── results
        ├── plots
        └── models
    └── README.md


## Implementations details


#### Report.ipynb

Import functions from `neural_networks.py`, `utilities.py`,`data_processing.py`

Jupyter notebook to describe our approach to solving this classification issue. Some parts may be slow to run (features extraction),
due to the short deadline, not everything is optimized. This is why there are files `.csv` saved with all relevant transformations
already computed

#### data_processing.py

Take in charge all of the data pipeline, the features extraction and so on

#### results

Contains files saved for easy later used:

      `features_exploration.html` profiling of the features

 #### data
  The folder containing the audio files is `wav`. The other folders are included for the sake of completeness and contain preprocessed data. We did not use them since we wanted computable features from the audio data only, not preprocessed by experts. Could be used to improve the accuracy of our model.


## Model and accuracy

Our model is a simple neural network, fully connected, with four hidden layers and Relu activation function. Its prediction is globally of approximately 69%, and in each class individually range from 40% to 90%


<p align="center">

<img src="https://github.com/dufourc1/visium_use_case/blob/master/results/plots/neural_net_fully%20connected.png" height="300" width="500">

<img src="https://github.com/dufourc1/visium_use_case/blob/master/results/plots/neural_net_fully%20connected_accuracy_per_class.png" height="250" width="500">

</p>


## References:
  **Emotion recognition from the human voice**. Parlak, Cevahir & Diri, Banu. (2013).  21st Signal Processing and Communications Applications Conference, SIU 2013. 1-4. 0.1109/SIU.2013.6531196


  **Speech Emotion Recognition: Methods and Cases Study**. Kerkeni, Leila & Serrestou, Youssef & Mbarki, Mohamed & Raoof, Kosai & Mahjoub, Mohamed. (2018). 175-182. 10.5220/0006611601750182


  **Emotion Recognition from Speech using Discriminative Features**. Chandrasekar, Purnima & Chapaneri, Santosh & Jayaswal, Deepak. (2014). International Journal of Computer Applications. 101. 31-36. 10.5120/17775-8913


  [Berlin emotional speech database](http://emodb.bilderbar.info/index-1024.html)


  Librosa: Audio and music signal analysis in python.McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto


  Hastie, T.; Tibshirani, R. & Friedman, J. (2001), The Elements of Statistical Learning, Springer New York Inc., New York, NY, USA.


## Author

* *Charles Dufour*
