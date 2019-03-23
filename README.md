
 
# Visium user-case interview

Use case for a summer intership at Visium, Speech emotion recognition based on the berlin database Emodb 

The purpose of this project is to find a model to properly classify emotion of the speaker given sound samples

## Libraries used
We used the following libraries for this project, with Python 3.6.5


 Computational:

    numpy (as np)
    sklearn (scikit-learn)
    torch
    librosa
    python_speech_features
    
Graphical:

    seaborn (as sns) (version 0.9.0)
    matplotlib (as plt)


## Prerequisites


To install some of the libraries mentionned before , please use the following command (linux):

    pip3 install librosa
    pip install python_speech_features

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
        ├── 
        └── models
    └── README.md


## Implementations details


### Report.ipynb

Import functions from `neural_networks.py`, `utilities.py`,`data_processing.py`

Jupyter notebook to describe our approach to solve this classification issue. Some parts may be slow to run (features extraction),
due to the short deadline, not everything is optimized. This is why there are files `.csv` saved with all relevant transformations 
already computed

### data_processing.py

Take in charge all of the data pipeline, the features extraction and so on

## References:
  **Emotion recognition from the human voice**. Parlak, Cevahir & Diri, Banu. (2013).  21st Signal Processing and Communications Applications Conference, SIU 2013. 1-4. 0.1109/SIU.2013.6531196
  
  
  **Speech Emotion Recognition: Methods and Cases Study**. Kerkeni, Leila & Serrestou, Youssef & Mbarki, Mohamed & Raoof, Kosai & Mahjoub, Mohamed. (2018). 175-182. 10.5220/0006611601750182
  
  
  **Emotion Recognition from Speech using Discriminative Features**. Chandrasekar, Purnima & Chapaneri, Santosh & Jayaswal, Deepak. (2014). International Journal of Computer Applications. 101. 31-36. 10.5120/17775-8913
  
  
  [Berlin emotional speech database](http://emodb.bilderbar.info/index-1024.html)
  
  
  Librosa: Audio and music signal analysis in python.McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto
  
  
  Hastie, T.; Tibshirani, R. & Friedman, J. (2001), The Elements of Statistical Learning , Springer New York Inc. , New York, NY, USA .


## Author

* *Charles Dufour*
