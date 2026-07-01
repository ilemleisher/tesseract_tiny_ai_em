# Tiny AI Environmental Monitoring System for TESSERACT data stream

This repository contains all codes being developed for a real-time anomalous noise detection system for the data stream from the HeRALD detector at Lawrence Berkeley National Laboratory, as part of the TESSERACT experiment.

## Workflow

Waveform data from both ADCs in the detector are uploaded onto the TESSERACT servers in real time in ~1 second increments. 
The data outputs are contained in HDF5 files. 

### ```preprocessing```

- ```preprocess.py```
1- Loads all consecutive HDF5 files.
    - In the future modify for real-time
2- Constructs waveform data from defined channel.
3- Downsamples.
4- Chunks data.
5- Filters out signal.
6- FFT.
7- Concatenates chunks into one file.
8- Saves as npz to server.

### ```modules```

Modules are toggleable anomaly detection algorithms. Each module must contain a ```flag``` function that reads in frequency data and ASD data, and outputs an array of binary anomaly labeling, corresponding chunk indices, and metadata.

- ```pca.py```:
1- Runs PCA.
2- Evaluates PCA reconstruction error compared to threshold.

- ```ema.py```:
1- Calculates drift score based on linear baseline residuals.
2- Identifies outlier drift scores.

### ```main.py```

1- Loads modules.
2- Loads all consecutive npz data files based on target dataset.
3- Runs modules and creates complete flag array 

### Dev Scripts

- ```preprocess.py```: reads the raw HDF5 files, downsamples them, divides each data file into uniform chunks, filters out any chunks that contain signal events (represented by peaks in the waveform data), performs FFT on the chunks, then concatenates the chunks so that they represent time-continuous data (across several data files).

- ```anomaly.py```: loads ```.npz``` files saved from ```preprocess.py``` and uses PCA to calculate a reconstruction error and identify data chunks that contain anomalous peaks. Also calculates EMA baseline residual to identify data chunks that contian anomalous baseline drift. Plots the PCA reconstruction residual against the ASD for each anomalous chunk, and plots 5 data chunks centered on each chunk labeled with a baseline anomaly.

- ```model.py```: TensorFlow 2D Convolutional Neural Network (CNN). Has three 2D convolution layers, a global pooling layer, and a linear output layer.

- ```train.py```: loads data files saved from ```preprocess.py``` and corresponding label files from ```pca.py``` to create data patches for the CNN. Trains the model using binary cross entropy loss, saves the best model, metrics, and loss/accuracy plots.

- ```utils.py```: contains helper functions for loading files and calculating linear baseline.

- ```infer.py```: work in progress, eventually to be used with the trained CNN.

- ```noise.py```: works the same as ```preprocess.py``` for loading in data and creating spectral chunks, however with the addition of artifical noise modes, with the goal being to implement specific noise to have a sufficiently large labeled dataset for supervised training of the CNN.

## Required Packages

- ```h5py```
- ```numpy```
- ```TensorFlow```
- ```matplotlib```
- ```sklearn```
