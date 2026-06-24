# Tiny AI Environmental Monitoring System for TESSERACT data stream

This repository contains all codes being developed for a real-time anomalous noise detection system for the data stream from the HeRALD detector at Lawrence Berkeley National Laboratory, as part of the TESSERACT experiment.

## Workflow

Currently, the system uses a 2D Convolutional Neural Network (CNN) to detect anomalies in power spectra from the detector. 
Waveform data from both ADCs in the detector are uploaded onto the TESSERACT servers in real time in ~1 second increments. 
The data outputs are contained in HDF5 files. 
A preprocessing script loads an HDF5 file, downsamples it to a more managable sampling frequency, divides the data into smaller chunks, and throws out any chunks that contain signal from the experiment. 
The script performs a FFT on the chunks, and then stacks the spectral data into a continuous dataset for the CNN.
