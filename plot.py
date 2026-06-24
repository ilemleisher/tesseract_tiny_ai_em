import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys
import time
from functools import wraps
from scipy.ndimage import distance_transform_edt
import pandas as pd
import os
import glob

#name = 'data_cont_I4_D20250102_T224816_F0001'

folder = "/home/ilemleisher/data/"
pattern = os.path.join(folder, "data_cont_I4_D*_T*_F*.npz")
filepaths = sorted(glob.glob(pattern))
print(f"Found {len(filepaths)} files")

for fp in filepaths:
    filename = os.path.basename(fp)
    with np.load(fp) as d:
        print("Number of chunks: ", len(d['freqs_list']))
        time_data_list = d['time_data_list']
        waveform_data_list = d['waveform_data_list']
        new_tdata_list = d['new_tdata_list']
        new_data_list = d['new_data_list']
        freqs_list = d['freqs_list']
        asd_list = d['asd_list']
        count = 0
        for i in np.arange(0, len(freqs_list)-4, 5):

        # i = 140

        # fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # axs[0, 0].plot(time_data_list[i], waveform_data_list[i], label="original data")
        # axs[0, 0].set_title("Original waveform")
        # axs[0, 0].set_xlabel("Time [s]")
        # axs[0, 0].set_ylabel("Amplitude")
        # axs[0, 0].legend()

        # axs[1, 0].plot(new_tdata_list[i], new_data_list[i], label="peakless data")
        # axs[1, 0].set_title("Waveform after peak removal/downsampling")
        # axs[1, 0].set_xlabel("Time [s]")
        # axs[1, 0].set_ylabel("Amplitude")
        # axs[1, 0].legend()

        # axs[0, 1].loglog(freqs_list[i], asd_list[i], label="fft data")
        # axs[0, 1].set_title("FFT magnitude")
        # axs[0, 1].set_xlabel("Frequency [Hz]")
        # axs[0, 1].set_ylabel(r"[A/$\sqrt{\mathrm{Hz}}$]")
        # axs[0, 1].legend()

        # axs[1, 1].hist(new_data_list[i], bins=100, label="histogram of data")
        # axs[1, 1].set_title("Noise histogram")
        # axs[1, 1].set_xlabel("Amplitude")
        # axs[1, 1].set_ylabel("Count")
        # axs[1, 1].legend()

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.loglog(freqs_list[i], asd_list[i], label="Chunk "+str(i), alpha=0.5)
            ax.loglog(freqs_list[i+1], asd_list[i+1], label="Chunk "+str(i+1), alpha=0.5)
            ax.loglog(freqs_list[i+2], asd_list[i+2], label="Chunk "+str(i+2), alpha=0.5)
            ax.loglog(freqs_list[i+3], asd_list[i+3], label="Chunk "+str(i+3), alpha=0.5)
            ax.loglog(freqs_list[i+4], asd_list[i+4], label="Chunk "+str(i+4), alpha=0.5)
            ax.set_title("FFT magnitude")
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel(r"[A/$\sqrt{\mathrm{Hz}}$]")
            ax.legend()

            fig.savefig("/home/ilemleisher/plots/plot_"+str(filename[:-4])+"_chunks_"+str(i)+"to_"+str(i+4)+"_chunkfilter.png")
            plt.close(fig)
            # fig.savefig("/home/ilemleisher/plots/plot_"+str(filename[:-4])+"_chunk_"+str(i)+"_2chunks.png")
            # plt.close(fig)