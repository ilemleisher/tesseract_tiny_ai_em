import sys, os, glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

# path = "/data/lbl/run21/raw/continuous_I4_D20250102_T224744/"
# pattern = os.path.join(path, "cont_I4_D*_T*_F*.hdf5")
# filepaths = sorted(glob.glob(pattern))
# print(len(filepaths), "files found")

# for filepath in filepaths:
#     filename = os.path.basename(filepath)   # just file name
#     print("Reading:", filename)

#     with h5py.File(filepath, "r") as data:
#         print("Keys in HDF5 file:", list(data['adc1'].keys()))
#         waveforms = np.array(data["adc1"]["event_1"])
#     print(len(waveforms[0]), "waveforms found")

# path = "/home/ilemleisher/data/"
# pattern = os.path.join(path, "data_cont_I4_D*_T*_F*.npz")
# files = sorted(glob.glob(pattern))
# print(f"Found {len(files)} files")
# for fp in files:
#     with np.load(fp) as d:
#         freqs_list = d['freqs_list']
#         asd_list = d['asd_list']
#         baseline_list = d['baseline_list']
#         residual_list = d['residual_list']

# t, n_bins, k = len(freqs_list), len(freqs_list[0]), 6
# print(t, n_bins, k)

import h5py, glob, os
import numpy as np
import matplotlib.pyplot as plt

# Variables
n_chunks = 2
post_downsample_length = 12000
sampling_rate = 1.25e6
channel_number = 0

import numpy as np

def preprocess(tdata, data, target_len=12000, sigma_thresh=4, radius=1000):
    """
    Remove high points and also remove points within +/- radius of each removed point.
    """
    raw_data = np.asarray(data).copy()
    tdata = np.asarray(tdata)

    med = np.median(raw_data)
    height = med + np.std(raw_data) * sigma_thresh

    # Initial keep mask
    keep = raw_data < height   # True = keep, False = remove spike

    # Expand removals by radius
    if radius > 0:
        remove = ~keep
        kernel = np.ones(2 * radius + 1, dtype=int)
        expanded_remove = np.convolve(remove.astype(int), kernel, mode="same") > 0
        keep = ~expanded_remove

    filtered_data = raw_data[keep]
    filtered_tdata = tdata[keep]

    n = len(filtered_data)
    if n == 0:
        raise ValueError("No data left after filtering.")
    if n < target_len:
        raise ValueError(f"Filtered length ({n}) is smaller than target_len ({target_len}).")

    # Downsample to exactly target_len points
    idx = np.linspace(0, n - 1, target_len, dtype=int)
    return filtered_tdata[idx], filtered_data[idx]

def chunk(tdata,data,n_chunks=n_chunks):
    chunk_size = len(data) // n_chunks
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < n_chunks - 1 else len(data)
        chunks.append((tdata[start:end], data[start:end]))
    return chunks

def fft(tdata,data):
    fs = sampling_rate
    n = len(data)

    x = data - np.mean(data)
    N = len(x)

    w = np.hanning(N)   # Hann window

    xw = x * w

    Y = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N, d=1/fs)

    U = (1/N) * np.sum(w**2)

    psd = (1 / (fs * N * U)) * np.abs(Y)**2
    if N % 2 == 0:
        psd[1:-1] *= 2
    else:
        psd[1:] *= 2

    asd = np.sqrt(psd)
    
    return freqs,asd



path = "/data/lbl/run21/raw/continuous_I4_D20250102_T224744/"
pattern = os.path.join(path, "cont_I4_D20250102_T224816_F0002.hdf5")
filepaths = sorted(glob.glob(pattern))

for filepath in filepaths[:1]:
    filename = os.path.basename(filepath)   # just file name
    print("Reading:", filename[:-5]) 

    with h5py.File(filepath, "r") as data:
        print(data['adc1'].keys())
        events = ['event_38','event_39','event_48','event_49'] 

        for event in events:

            waveforms = np.array(data["adc1"][str(event)])
            time_data = np.arange(waveforms.shape[1]) / sampling_rate
            waveform_data = waveforms[channel_number]

            new_tdata, new_data = preprocess(time_data, waveform_data)
            chunks = chunk(new_tdata, new_data)
            count = 0 
            for data_chunk in chunks:
                count+=1
                freqs, asd = fft(*data_chunk)

                fig, axs = plt.subplots(2, 2, figsize=(12, 8))

                axs[0, 0].plot(time_data, waveform_data, label="original data")
                axs[0, 0].set_title("Original waveform")
                axs[0, 0].set_xlabel("Time [s]")
                axs[0, 0].set_ylabel("Amplitude")
                axs[0, 0].legend()

                axs[1, 0].plot(new_tdata, new_data, label="peakless data")
                axs[1, 0].set_title("Waveform after peak removal/downsampling")
                axs[1, 0].set_xlabel("Time [s]")
                axs[1, 0].set_ylabel("Amplitude")
                axs[1, 0].legend()

                axs[0, 1].loglog(freqs, asd, label="fft data")
                axs[0, 1].set_title("FFT magnitude")
                axs[0, 1].set_xlabel("Frequency [Hz]")
                axs[0, 1].set_ylabel(r"[A/$\sqrt{\mathrm{Hz}}$]")
                axs[0, 1].legend()

                axs[1, 1].hist(new_data, bins=100, label="histogram of data")
                axs[1, 1].set_title("Noise histogram")
                axs[1, 1].set_xlabel("Amplitude")
                axs[1, 1].set_ylabel("Count")
                axs[1, 1].legend()

                fig.savefig("/home/ilemleisher/plots/plot_"+str(filename[:-5])+"_"+str(event)+"_chunk_"+str(count)+"_test.png")
                plt.close(fig)