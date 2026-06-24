import h5py, glob, os
import numpy as np

# Variables
n_chunks = 2
post_downsample_length = 12000
sampling_rate = 1.25e6
channel_number = 0

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

def downsample(tdata, data, target_len=12000):

    raw_data = np.asarray(data).copy()
    tdata = np.asarray(tdata)

    # Downsample to exactly target_len points
    idx = np.linspace(0, len(tdata) - 1, target_len, dtype=int)
    return tdata[idx], raw_data[idx]

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

def linear_baseline(freqs, amplitude):

    f = freqs
    a = amplitude

    if f.ndim != 1 or a.ndim != 1 or f.shape[0] != a.shape[0]:
        raise ValueError("freqs and amplitude must be 1D and same length")

    m, b = np.polyfit(f, a, deg=1)

    baseline = m * f + b
    residual = a - baseline
    return baseline, residual

def filter_chunks(chunks, sigma_thresh=5):

    filtered_chunks = []

    for chunk in chunks:
        data = chunk[1]
        med = np.median(data)
        height = med + np.std(data) * sigma_thresh
        if np.any(data > height):
            continue
        filtered_chunks.append(chunk)

    return filtered_chunks


if __name__ == '__main__':

    # Path to the folder containing the .hdf5 files
    path = "/data/lbl/run21/raw/continuous_I4_D20250102_T224744/"
    # Pattern to find all .hdf5 files in the folder
    pattern = os.path.join(path, "cont_I4_D*_T*_F*.hdf5")
    filepaths = sorted(glob.glob(pattern))
    print(f"Found {len(filepaths)} files")

    # Loop over each file in the folder
    for filepath in filepaths[10:]:
        filename = os.path.basename(filepath)
        print("Reading:", filename[:-5]) 

        with h5py.File(filepath, "r") as data:
            events = data['adc1'].keys()
            print("Found", len(events), "events")

            # Data containers
            freqs_list, asd_list, waveform_data_list = [], [], []
            new_tdata_list, new_data_list, time_data_list = [], [], []

            # Loop over each event in the file
            for event in events:

                # Read ADC1 output from the specified channel
                waveforms = np.array(data["adc1"][str(event)])
                time_data = np.arange(waveforms.shape[1]) / sampling_rate
                waveform_data = waveforms[channel_number]

                # Downsample the raw waveform data
                new_tdata, new_data = downsample(time_data, waveform_data)
                # Divide the raw data into chunks
                chunks = chunk(new_tdata, new_data)
                # Discard any chunks that contains peaks above 5 sigma
                filtered_chunks = filter_chunks(chunks)

                # Loop over each remaining chunk
                for data_chunk in filtered_chunks:

                    # Compute the ASD for each chunk
                    freqs, asd = fft(*data_chunk)

                    freqs_list.append(freqs.astype(np.float32))
                    asd_list.append(asd.astype(np.float32))
                    waveform_data_list.append(waveform_data.astype(np.float32))
                    new_tdata_list.append(new_tdata.astype(np.float32))
                    new_data_list.append(new_data.astype(np.float32))
                    time_data_list.append(time_data.astype(np.float32))

        np.savez_compressed(
            f"/home/ilemleisher/data/data_{filename[:-5]}.npz",
            freqs_list=np.stack(freqs_list),         # (N, F) or (F,)
            asd_list=np.stack(asd_list),             # (N, F)
            # optional metadata
            new_tdata_list=np.stack(new_tdata_list),
            new_data_list=np.stack(new_data_list),
            time_data_list=np.stack(time_data_list),
            waveform_data_list=np.stack(waveform_data_list)
)
