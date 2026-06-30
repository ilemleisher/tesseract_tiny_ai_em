import numpy as np
import os
from sklearn.decomposition import PCA
from utils import get_files, filter_files, get_drift_score
import matplotlib.pyplot as plt

# Reconstruction-error anomaly score
def pca_recon_flags(X, pca, threshold_percentile=99.5):
    """
    This function computes the PCA reconstruction-error per sample and assigns flags based on a defined threshold.

    Inputs:
    - X: noramlized input spectra with shape (N, F) (N: number of samples, F: number of frequency bins)
    - pca: PCA model
    - threshold_percentile: the threshold above which to label an anomaly
    Returns:
    - peak_flags: list of anomaly labels (0 = no anomaly, 1 = anomaly)
    - idx: list of corresponding indices for the anomaly labels
    - residual_data: array of PCA error residuals for each anomalous chunk
    """
    # Project each sample into PCA space
    Z = pca.transform(X)

    # Reconstruct back into original space
    Xhat = pca.inverse_transform(Z)

    # Compute MSE
    err = np.mean((X - Xhat)**2, axis=1)

    # Define PCA threshold based on desired false alarm rate <<< this needs to be tuned
    thr = np.percentile(err, threshold_percentile)

    peak_flags = (err > thr).astype(np.int32)
    idx = np.where(peak_flags == 1)[0]
    print(f"Labeled chunks with indices {idx.tolist()} as anomalous by PCA reconstruction")

    residual_data = X - Xhat

    return peak_flags, idx, residual_data

def ema_baseline_flag(freqs, X, s):
    '''
    This function calculates a drift score for each data chunk and assigns anomaly flags based on a
    defined threshold of s sigma above the mean.

    Parameters:
    - freqs: an array of frequency lists for data chunks
    - X: an array of ASD amplitude (in logspace) lists for the same data chunks
    - s: sigma threshold
    Returns:
    - drift_flags: list of anomaly labels (0 = no anomaly, 1 = anomaly) 
    - idx: list of corresponding indices for the anomaly labels
    - baselines: list of baseline value lists for data chunks
    - residuals: list of residual value lists for data chunks
    - drift_scores: list of drift scores for data chunks
    '''

    drift_scores=[]
    baselines = []
    residuals = []

    for i in range(len(X)):
        baseline, residual, drift_score = get_drift_score(freqs[i], X[i])
        drift_scores.append(drift_score)
        baselines.append(baseline)
        residuals.append(residual)

    thr = np.mean(drift_scores) + s * np.std(drift_scores)
    drift_flags = (drift_scores > thr).astype(np.int32)
    idx = np.where(drift_flags == 1)[0]
    print(f"Labeled chunks with indices {idx.tolist()} as anomalous by EMA baseline drift")

    return drift_flags, idx, baselines, residuals, drift_scores

if __name__ == '__main__':

    # Path to preprocessed data (assumes format from preprocess.py)
    path = "/home/ilemleisher/data/"

    # Dataset
    target = 'I4_D20250102_T230851'

    # Load continuous data from preprocessed files following naming format from preprocess.py
    filenames = filter_files(get_files(path),target)
    print(f"Found {len(filenames)} files")

    freqs_container,asd_container = [],[]

    # Stitch together all continuous data
    for filename in filenames:
        filepath = path+filename
        with np.load(filepath) as d:
            filename = os.path.basename(filepath)
            print("Reading:", filename) 
            freqs_list = d['freqs_list']
            asd_list = d['asd_list']

        freqs_container.append(freqs_list)
        asd_container.append(asd_list)

    freqs_total = np.concatenate(freqs_container)
    asd_total = np.concatenate(asd_container)

    X = np.log10(asd_total + 1e-12).astype(np.float32)
    print(f"Found {len(X)} chunks")

    # PCA
    pca = PCA(n_components=0.99, svd_solver="full")  # keep 99% variance
    pca.fit(X)

    peak_flags, idx, residuals = pca_recon_flags(X, pca, 99.5)

    # Plot each anomalous original ASD with the residuals overlayed
    for index in idx:
        
        f = np.asarray(freqs_total[index]).ravel()       # frequency
        asd = np.asarray(asd_total[index]).ravel()         # ASD
        res = np.abs(np.asarray(residuals[index]).ravel())  # residual magnitude per freq bin

        fig, ax1 = plt.subplots(figsize=(10, 5))

        # ASD curve (log-log)
        ax1.loglog(f, asd, color="tab:blue", lw=1.4, label="ASD")
        ax1.set_xlabel("Frequency")
        ax1.set_ylabel("ASD", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(True, which="both", ls="--", alpha=0.3)

        # Histogram overlay sharing same x-axis (frequency), weighted by residual magnitude
        ax2 = ax1.twinx()
        ax2.plot(f, res,color='tab:red',alpha = 0.5,label='residuals')
        ax2.set_xscale("log")     # ensure same log x scaling
        ax2.set_ylabel("PCA reconstruction residual", color="tab:red")
        ax2.set_ylim(0,2)
        ax2.tick_params(axis="y", labelcolor="tab:red")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left")

        plt.title(f"ASD (log-log) with aligned residuals")
        plt.savefig(f"/home/ilemleisher/plots/residuals/{target}_residuals_chunk_{index}")


    # Get EMA baseline drift flags
    drift_flags, idx, baselines, residuals, drift_scores = ema_baseline_flag(freqs_total, X, 3)

    # For each baseline anomaly, plot the nearest 5 data chunks with baselines overlayed
    for center_idx in idx:
        fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)

        # plot chunks [center_idx-2, ..., center_idx+2]
        for offset, ax in zip(range(-2, 3), axes):
            i = center_idx + offset

            ax.loglog(freqs_total[i], asd_total[i], label="fft data")
            ax.loglog(
                freqs_total[i],
                10**baselines[i],
                "r--",
                label=f"drift score={drift_scores[i]:.3f}",
            )
            ax.set_title(f"Chunk {i}")
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel(r"[A/$\sqrt{\mathrm{Hz}}$]")
            ax.legend()

        plt.tight_layout()
        plt.savefig(f"/home/ilemleisher/plots/residuals/{target}_drift_chunk_{center_idx}")
        plt.close(fig)
        

    np.savez_compressed(f"{path}labels/pca_labels_{target}.npz",pca_labels=peak_flags.astype(np.int8),
                        ema_labels=drift_flags.astype(np.int8))