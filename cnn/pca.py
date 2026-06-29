import numpy as np
import os
from sklearn.decomposition import PCA
from utils import get_files, filter_files, linear_baseline
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
    - pred_x: list of anomaly labels (0 = no anomaly, 1 = anomaly)
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

    pred_x = (err > thr).astype(np.int32)
    idx = np.where(pred_x == 1)[0]
    print(f"Labeled chunks with indices {idx.tolist()} as anomalous by PCA reconstruction")

    residual_data = X[idx] - Xhat[idx]

    return pred_x, residual_data

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

    # Calculate references stats for normalization
    mu = X.mean(axis=(0,1), keepdims=True)
    sigma = X.std(axis=(0,1), keepdims=True) + 1e-8
    X = (X - mu) / sigma

    # EMA baseline shift
    drift_scores = []
    for i in range(len(X)):
        baseline, residual = linear_baseline(freqs_total[i], X[i])
        drift_scores.append(np.mean(np.abs(residual)))

    # PCA
    pca = PCA(n_components=0.99, svd_solver="full")  # keep 99% variance
    pca.fit(X)

    pred_x, residuals = pca_recon_flags(X, pca, 99.5)

    # Plot each anomalous original ASD with the residuals overlayed
    for i in range(len(residuals)):
        
        f = np.asarray(freqs_total[i]).ravel()       # frequency
        asd = np.asarray(asd_total[i]).ravel()         # ASD
        res = np.abs(np.asarray(residuals[i]).ravel())  # residual magnitude per freq bin

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
        plt.savefig(f"/home/ilemleisher/plots/residuals_{i}")


    # 5 sigma threshold for EMA
    thr = np.mean(drift_scores) + 5 * np.std(drift_scores)
    drift_flags = (drift_scores > thr).astype(np.int32)
    idx = np.where(drift_flags == 1)[0]
    print(f"Labeled chunks with indices {idx.tolist()} as anomalous by EMA baseline drift")

    np.savez_compressed(f"{path}labels/pca_labels_{target}.npz",pca_labels=pred_x.astype(np.int8),
                        ema_labels=drift_flags.astype(np.int8))