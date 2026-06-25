import numpy as np
import os
from sklearn.decomposition import PCA
from utils import get_files, filter_files, linear_baseline

# Reconstruction-error anomaly score
def pca_recon_error(X, pca):
    """
    This function computes the PCA reconstruction-error per sample

    Inputs:
    - X: noramlized input spectra with shape (N, F) (N: number of samples, F: number of frequency bins)
    - pca: PCA model
    Returns:
    - err: mean squared reconstruction error for each sample
    """
    # Project each sample into PCA space
    Z = pca.transform(X)

    # Reconstruct back into original space
    Xhat = pca.inverse_transform(Z)

    # Compute MSE
    err = np.mean((X - Xhat)**2, axis=1)
    return err

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

err = pca_recon_error(X, pca)

# Define PCA threshold based on desired false alarm rate <<< this needs to be tuned
thr = np.percentile(err, 99.9)

pred_x = (err > thr).astype(np.int32)
idx = np.where(pred_x == 1)[0]
print(f"Labeled chunks with indices {idx.tolist()} as anomalous by PCA reconstruction")

# 5 sigma threshold for EMA
thr = np.mean(drift_scores) + 5 * np.std(drift_scores)
drift_flags = (drift_scores > thr).astype(np.int32)
idx = np.where(drift_flags == 1)[0]
print(f"Labeled chunks with indices {idx.tolist()} as anomalous by EMA baseline drift")

np.savez_compressed(f"{path}labels/pca_labels_{target}.npz",pca_labels=pred_x.astype(np.int8),
                    ema_labels=drift_flags.astype(np.int8))