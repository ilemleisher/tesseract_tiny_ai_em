import numpy as np
import os, glob
from sklearn.decomposition import PCA

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
    # Project each sapmle into PCA space
    Z = pca.transform(X)

    # Reconstruct back into original space
    Xhat = pca.inverse_transform(Z)

    # Compute MSE
    err = np.mean((X - Xhat)**2, axis=1)
    return err

# Path to preprocessed data (assumes format from preprocess.py)
path = "/home/ilemleisher/data/"

# Define number of files to analyze
number_of_files = 1

# Load all preprocessed files following naming format from preprocess.py
pattern = os.path.join(path, "data_cont_I4_D*_T*_F*.npz")
files = sorted(glob.glob(pattern))
print(f"Found {len(files)} files")
for filepath in files[:int(number_of_files)]:
    with np.load(filepath) as d:
        filename = os.path.basename(filepath)
        print("Reading:", filename[:-4]) 
        freqs_list = d['freqs_list']
        asd_list = d['asd_list']

        X = np.log10(asd_list + 1e-12).astype(np.float32)

        print(f"Found {len(X)} chunks")

        # Calculate references stats for normalization
        mu = X.mean(axis=(0,1), keepdims=True)
        sigma = X.std(axis=(0,1), keepdims=True) + 1e-8
        X = (X - mu) / sigma

        # PCA
        pca = PCA(n_components=0.99, svd_solver="full")  # keep 99% variance
        pca.fit(X)

        err = pca_recon_error(X, pca)

        # Define threshold based on desired false alarm rate
        thr = np.percentile(err, 99.0)

        pred_x = (err > thr).astype(np.int32)
        idx = np.where(pred_x == 1)[0]
        print(f"Chunks with indices {idx.tolist()} are anomalous")

        np.savez_compressed(f"{path}pca_labels_{filename[:-4]}.npz",pseudo_labels=pred_x.astype(np.int8))