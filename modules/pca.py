import numpy as np
import os, sys
from sklearn.decomposition import PCA
sys.path.append("/home/ilemleisher/em_project/dev/utils.py") 
from utils import track_runtime

# Reconstruction-error anomaly score
@track_runtime
def flag(freqs, X, thr=0.0015):
    """
    This function computes the PCA reconstruction-error per sample and assigns flags based on a defined threshold.

    Inputs:
    - freqs: list of frequency lists for data chunks
    - X: noramlized input spectra with shape (N, F) (N: number of samples, F: number of frequency bins)
    - thr: the threshold above which to label an anomaly
    Returns:
    - peak_flags: list of anomaly labels (0 = no anomaly, 1 = anomaly)
    - idx: list of corresponding indices for the anomaly labels
    - metadata:
        - pca_components: PCA components used for reconstruction
        - pca_explained_variance: PCA explained variance for each component
        - pca_explained_variance_ratio: PCA explained variance ratio for each component
        - residual_data: array of PCA error residuals for each anomalous chunk
    """
    # PCA
    pca = PCA(n_components=0.99, svd_solver="full")  # keep 99% variance
    pca.fit(X)
    
    # Project each sample into PCA space
    Z = pca.transform(X)

    # Reconstruct back into original space
    Xhat = pca.inverse_transform(Z)

    # Compute MSE
    err = np.mean((X - Xhat)**2, axis=1)

    peak_flags = (err > thr).astype(np.int32)
    idx = np.where(peak_flags == 1)[0]

    residual_data = X - Xhat
    metadata = {}
    metadata['pca_components'] = pca.components_
    metadata['residual_data'] = residual_data

    return peak_flags, idx, metadata