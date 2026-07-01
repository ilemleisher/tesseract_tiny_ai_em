import numpy as np
import sys
from sklearn.decomposition import PCA
sys.path.append("/home/ilemleisher/em_project/dev/utils.py") 
from utils import track_runtime

def get_drift_score(freqs, amplitude):
    """
    This function fits a linear baseline to the given frequency and amplitude data using least squares regression and calculates
    a baseline drift score using the residual.

    Parameters:
    - freqs: frequency bins array
    - amplitude: amplitude values array
    
    Returns:
    - baseline: fitted linear baseline values corresponding to the frequency bins
    - residual: difference between the original amplitude values and the fitted baseline
    - drift_score: mean of the baseline residual
    """

    f = freqs
    a = amplitude
    if f.ndim != 1 or a.ndim != 1 or f.shape[0] != a.shape[0]:
        raise ValueError("freqs and amplitude must be 1D and same length")
    m, b = np.polyfit(f, a, deg=1)
    baseline = m * f + b
    residual = a - baseline

    drift_score = np.mean(np.abs(residual))

    return baseline, residual, drift_score

@track_runtime
def flag(freqs, X, s=3):
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
    - metadata:
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

    metadata = {}
    metadata['baselines'] = baselines
    metadata['residuals'] = residuals
    metadata['drift_scores'] = drift_scores

    return drift_flags, idx, metadata