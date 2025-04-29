import numpy as np
import pandas as pd
from scipy.signal import hilbert, argrelextrema
from scipy.stats import entropy as shannon_entropy
from sympy import primerange
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
import os

# --------------------------
# Utility Functions
# --------------------------
def load_spectrum_from_txt(path, num_freqs_to_load=None):
    """
    Load a spectrum (1D array of positive frequencies) from a text file.
    Each line can contain one or more whitespace-separated numbers.
    """
    freqs = []
    with open(path, 'r') as f:
        count = 0
        for line in f:
            if num_freqs_to_load is not None and count >= num_freqs_to_load:
                break
            parts = line.strip().split()
            for part in parts:
                try:
                    val = float(part)
                    if val > 1e-9:
                        freqs.append(val)
                        count += 1
                        if num_freqs_to_load is not None and count >= num_freqs_to_load:
                            break
                except ValueError:
                    continue
            if num_freqs_to_load is not None and count >= num_freqs_to_load:
                break
    return np.array(freqs)

def is_prime_array(N_min, N_max):
    primes = set(primerange(N_min, N_max+1))
    return np.array([1 if n in primes else 0 for n in range(N_min, N_max+1)])

# Robust Normalization Function
def safe_norm01(x):
    """Normalize array x to [0, 1], handling potential division by zero."""
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    if range_val < 1e-12: # Check if range is effectively zero
        # Return array of 0s or 0.5s if range is zero, shape like input
        return np.zeros_like(x) if min_val == 0 else np.full_like(x, 0.5)
    return (x - min_val) / range_val

# --------------------------
# Generalized Spectral Feature Extractor
# --------------------------
class SpectralFieldFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    scikit-learn compatible transformer that computes field features for each integer N
    using any input spectrum (array of positive frequencies).
    Features: amplitude, curvature, frequency, entropy, harmonics, extrema, etc.
    Note: The input X to transform is ignored; features are generated based on init parameters.
    """
    def __init__(self, spectrum, N_min=2, N_max=1000, num_points=5000, entropy_bins=32, random_state=0):
        self.spectrum = np.array(spectrum)
        self.N_min = N_min
        self.N_max = N_max
        self.num_points = num_points
        self.entropy_bins = entropy_bins
        self.random_state = random_state

    def fit(self, X=None, y=None):
        # No fitting needed for feature extraction
        return self

    def transform(self, X=None):
        # Compute features for all N in [N_min, N_max]
        # The input X is ignored here, as features depend only on the spectrum and N range.
        np.random.seed(self.random_state)
        spectrum = self.spectrum / self.spectrum[0]
        y_grid = np.linspace(np.log(self.N_min), np.log(self.N_max), self.num_points)
        N_grid = np.exp(y_grid)
        dy = y_grid[1] - y_grid[0]
        weights = 1 / (1 + spectrum**2)
        V0 = np.sum(weights[:, None] * np.cos(spectrum[:, None] * y_grid), axis=0)
        analytic = hilbert(V0)
        amplitude = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        frequency = np.gradient(phase, dy)
        curvature = np.gradient(np.gradient(amplitude, dy), dy)
        # Use safe normalization
        amp_norm = safe_norm01(amplitude)
        curv_norm = safe_norm01(np.abs(curvature))
        freq_norm = safe_norm01(np.abs(frequency))
        # Entropy (windowed)
        window = max(5, int(0.01 * self.num_points))
        ent = np.zeros_like(amp_norm)
        half_w = window // 2
        padded = np.pad(amp_norm, (half_w, half_w), mode='reflect')
        for i in range(len(amp_norm)):
            win = padded[i:i+window]
            hist, _ = np.histogram(win, bins=self.entropy_bins, density=True)
            ent[i] = shannon_entropy(hist[hist > 1e-12])
        # Normalize entropy: low entropy (high order) -> high score
        # Use safe_norm01 on the raw entropy before inverting
        ent_norm_raw = safe_norm01(ent)
        ent_score = 1.0 - ent_norm_raw
        # Harmonic content: local std of cos(freq_k y)
        harmonics = np.std(np.cos(spectrum[:, None] * y_grid), axis=0)
        harmonics_norm = 1.0 - safe_norm01(harmonics) # High std (low coherence) -> low score
        # Extrema (min/max indicators)
        is_max = np.zeros_like(amp_norm)
        is_min = np.zeros_like(amp_norm)
        max_idx = argrelextrema(amp_norm, np.greater)[0]
        min_idx = argrelextrema(amp_norm, np.less)[0]
        is_max[max_idx] = 1
        is_min[min_idx] = 1
        # Interpolate features to integer N
        N_ints = np.arange(self.N_min, self.N_max+1)
        features = np.stack([
            np.interp(N_ints, N_grid, amp_norm),
            np.interp(N_ints, N_grid, curv_norm),
            np.interp(N_ints, N_grid, freq_norm),
            np.interp(N_ints, N_grid, ent_score),
            np.interp(N_ints, N_grid, harmonics_norm),
            np.interp(N_ints, N_grid, is_max),
            np.interp(N_ints, N_grid, is_min)
        ], axis=1)
        return features

# --------------------------
# MLP Classifier (unchanged)
# --------------------------
class SpectralMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    scikit-learn compatible MLP classifier for spectral field features.
    """
    def __init__(self, hidden_layer_sizes=(32, 16), random_state=0, max_iter=500):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, random_state=self.random_state, max_iter=self.max_iter)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

# --------------------------
# Example Usage & Comparison
# --------------------------
if __name__ == "__main__":
    # Parameters
    spectrum_file = "zetazeros-50k.txt"  # Can be any spectrum file (zeta zeros, GUE, etc.)
    num_freqs = 1000
    N_min, N_max = 2, 1000
    random_state = 42
    # Load spectrum (can be zeta zeros, GUE, or any positive frequencies)
    spectrum = load_spectrum_from_txt(spectrum_file, num_freqs)
    # Feature extraction
    extractor = SpectralFieldFeatureExtractor(spectrum=spectrum, N_min=N_min, N_max=N_max, num_points=5000, random_state=random_state)
    # Pass a dummy X (e.g., None or an empty array), it will be ignored by transform
    X = extractor.transform(X=None)
    y = is_prime_array(N_min, N_max)
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)
    # Spectral MLP
    spectral_clf = SpectralMLPClassifier(random_state=random_state)
    spectral_clf.fit(X_train, y_train)
    y_pred = spectral_clf.predict(X_test)
    y_proba = spectral_clf.predict_proba(X_test)[:,1]
    # Baselines
    logreg = LogisticRegression(max_iter=500, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    svm = SVC(probability=True, random_state=random_state)
    for clf, name in zip([logreg, rf, svm], ["LogisticRegression", "RandomForest", "SVM"]):
        clf.fit(X_train, y_train)
        y_pred_b = clf.predict(X_test)
        y_proba_b = clf.predict_proba(X_test)[:,1]
        auroc = roc_auc_score(y_test, y_proba_b)
        ap = average_precision_score(y_test, y_proba_b)
        prec, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_b, average='binary')
        print(f"{name}: AUROC={auroc:.3f}, AP={ap:.3f}, Precision={prec:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    # Spectral MLP metrics
    auroc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    prec, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print(f"SpectralMLP: AUROC={auroc:.3f}, AP={ap:.3f}, Precision={prec:.3f}, Recall={recall:.3f}, F1={f1:.3f}") 