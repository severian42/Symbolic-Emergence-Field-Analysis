import numpy as np
from scipy.signal import hilbert, argrelextrema, chirp, butter, filtfilt, savgol_filter, find_peaks
from scipy.stats import entropy as shannon_entropy
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, validate_data
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from typing import Optional, List, Dict, Any
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view
import warnings
import copy


def safe_norm01(x):
    """Scales an array to the range [0, 1], handling cases with zero range."""
    x = np.asarray(x)
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    if range_val < 1e-12:
        return np.zeros_like(x) if min_val == 0 else np.full_like(x, 0.5)
    return (x - min_val) / range_val


def safe_norm_maxabs(x):
    """Scales an array by its maximum absolute value, preserving sign."""
    x = np.asarray(x)
    max_abs_val = np.max(np.abs(x))
    if max_abs_val < 1e-12:
        return np.zeros_like(x)
    return x / max_abs_val


def calculate_channel_entropy(feature_values: np.ndarray, bins: int = 32) -> float:
    """Calculates entropy of the global distribution of a feature set using natural log."""
    feature_values = feature_values.flatten()
    feature_values = feature_values[np.isfinite(feature_values)]
    if len(feature_values) == 0:
        return np.log(bins) # Max entropy if no valid data

    # Use histogram for distribution estimation
    hist, bin_edges = np.histogram(feature_values, bins=bins, density=True)
    probs = hist * np.diff(bin_edges)
    probs = probs[probs > 1e-12]
    if len(probs) == 0:
        return np.log(bins)
    probs /= probs.sum()
    return -np.sum(probs * np.log(probs)) # Natural log


class GeneralizedSpectralFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts local geometric and information-theoretic features (A, C, F, E)
    from 1D signals, normalized appropriately for SEFA.md.
    Outputs a 3D array: (n_samples, n_points, n_features=4).
    """
    FEATURE_NAMES = ['amplitude_norm', 'curvature_norm', 'frequency_norm', 'entropy_align_norm']

    def __init__(self, entropy_bins=32, entropy_window_ratio=0.03,
                 use_savgol=False, savgol_window=None, savgol_polyorder=3,
                 n_jobs=1):
        self.entropy_bins = entropy_bins
        self.entropy_window_ratio = entropy_window_ratio
        self.use_savgol = use_savgol
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit the extractor. Learns expected number of input points."""
        X = validate_data(self, X, ensure_2d=True, dtype=np.float64, y=None)
        self.n_features_in_ = X.shape[1]
        self._set_output_feature_names()
        return self

    def _set_output_feature_names(self):
        """Sets the names for the 4 local features."""
        self.feature_names_out_ = ['amplitude_norm', 'curvature_norm', 'frequency_norm', 'entropy_align_norm']

    def _validate_savgol_window(self, n_points):
        """Helper to validate/adjust savgol window based on n_points."""
        savgol_window = self.savgol_window
        use_savgol = self.use_savgol
        if use_savgol:
            if savgol_window is None:
                savgol_window = min(51, n_points // 5)
            # Ensure window is valid
            savgol_window = max(self.savgol_polyorder + 1, savgol_window)
            if savgol_window % 2 == 0: savgol_window += 1
            if savgol_window >= n_points: savgol_window = n_points - 1 if n_points > 1 else 1
            if savgol_window % 2 == 0 and savgol_window > 1: savgol_window -= 1
            if savgol_window <= self.savgol_polyorder:
                warnings.warn(f"Adjusted SavGol window ({savgol_window}) too small for polyorder ({self.savgol_polyorder}). Disabling SavGol for transform.", UserWarning)
                use_savgol = False
                savgol_window = None
        return use_savgol, savgol_window

    def transform(self, X):
        """Transforms signals into local, normalized feature space."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_2d=True, dtype=np.float64)
        n_samples, n_points = X.shape

        if n_points != self.n_features_in_:
            raise ValueError(f"Input shape mismatch: expected {self.n_features_in_} points, got {n_points}.")

        win = max(5, int(self.entropy_window_ratio * n_points))
        # Use consistent bins based on self.entropy_bins
        bins = np.linspace(0, 1, self.entropy_bins + 1)

        use_savgol_transform, savgol_window_transform = self._validate_savgol_window(n_points)

        def process_sample(signal, sample_idx):
            """Process a single signal, return stacked local feature arrays."""
            try:
                signal_proc = signal
                if use_savgol_transform and savgol_window_transform is not None:
                    signal_proc = savgol_filter(signal, window_length=savgol_window_transform, polyorder=self.savgol_polyorder)

                analytic = hilbert(signal_proc)
                amp = np.abs(analytic)
                phase = np.unwrap(np.angle(analytic))

                if use_savgol_transform and savgol_window_transform is not None:
                    freq = savgol_filter(phase, window_length=savgol_window_transform, polyorder=self.savgol_polyorder, deriv=1)
                    amp_smooth = savgol_filter(amp, window_length=savgol_window_transform, polyorder=self.savgol_polyorder)
                    curv = savgol_filter(amp_smooth, window_length=savgol_window_transform, polyorder=self.savgol_polyorder, deriv=2)
                else:
                    freq = np.gradient(phase)
                    curv = np.gradient(np.gradient(amp))
            except Exception as e:
                warnings.warn(f"Error processing signal {sample_idx}: {e}. Returning zeros.", RuntimeWarning)
                amp = np.zeros_like(signal)
                freq = np.zeros_like(signal)
                curv = np.zeros_like(signal)

            # --- Normalization ---
            amp_norm = safe_norm01(amp)         # A': [0, 1]
            curv_norm = safe_norm_maxabs(curv)  # C': [-1, 1] sign preserved
            freq_norm = safe_norm_maxabs(freq)  # F': [-1, 1] sign preserved

            # --- Local Entropy ---
            ent = np.zeros(n_points)
            try:
                pad = np.pad(amp_norm, (win // 2, win // 2), mode='reflect')
                windows = sliding_window_view(pad, win)[:n_points]
                for j in range(n_points):
                    window_data = windows[j]
                    if np.all(window_data == window_data[0]):
                        ent[j] = 0.0
                        continue
                    # Use pre-defined bins based on normalized amplitude [0,1]
                    hist, _ = np.histogram(window_data, bins=bins, density=True)
                    hist_valid = hist[hist > 1e-12]
                    if len(hist_valid) > 0:
                        probs = hist_valid * np.diff(bins)[0] # Convert density using fixed bin width
                        probs /= probs.sum() # Ensure sum to 1
                        ent[j] = -np.sum(probs * np.log(probs)) # Natural log
                    else:
                        ent[j] = 0.0
            except Exception as e_ent:
                 warnings.warn(f"Error calculating entropy for signal {sample_idx}: {e_ent}. Setting entropy to zero.", RuntimeWarning)
                 ent = np.zeros(n_points)

            ent_align_norm = 1.0 - safe_norm01(ent) # E': [0, 1]

            # Stack the four normalized local features
            return np.stack([amp_norm, curv_norm, freq_norm, ent_align_norm], axis=-1)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_sample)(X[i], i) for i in range(n_samples)
        )
        X_local_features = np.stack(results, axis=0)
        return X_local_features

    def get_feature_names_out(self, input_features=None):
        """Get output feature names (local features)."""
        check_is_fitted(self)
        # Use feature_names_ attribute set during fit
        return np.array(self.feature_names_out_, dtype=object)


class SEFACalibrator(BaseEstimator, TransformerMixin):
    """
    Performs SEFA self-calibration and calculates the local SEFA score.

    Takes the 3D output (n_samples, n_points, 4) of
    GeneralizedSpectralFeatureExtractor during `fit` to calculate the alpha
    exponents based on global information deficit. The `transform` method takes
    the same 3D input and calculates the local SEFA score (n_samples, n_points)
    using the stored alphas.
    """
    FEATURE_NAMES = ['amplitude_norm', 'curvature_norm', 'frequency_norm', 'entropy_align_norm']

    def __init__(self, entropy_bins=32, epsilon=1e-16):
        self.entropy_bins = entropy_bins
        self.epsilon = epsilon

    def fit(self, X_local_norm, y=None):
        """
        Fit the calibrator: Perform self-calibration to find alpha exponents.
        Input X_local_norm is the output of GeneralizedSpectralFeatureExtractor
        (n_samples, n_points, 4).
        """
        X_local_norm = validate_data(self, X_local_norm, ensure_2d=False, ensure_min_features=4, dtype=np.float64, y=None, allow_nd=True)
        if X_local_norm.ndim != 3 or X_local_norm.shape[2] != 4:
            raise ValueError("Input X_local_norm must be a 3D array with shape (n_samples, n_points, 4).")

        self.n_features_in_ = X_local_norm.shape[1] # Store number of points

        # Perform self-calibration (Calculate Alphas)
        weights = {}
        self.alpha_ = {}
        max_entropy = np.log(self.entropy_bins) # Natural log

        for i, fname in enumerate(self.FEATURE_NAMES):
            feature_data = X_local_norm[:, :, i]
            I_x = calculate_channel_entropy(feature_data, bins=self.entropy_bins)
            # Clamp deficit to >= 0 (Limitation 2.3 in SEFA.md)
            w_x = max(0, max_entropy - I_x)
            weights[fname] = w_x

        W_total = sum(weights.values())
        if W_total < 1e-12:
            warnings.warn("Total information deficit near zero. Assigning equal alpha weights.", RuntimeWarning)
            num_features = len(self.FEATURE_NAMES)
            p = num_features
            for fname in self.FEATURE_NAMES:
                 # Assign equal weight (p/p = 1) *before* scaling by p? No, follow formula.
                 # Alpha[X] = p * w[X] / WTotal. If WTotal is 0, w[X] must be 0.
                 # Let's assign alpha = 1.0, which means Sum(alpha)=p, effectively equal contribution.
                self.alpha_[fname] = 1.0 # Equal contribution
        else:
            p = len(self.FEATURE_NAMES) # Scaling factor p
            for fname in self.FEATURE_NAMES:
                self.alpha_[fname] = p * weights[fname] / W_total

        # No output feature names needed directly from this transformer
        return self

    def transform(self, X_local_norm):
        """
        Calculate the local SEFA score using stored alphas.
        Input X_local_norm is the output of GeneralizedSpectralFeatureExtractor
        (n_samples, n_points, 4).
        Output is the local SEFA score (n_samples, n_points).
        """
        check_is_fitted(self)
        X_local_norm = validate_data(self, X_local_norm, reset=False, ensure_2d=False, ensure_min_features=4, dtype=np.float64, allow_nd=True)
        if X_local_norm.ndim != 3 or X_local_norm.shape[2] != 4:
            raise ValueError("Input X_local_norm must be a 3D array with shape (n_samples, n_points, 4).")
        if X_local_norm.shape[1] != self.n_features_in_:
             raise ValueError(f"Input n_points mismatch: expected {self.n_features_in_} points, got {X_local_norm.shape[1]}.")

        n_samples, n_points, _ = X_local_norm.shape
        log_sefa_local = np.zeros((n_samples, n_points), dtype=np.float64)

        for i, fname in enumerate(self.FEATURE_NAMES):
            feature_norm = X_local_norm[:, :, i]
            alpha = self.alpha_[fname]

            # Implement sign-aware epsilon clipping (SEFA.md Eq. 23 footnote)
            # X_clip = Sign(X') * Max(eps, Abs(X'))
            # Log is then applied to Abs(X_clip)
            feature_clipped_abs = np.maximum(self.epsilon, np.abs(feature_norm))

            # Add weighted log absolute value. Add epsilon inside log for safety.
            log_sefa_local += alpha * np.log(feature_clipped_abs) # Natural log, eps already handled by Max

        # Exponentiate to get final local SEFA score
        sefa_local = np.exp(log_sefa_local)

        # Handle potential overflows or NaNs after exp
        sefa_local = np.nan_to_num(sefa_local, nan=0.0, posinf=np.finfo(np.float64).max, neginf=0.0)

        return sefa_local

    def get_feature_names_out(self, input_features=None):
        """
        Return generic feature names for the local SEFA score output.
        Output shape is (n_samples, n_points), so names are sefa_local_0, sefa_local_1, ...
        """
        check_is_fitted(self)
        n_points = getattr(self, 'n_features_in_', None)
        if n_points is None:
            raise AttributeError("SEFACalibrator must be fitted before calling get_feature_names_out.")
        return np.array([f"sefa_local_{i}" for i in range(n_points)], dtype=object)


class SEFAScoreAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregates the local SEFA score profile into summary statistics.

    Takes the 2D local SEFA score (n_samples, n_points) output by
    SEFACalibrator and calculates specified aggregate statistics (mean, std, etc.)
    across the points dimension.
    Outputs a 2D array (n_samples, n_aggregations).
    """
    SEFA_AGG_FUNCS = {
        'mean': np.nanmean, 'std': np.nanstd, 'min': np.nanmin, 'max': np.nanmax,
        'median': np.nanmedian, 'ptp': np.ptp # peak-to-peak
        # Consider adding skew, kurtosis if needed
    }

    def __init__(self, sefa_score_aggregations=['mean', 'std', 'max', 'min']):
        self.sefa_score_aggregations = sefa_score_aggregations

    def fit(self, X_local_sefa, y=None):
        """Fit the aggregator. Mainly sets up feature names."""
        X_local_sefa = validate_data(self, X_local_sefa, ensure_2d=True, dtype=np.float64, y=None)
        self.n_features_in_ = X_local_sefa.shape[1] # Store number of points
        self._set_output_feature_names()
        return self

    def transform(self, X_local_sefa):
        """Aggregate local SEFA score for each sample."""
        check_is_fitted(self)
        X_local_sefa = validate_data(self, X_local_sefa, reset=False, ensure_2d=True, dtype=np.float64)
        if X_local_sefa.shape[1] != self.n_features_in_:
             raise ValueError(f"Input n_points mismatch: expected {self.n_features_in_} points, got {X_local_sefa.shape[1]}.")

        X_sefa_agg = []
        valid_aggregations = []
        for stat_name in self.sefa_score_aggregations:
            if stat_name in self.SEFA_AGG_FUNCS:
                agg_func = self.SEFA_AGG_FUNCS[stat_name]
                # Apply aggregation along the points axis (axis=1)
                try:
                    aggregated_feature = agg_func(X_local_sefa, axis=1)
                    X_sefa_agg.append(aggregated_feature)
                    valid_aggregations.append(stat_name)
                except Exception as e:
                     warnings.warn(f"Could not compute aggregation '{stat_name}': {e}. Skipping.", UserWarning)
            else:
                warnings.warn(f"Unsupported SEFA aggregation stat: {stat_name}. Skipping.", UserWarning)

        if not X_sefa_agg:
            raise ValueError("No valid SEFA aggregation statistics provided or computed.")

        # Combine aggregated features into final 2D feature matrix
        X_sefa_final = np.stack(X_sefa_agg, axis=-1)
        return X_sefa_final

    def _set_output_feature_names(self):
        """Sets feature names based on the chosen aggregations."""
        self.feature_names_out_ = []
        for stat_name in self.sefa_score_aggregations:
             if stat_name in self.SEFA_AGG_FUNCS: # Check if stat is valid
                self.feature_names_out_.append(f"sefa_agg_{stat_name}")

    def get_feature_names_out(self, input_features=None):
        """Get output feature names (aggregated SEFA score features)."""
        check_is_fitted(self)
        if not hasattr(self, 'feature_names_out_') or not self.feature_names_out_:
             self._set_output_feature_names() # Ensure it's set
        return np.array(self.feature_names_out_, dtype=object)


class SEFALocalProfileFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts features from the local SEFA score profile using peak analysis.

    Takes the 2D local SEFA score (n_samples, n_points) output by
    SEFACalibrator and extracts features like peak count, heights, widths, etc.
    Outputs a 2D array (n_samples, n_profile_features).
    """
    def __init__(self, peak_height_rel=0.1, peak_prominence_rel=0.05, peak_min_dist_frac=0.01):
        # Parameters for scipy.signal.find_peaks, relative to max height/prominence
        self.peak_height_rel = peak_height_rel           # Minimum height as fraction of max SEFA score
        self.peak_prominence_rel = peak_prominence_rel # Minimum prominence as fraction of max SEFA score
        self.peak_min_dist_frac = peak_min_dist_frac     # Minimum distance between peaks as fraction of total points

    def fit(self, X_local_sefa, y=None):
        """Fit the extractor. Sets up feature names."""
        X_local_sefa = validate_data(self, X_local_sefa, ensure_2d=True, dtype=np.float64, y=None)
        self.n_features_in_ = X_local_sefa.shape[1] # Store number of points
        self._set_output_feature_names()
        return self

    def transform(self, X_local_sefa):
        """Extract features from the local SEFA score profile."""
        check_is_fitted(self)
        X_local_sefa = validate_data(self, X_local_sefa, reset=False, ensure_2d=True, dtype=np.float64)
        if X_local_sefa.shape[1] != self.n_features_in_:
             raise ValueError(f"Input n_points mismatch: expected {self.n_features_in_} points, got {X_local_sefa.shape[1]}.")

        n_samples, n_points = X_local_sefa.shape
        min_dist = max(1, int(self.peak_min_dist_frac * n_points))

        all_profile_features = []
        for i in range(n_samples):
            profile = X_local_sefa[i, :]
            max_prof_val = np.max(profile)
            if max_prof_val < 1e-12: # Handle flat zero profiles
                 all_profile_features.append(np.zeros(len(self.feature_names_out_)))
                 continue

            # Determine dynamic thresholds based on profile max
            min_h = self.peak_height_rel * max_prof_val
            min_p = self.peak_prominence_rel * max_prof_val

            try:
                # Find peaks
                peaks, properties = find_peaks(
                    profile,
                    height=min_h,
                    prominence=min_p,
                    distance=min_dist,
                    width=(None, None) # Request width calculation
                )

                num_peaks = len(peaks)
                if num_peaks == 0:
                    # Append zeros for all features if no peaks found
                    all_profile_features.append(np.zeros(len(self.feature_names_out_)))
                    continue

                # Extract peak properties
                heights = properties['peak_heights']
                prominences = properties['prominences']
                widths = properties['widths'] # Full width at half prominence

                # Calculate features
                features = {
                    'num_peaks': num_peaks,
                    'mean_peak_height': np.mean(heights),
                    'max_peak_height': np.max(heights),
                    'std_peak_height': np.std(heights),
                    'mean_peak_prominence': np.mean(prominences),
                    'max_peak_prominence': np.max(prominences),
                    'std_peak_prominence': np.std(prominences),
                    'mean_peak_width': np.mean(widths),
                    'max_peak_width': np.max(widths),
                    'std_peak_width': np.std(widths),
                    'highest_peak_loc': peaks[np.argmax(heights)] / n_points, # Normalized location
                    'profile_entropy': calculate_channel_entropy(profile, bins=32) # Entropy of the SEFA profile itself
                }

                # Ensure order matches feature_names_out_
                ordered_features = [features[fname.replace('sefa_prof_', '')] for fname in self.feature_names_out_]
                all_profile_features.append(np.array(ordered_features))

            except Exception as e:
                warnings.warn(f"Peak feature extraction failed for sample {i}: {e}. Returning zeros.", RuntimeWarning)
                all_profile_features.append(np.zeros(len(self.feature_names_out_)))

        return np.nan_to_num(np.array(all_profile_features), nan=0.0, posinf=0.0, neginf=0.0)


    def _set_output_feature_names(self):
        """Sets the names for the profile features."""
        self.feature_names_out_ = [
            'sefa_prof_num_peaks',
            'sefa_prof_mean_peak_height', 'sefa_prof_max_peak_height', 'sefa_prof_std_peak_height',
            'sefa_prof_mean_peak_prominence', 'sefa_prof_max_peak_prominence', 'sefa_prof_std_peak_prominence',
            'sefa_prof_mean_peak_width', 'sefa_prof_max_peak_width', 'sefa_prof_std_peak_width',
            'sefa_prof_highest_peak_loc',
            'sefa_prof_profile_entropy'
        ]

    def get_feature_names_out(self, input_features=None):
        """Get output feature names (SEFA profile features)."""
        check_is_fitted(self)
        if not hasattr(self, 'feature_names_out_') or not self.feature_names_out_:
             self._set_output_feature_names() # Ensure it's set
        return np.array(self.feature_names_out_, dtype=object)


def evaluate_classifier(model, X, y, test_size=0.3, seed=42):
    """
    Evaluates a scikit-learn compatible classifier on a given dataset.
    Handles pipelines for naming and checks for predict_proba/decision_function.
    Also prints the number of features used by the final classifier.
    """
    X, y = check_X_y(X, y, ensure_2d=True, dtype=np.float64, multi_output=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # --- Get feature count before fitting ---
    n_features_before_fit = X_train.shape[1]
    # If the model is a pipeline, try to determine the number of features
    # that will be fed into the final classifier step *after* transformation.
    # This requires fitting the transformer steps first.
    temp_model = copy.deepcopy(model)
    try:
        if isinstance(temp_model, Pipeline):
            # Fit only transformer steps
            X_transformed_train = X_train
            for name, step in temp_model.steps[:-1]:
                 if hasattr(step, 'transform'):
                      step.fit(X_transformed_train, y_train)
                      X_transformed_train = step.transform(X_transformed_train)
            n_features_final = X_transformed_train.shape[1]
        else:
             # Assume no transformation if not a pipeline
             n_features_final = n_features_before_fit
    except Exception as e:
        print(f"Warning: Could not determine final feature count before fitting full pipeline. Error: {e}")
        n_features_final = "Unknown"


    print(f"[Fitting model on training data (Input features: {n_features_before_fit}, Final features: {n_features_final})...]")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    auroc, ap = np.nan, np.nan
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
    except AttributeError:
        try:
            y_scores = model.decision_function(X_test)
            if y_scores.ndim > 1: y_scores = y_scores[:, 1] # Handle multi-class cases if needed
            auroc = roc_auc_score(y_test, y_scores)
            ap = average_precision_score(y_test, y_scores)
            warnings.warn("Model lacks predict_proba, using decision_function for AUROC/AP.", UserWarning)
        except AttributeError:
            warnings.warn("Model lacks predict_proba/decision_function. Cannot compute AUROC/AP.", UserWarning)
    except Exception as e:
        print(f"Warning: Could not calculate probability metrics (AUROC, AP). Error: {e}")

    prec, recall, f1 = np.nan, np.nan, np.nan
    try:
        prec, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    except ValueError as e:
         print(f"Warning: Could not calculate precision/recall/F1. Error: {e}")

    model_name = getattr(model, '__class__', type(model)).__name__
    final_estimator_name = 'UnknownEstimator'
    if isinstance(model, Pipeline):
        # More robustly get the name of the final estimator class
        final_estimator_name = getattr(type(model.steps[-1][1]), '__name__', 'UnknownEstimator')
        model_name = f"Pipeline->{final_estimator_name}"
        # If the first step is FeatureUnion, list the transformers inside it
        if isinstance(model.steps[0][1], FeatureUnion):
             union_transformers = [name for name, _ in model.steps[0][1].transformer_list]
             model_name = f"FeatureUnion({'+'.join(union_transformers)})->{final_estimator_name}"


    print(f"\n--- [Model Evaluation: {model_name}] ---")
    print(f"  Features into Classifier = {n_features_final}")
    print(f"  AUROC       = {auroc:.4f}")
    print(f"  AvgPrec     = {ap:.4f}")
    print(f"  Precision   = {prec:.4f}")
    print(f"  Recall      = {recall:.4f}")
    print(f"  F1 Score    = {f1:.4f}")
    print("------------------------------------\\n")

    return {'AUROC': auroc, 'AP': ap, 'Precision': prec, 'Recall': recall, 'F1': f1, 'n_features_final': n_features_final}


def generate_fwi_like_signal(n_points=512, noise_level=0.1, seed=None):
    """Generates a more complex base signal with slight random variations."""
    if seed is not None: np.random.seed(seed)
    t = np.linspace(0, 1, n_points)
    center1 = np.random.normal(0.2, 0.02); center2 = np.random.normal(0.5, 0.03); center3 = np.random.normal(0.75, 0.02); center4 = np.random.normal(0.35, 0.04)
    f0_1=np.random.normal(2,0.5); f1_1=np.random.normal(10,1); f0_2=np.random.normal(8,1); f1_2=np.random.normal(16,1.5); f0_3=np.random.normal(20,2); f1_3=np.random.normal(30,2); f0_4=np.random.normal(5,1); f1_4=np.random.normal(12,1)
    amp1=np.random.normal(1.0,0.1); amp2=np.random.normal(0.6,0.05); amp3=np.random.normal(0.3,0.05); amp4=np.random.normal(0.4,0.05)
    width1=np.random.normal(20,2); width2=np.random.normal(40,3); width3=np.random.normal(60,5); width4=np.random.normal(30,4)
    wave1 = chirp(t, f0=f0_1, f1=f1_1, t1=1, method='quadratic') * np.exp(-width1*(t - center1)**2)
    wave2 = chirp(t, f0=f0_2, f1=f1_2, t1=1, method='linear') * np.exp(-width2*(t - center2)**2)
    wave3 = chirp(t, f0=f0_3, f1=f1_3, t1=1, method='linear') * np.exp(-width3*(t - center3)**2)
    wave4 = chirp(t, f0=f0_4, f1=f1_4, t1=1, method='hyperbolic', vertex_zero=False) * np.exp(-width4*(t - center4)**2)
    signal = amp1*wave1 + amp2*wave2 + amp3*wave3 + amp4*wave4
    b, a = butter(4, [0.05, 0.4], btype='band')
    noise = filtfilt(b, a, np.random.randn(n_points)) * noise_level
    n_outliers = max(1, int(0.005 * n_points)); outlier_indices = np.random.choice(n_points, size=n_outliers, replace=False)
    noise[outlier_indices] += np.random.normal(0, 5 * noise_level, size=outlier_indices.shape)
    return signal + noise

def generate_fwi_dataset(n_samples=500, n_points=512, noise_level=0.1,
                         anomaly_frac=0.5, anomaly_magnitude=0.08,
                         anomaly_magnitude_std=0.03, anomaly_length_mean=6,
                         anomaly_length_std=2, seed=42):
    """Generates a highly complex dataset with diverse, subtle anomalies."""
    np.random.seed(seed)
    base_signals = np.array([generate_fwi_like_signal(n_points=n_points, noise_level=noise_level, seed=1000 + i)
                             for i in range(n_samples)])
    y = np.zeros(n_samples, dtype=int)
    X = base_signals.copy()
    n_anomalies = int(n_samples * anomaly_frac); anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False); y[anomaly_indices] = 1
    min_len = 2; anomaly_types = ['noise', 'phase_shift', 'freq_mod']
    skipped_count = 0

    for i in anomaly_indices:
        current_magnitude_factor = np.random.normal(anomaly_magnitude, anomaly_magnitude_std)
        current_length = max(min_len, int(np.random.normal(anomaly_length_mean, anomaly_length_std)))
        min_start_idx = n_points // 10; max_start_idx = 9 * n_points // 10 - current_length
        if max_start_idx <= min_start_idx:
            current_length = 9 * n_points // 10 - n_points // 10 - 1; max_start_idx = min_start_idx
            if current_length < min_len: skipped_count += 1; y[i] = 0; continue
        idx = np.random.randint(min_start_idx, max_start_idx)
        anomaly_type = np.random.choice(anomaly_types)
        try:
            segment = X[i, idx : idx + current_length]
            if anomaly_type == 'noise':
                local_std = np.std(segment) + 1e-6; scale = abs(current_magnitude_factor * local_std)
                anomaly_shape = np.random.normal(0, scale, size=current_length)
                X[i, idx : idx + current_length] += anomaly_shape
            elif anomaly_type == 'phase_shift' and current_length > 1:
                analytic = hilbert(segment); phase = np.angle(analytic); shift_amount = np.random.uniform(-np.pi/4, np.pi/4) * current_magnitude_factor / anomaly_magnitude
                taper = np.sin(np.linspace(0, np.pi, current_length))**2; shifted_phase = np.unwrap(phase + shift_amount * taper)
                X[i, idx : idx + current_length] = np.abs(analytic) * np.cos(shifted_phase)
            elif anomaly_type == 'freq_mod' and current_length > 1:
                analytic = hilbert(segment); phase = np.unwrap(np.angle(analytic)); t_local = np.linspace(0, 1, current_length)
                freq_mod_strength = current_magnitude_factor * 5; modulation = freq_mod_strength * np.sin(2 * np.pi * t_local * np.random.uniform(0.5, 2))
                modulated_phase = phase + np.cumsum(modulation) * (t_local[1]-t_local[0])
                X[i, idx : idx + current_length] = np.abs(analytic) * np.cos(modulated_phase)
        except Exception as e:
            warnings.warn(f"Anomaly generation failed type='{anomaly_type}', sample={i}: {e}", RuntimeWarning)
            skipped_count += 1; y[i] = 0; continue

    valid_indices = np.where(y != -999)[0] # Use -999 or similar if needed, 0 works if we only set 1s
    X = X[valid_indices]; y = y[valid_indices]
    if len(np.unique(y)) < 2: raise ValueError("Dataset generation resulted in only one class.")
    print(f"Generated dataset with {len(X)} samples. Anomaly count: {np.sum(y==1)}. Skipped: {skipped_count}")
    return X, y


if __name__ == "__main__":
    N_SAMPLES = 500
    N_POINTS = 256
    NOISE = 0.15
    RANDOM_SEED = 42

    print("[Generating highly complex synthetic FWI-like dataset...]")
    X_fwi, y_fwi = generate_fwi_dataset(n_samples=N_SAMPLES, n_points=N_POINTS,
                                        noise_level=NOISE, anomaly_frac=0.5,
                                        anomaly_magnitude=0.05, anomaly_magnitude_std=0.02,
                                        anomaly_length_mean=5, anomaly_length_std=1.5,
                                        seed=RANDOM_SEED)

    print("[Scaling data...]")
    # Scaling is now handled within the pipeline if needed, raw data is used.
    # scaler = StandardScaler()
    # X_fwi_std = scaler.fit_transform(X_fwi)
    # Use X_fwi directly now.

    print("[Initializing models...]")

    # --- SEFA Ensemble Pipeline Components ---
    # Shared extractor parameters
    extractor_params = dict(
        entropy_bins=32,
        entropy_window_ratio=0.03,
        use_savgol=False,
        n_jobs=-1
    )
    # Shared calibrator parameters
    calibrator_params = dict(
        entropy_bins=extractor_params['entropy_bins'], # Ensure consistency
        epsilon=1e-16
    )
    # Shared aggregator parameters
    aggregator_params = dict(
        sefa_score_aggregations=['mean', 'std', 'max', 'min', 'median', 'ptp']
    )
    # Shared profiler parameters
    profiler_params = dict(
        peak_height_rel=0.05, # Adjusted threshold
        peak_prominence_rel=0.02, # Adjusted threshold
        peak_min_dist_frac=0.01
    )

    # --- Pipeline Path A: Aggregated SEFA Scores ---
    sefa_agg_pipeline = Pipeline([
        ('extractor', GeneralizedSpectralFeatureExtractor(**extractor_params)),
        ('calibrator', SEFACalibrator(**calibrator_params)),
        ('aggregator', SEFAScoreAggregator(**aggregator_params))
    ])

    # --- Pipeline Path B: SEFA Profile Features ---
    sefa_profile_pipeline = Pipeline([
        ('extractor', GeneralizedSpectralFeatureExtractor(**extractor_params)),
        ('calibrator', SEFACalibrator(**calibrator_params)),
        ('profiler', SEFALocalProfileFeatureExtractor(**profiler_params))
    ])

    # --- Feature Union: Combine Aggregated and Profile Features ---
    # Note: n_jobs in FeatureUnion parallelizes the pipelines within it
    combined_sefa_features = FeatureUnion([
        ('sefa_agg', sefa_agg_pipeline),
        ('sefa_profile', sefa_profile_pipeline)
    ], n_jobs=1) # Set n_jobs=1 for FeatureUnion to avoid nested parallelism issues initially

    # --- Final SEFA Ensemble Pipeline ---
    sefa_ensemble_pipeline = Pipeline([
        ('features', combined_sefa_features),
        # Optional: Add StandardScaler here if features have very different scales
        ('scaler', StandardScaler()),
        ('classifier', MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), # Increased capacity for more features
            max_iter=1500,
            random_state=RANDOM_SEED,
            activation='relu',
            solver='adam',
            alpha=0.0001,
            learning_rate_init=0.001, # Default
            early_stopping=True,      # Helps prevent overfitting
            n_iter_no_change=20       # Stop if validation score doesn't improve
        ))
    ])


    # --- Baseline Pipeline (Logistic Regression on Raw Scaled Data) ---
    lr_baseline_model = Pipeline([
        ('scaler', StandardScaler()), # Scale raw data for LR
        ('logistic_regression', LogisticRegression(random_state=RANDOM_SEED, max_iter=1000))
    ])

    # --- Evaluation ---
    print("\n[Evaluating SEFA Ensemble Pipeline (FeatureUnion + MLPClassifier)...]")
    # Pass raw X_fwi, scaling is handled inside FeatureUnion/final pipeline if added
    metrics_SEFA_ensemble = evaluate_classifier(sefa_ensemble_pipeline, X_fwi, y_fwi, seed=RANDOM_SEED)

    print("\n[Evaluating Baseline (StandardScaler + Logistic Regression)...]")
    metrics_lr = evaluate_classifier(lr_baseline_model, X_fwi, y_fwi, seed=RANDOM_SEED)

    # --- Example: Accessing Feature Names ---
    print("\n[Feature Names from SEFA Ensemble Pipeline]")
    # Fit the pipeline first if not done during evaluation (it is)
    if not hasattr(sefa_ensemble_pipeline.named_steps['classifier'], 'classes_'):
         print("(Pipeline not fitted, fitting now to get feature names...)")
         sefa_ensemble_pipeline.fit(X_fwi, y_fwi)

    # Access feature names from the FeatureUnion step
    try:
        # Scikit-learn >= 1.0 automatically prefixes names from FeatureUnion
        feature_names = sefa_ensemble_pipeline.named_steps['features'].get_feature_names_out()
        print(f"Total number of combined output features: {len(feature_names)}")
        # Print first/last few names for brevity
        print("Sample Feature Names:")
        print(feature_names[:5])
        print("...")
        print(feature_names[-5:])
    except Exception as e:
        print(f"Could not get feature names from SEFA ensemble pipeline: {e}")
