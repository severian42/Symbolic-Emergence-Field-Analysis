# lore_scientific_standalone.py
# Standalone Script for LORE Algorithm Demonstration with In-Script Configuration

import numpy as np
import pandas as pd
from scipy.signal import hilbert, argrelextrema
from scipy.stats import entropy as shannon_entropy
from scipy.special import rel_entr
from sklearn.metrics import mutual_info_score
from sympy import primerange
import matplotlib.pyplot as plt
import os
import time
import json # For saving parameters
import pywt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sympy import primepi
# Robust import for logarithmic integral li(x)
try:
    from sympy import li
except ImportError:
    # Fallback: use scipy.special.expi for li(x) ≈ expi(log(x)) for x > 0
    from scipy.special import expi
    def li(x):
        import numpy as np
        x = np.asarray(x)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = expi(np.log(x))
            result[x <= 0] = 0.0
        return result

# Add a config flag at the top for self-calibrating mode
SELF_CALIBRATING_LORE = True  # Set to False to use original manual weights

# --------------------------
# Utility Functions
# --------------------------

def load_zeta_zeros_from_txt(path, num_zeros_to_load=None):
    """Loads zeta zeros from a text file (robustly handles formats)."""
    zeros = []
    try:
        with open(path, 'r') as f:
            count = 0
            for line in f:
                # Stop reading if we've loaded enough zeros (efficiency)
                if num_zeros_to_load is not None and count >= num_zeros_to_load:
                    break
                parts = line.strip().split()
                for part in parts:
                    try:
                        zero_val = float(part)
                        # Ensure only positive imaginary parts > epsilon are loaded
                        if zero_val > 1e-9:
                            zeros.append(zero_val)
                            count += 1
                            if num_zeros_to_load is not None and count >= num_zeros_to_load:
                                break # Break inner loop too
                    except ValueError:
                        continue # Skip non-float parts
                if num_zeros_to_load is not None and count >= num_zeros_to_load:
                     break # Break outer loop
    except FileNotFoundError:
        print(f"[ERROR] Zeta zero file not found: {path}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to read zeta zero file {path}: {e}")
        return None

    if not zeros:
        print(f"[WARN] No valid positive zeta zeros found in file: {path}")
        return np.array([])

    # If num_zeros_to_load was specified, slice the list to the exact count
    if num_zeros_to_load is not None and len(zeros) > num_zeros_to_load:
        zeros = zeros[:num_zeros_to_load]

    print(f"Loaded {len(zeros)} zeta zeros from {path}.")
    return np.array(zeros)

def shuffle_gamma(gamma_values, seed=42):
    rng = np.random.default_rng(seed)
    return rng.permutation(gamma_values)

def generate_gue_gamma(num, mean_spacing=1.0, seed=42):
    # GUE spacings: Wigner surmise for level spacings
    rng = np.random.default_rng(seed)
    spacings = rng.wald(mean_spacing, 1.0, size=num)
    return np.cumsum(spacings)

def synthetic_target_hamming_weight(x_min, x_max, weight=5):
    # Return all integers in [x_min, x_max] with exactly 'weight' 1-bits in binary
    return [n for n in range(x_min, x_max+1) if bin(n).count('1') == weight]

def compute_auroc_ap(score, N_grid, N_min, N_max, target_set):
    # For each integer in [N_min, N_max], get its score and label (1 if in target_set)
    N_ints = np.arange(N_min, N_max+1)
    y_ints = np.log(N_ints)
    score_interp = np.interp(N_ints, N_grid, score)
    labels = np.array([1 if n in target_set else 0 for n in N_ints])
    from sklearn.metrics import roc_auc_score, average_precision_score
    try:
        auroc = roc_auc_score(labels, score_interp)
        ap = average_precision_score(labels, score_interp)
    except Exception:
        auroc, ap = float('nan'), float('nan')
    return auroc, ap

def tukey_window_pad(signal, alpha=0.5):
    # Apply a Tukey window to the signal
    from scipy.signal import tukey
    window = tukey(len(signal), alpha=alpha)
    return signal * window

def mirror_pad(signal, pad_width):
    # Mirror-pad the signal at both ends
    return np.pad(signal, pad_width=pad_width, mode='reflect')

def wavelet_multiscale_score(score, wavelet='db1', maxlevel=4, agg='max'):
    # Perform wavelet packet decomposition and aggregate scores across scales
    wp = pywt.WaveletPacket(data=score, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    nodes = [n.path for n in wp.get_level(maxlevel, 'natural')]
    coeffs = np.array([wp[n].data for n in nodes])  # shape: (num_nodes, len_per_node)
    # Upsample each node's coeffs to full length
    upsampled = []
    for c in coeffs:
        up = np.repeat(c, len(score)//len(c))
        if len(up) < len(score):
            up = np.pad(up, (0, len(score)-len(up)), mode='edge')
        upsampled.append(up)
    upsampled = np.array(upsampled)
    if agg == 'max':
        agg_score = np.max(upsampled, axis=0)
    elif agg == 'mean':
        agg_score = np.mean(upsampled, axis=0)
    else:
        agg_score = np.max(upsampled, axis=0)
    return agg_score

def otsu_threshold(values, nbins=256):
    """Compute Otsu's threshold for a 1D array."""
    hist, bin_edges = np.histogram(values, bins=nbins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_centers) / (weight1 + 1e-12)
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / (weight2[::-1] + 1e-12))[::-1]
    inter_class_var = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(inter_class_var)
    return bin_centers[idx]

# --------------------------
# Core LORE Functions
# --------------------------

def compute_forcing_field(gamma_values, y_grid, taper='lorentz', gamma_cutoff=None):
    """Computes the spectral forcing field V0(y) from zeta zeros.
       taper: 'lorentz' (1/(1+gamma^2)) or 'gaussian' (exp(-(gamma/Gamma)^2))
       gamma_cutoff: Gamma for gaussian taper (if None, set to Nyquist freq)
    """
    V0 = np.zeros_like(y_grid, dtype=np.float64)
    dy = y_grid[1] - y_grid[0]
    if taper == 'gaussian':
        if gamma_cutoff is None:
            gamma_cutoff = np.pi / dy  # Nyquist freq for the grid
        weights = np.exp(-(gamma_values / gamma_cutoff) ** 2)
    else:
        weights = 1.0 / (1.0 + gamma_values ** 2)
    for g, w in zip(gamma_values, weights):
        V0 += w * np.cos(g * y_grid)
    return V0

def compute_local_entropy(signal, window_size=25, num_bins=20):
    """Computes local Shannon entropy in a sliding window using natural log."""
    entropies = np.zeros_like(signal, dtype=np.float64)
    half_w = window_size // 2
    signal_len = len(signal)
    padded_signal = np.pad(signal, pad_width=half_w, mode='edge')
    for i in range(signal_len):
        window = padded_signal[i : i + window_size]
        hist, _ = np.histogram(window, bins=num_bins, density=True)
        entropies[i] = shannon_entropy(hist[hist > 1e-12])
    return entropies

def calculate_lore_fields(y_grid, V0_y, local_entropy_window, entropy_bins):
    """Calculates derived fields and returns normalized scores."""
    print("  Calculating Hilbert transform...")
    analytic_signal = hilbert(V0_y)
    print("  Calculating Envelope Amplitude...")
    amplitude_envelope = np.abs(analytic_signal)
    print("  Calculating Instantaneous Phase and Frequency...")
    dy = y_grid[1] - y_grid[0]
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.gradient(instantaneous_phase, dy)
    print("  Calculating Envelope Curvature...")
    curvature = np.gradient(np.gradient(amplitude_envelope, dy), dy)
    print("  Calculating Local Energy Density...")
    local_energy = amplitude_envelope ** 2
    print("  Normalizing fields...")
    max_amp = np.max(amplitude_envelope)
    envelope_score = amplitude_envelope / max_amp if max_amp > 1e-12 else np.zeros_like(amplitude_envelope)
    max_abs_curv = np.max(np.abs(curvature))
    curvature_score = np.abs(curvature) / max_abs_curv if max_abs_curv > 1e-12 else np.zeros_like(curvature)
    max_abs_freq = np.max(np.abs(instantaneous_frequency))
    frequency_score = np.abs(instantaneous_frequency) / max_abs_freq if max_abs_freq > 1e-12 else np.zeros_like(instantaneous_frequency)
    max_energy = np.max(local_energy)
    energy_score = local_energy / max_energy if max_energy > 1e-12 else np.zeros_like(local_energy)
    print("  Calculating Local Entropy...")
    entropy_raw = compute_local_entropy(envelope_score, window_size=local_entropy_window, num_bins=entropy_bins)
    max_entropy = np.max(entropy_raw)
    entropy_alignment_score = 1.0 - (entropy_raw / max_entropy) if max_entropy > 1e-12 else np.ones_like(entropy_raw)
    return envelope_score, curvature_score, frequency_score, entropy_alignment_score, energy_score

def compute_symbolic_score(fields, weights):
    """Computes the final LORE symbolic score using weighted geometric mean."""
    envelope_score, curvature_score, frequency_score, entropy_score, energy_score = fields
    alpha = weights.get('envelope', 0.6)
    beta = weights.get('curvature', 0.8)
    gamma = weights.get('frequency', 0.6)
    delta = weights.get('entropy', 1.2)
    eta = weights.get('energy', 0.0)  # Default: not used unless set
    epsilon = 1e-12
    symbolic_score = (
        (envelope_score + epsilon) ** alpha *
        (curvature_score + epsilon) ** beta *
        (frequency_score + epsilon) ** gamma *
        (entropy_score + epsilon) ** delta *
        (energy_score + epsilon) ** eta
    )
    symbolic_score = np.clip(symbolic_score, 0, 1.0)
    return symbolic_score

def compute_selfcal_lore_score(envelope_score, curvature_score, frequency_score, y_grid):
    """Compute the self-calibrating LORE score and exponents."""
    eps = 1e-12
    def norm01(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x) + eps)
    A_ = norm01(envelope_score)
    C_ = norm01(curvature_score)
    F_ = norm01(frequency_score)
    dy = y_grid[1] - y_grid[0]
    W_min = int(np.ceil(dy**-0.5))
    W_max = int(0.20 * len(y_grid))
    candidate_windows = np.unique(np.logspace(np.log10(W_min), np.log10(W_max), num=10, dtype=int))
    var_S = []
    for W in candidate_windows:
        S = compute_local_entropy(A_, window_size=W, num_bins=64)
        var_S.append(np.var(S))
    W_opt = candidate_windows[0]
    for i in range(1, len(var_S)-1):
        if var_S[i] > var_S[i-1] and var_S[i] > var_S[i+1]:
            W_opt = candidate_windows[i]
            break
    S = compute_local_entropy(A_, window_size=W_opt, num_bins=64)
    E = 1 - S / (np.max(S) + eps)
    def channel_entropy(X):
        hist, _ = np.histogram(X, bins=64, density=True)
        p = hist / (hist.sum() + eps)
        return -np.sum(p[p > 0] * np.log(p[p > 0]))
    H_max = np.log(64)
    I_A = channel_entropy(A_)
    I_C = channel_entropy(C_)
    I_F = channel_entropy(F_)
    I_E = channel_entropy(E)
    w_A = H_max - I_A
    w_C = H_max - I_C
    w_F = H_max - I_F
    w_E = H_max - I_E
    w_sum = w_A + w_C + w_F + w_E
    wA_, wC_, wF_, wE_ = w_A/w_sum, w_C/w_sum, w_F/w_sum, w_E/w_sum
    alpha, beta, gamma_exp, delta = 4*wA_, 4*wC_, 4*wF_, 4*wE_
    symbolic_score = (A_+eps)**alpha * (C_+eps)**beta * (F_+eps)**gamma_exp * (E+eps)**delta
    symbolic_score = symbolic_score / (np.max(symbolic_score) + eps)
    details = dict(entropy_window=W_opt, exponents=dict(alpha=alpha, beta=beta, gamma=gamma_exp, delta=delta))
    return symbolic_score, details

# --------------------------
# Evaluation Functions
# --------------------------

def evaluate_predictions(predicted_N, true_primes, tolerance=1):
    """Evaluates predicted integers against true prime integers with a tolerance."""
    predicted_N_unique = np.unique(np.array(predicted_N, dtype=int))
    true_primes_set = set(true_primes)
    true_primes_unique = sorted(list(true_primes_set))
    is_hit_map = {p: False for p in predicted_N_unique}
    prime_hit_map = {t: False for t in true_primes_unique}
    for p in predicted_N_unique:
        for delta in range(-tolerance, tolerance + 1):
            check_val = p + delta
            if check_val in true_primes_set:
                is_hit_map[p] = True
                prime_hit_map[check_val] = True
                break
    tp_list = sorted([p for p, is_hit in is_hit_map.items() if is_hit])
    fp_list = sorted([p for p, is_hit in is_hit_map.items() if not is_hit])
    fn_list = sorted([t for t, was_hit in prime_hit_map.items() if not was_hit])
    num_distinct_primes_hit = sum(prime_hit_map.values())
    num_pred = len(predicted_N_unique)
    num_true = len(true_primes_unique)
    num_tp = len(tp_list)
    num_fp = len(fp_list)
    num_fn = len(fn_list)
    precision = num_tp / num_pred if num_pred > 0 else 0
    recall = num_distinct_primes_hit / num_true if num_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "Total Unique Predictions": num_pred, "Total True Primes in Range": num_true,
        "True Positives (Hits)": num_tp, "False Positives (Misses)": num_fp,
        "False Negatives (Primes Missed)": num_fn, "Distinct Primes Hit": num_distinct_primes_hit,
        "Precision": precision, "Recall": recall, "F1 Score": f1,
        "TP List": tp_list, "FP List": fp_list, "FN List": fn_list
    }

def random_baseline(num_preds, x_min, x_max, true_values, tolerance=1):
    """Generates random predictions for baseline comparison."""
    available_range = x_max - x_min + 1
    if available_range < num_preds:
         print(f"[WARN] Cannot select {num_preds} unique random numbers from range [{x_min}, {x_max}]. Adjusting to {available_range}.")
         num_preds = available_range
    if num_preds <= 0: return {'Precision': 0, 'Recall': 0, 'F1 Score': 0}
    random_preds = np.random.choice(np.arange(x_min, x_max + 1), size=num_preds, replace=False)
    eval_dict = evaluate_predictions(random_preds, true_values, tolerance=tolerance)
    return {'Precision': eval_dict['Precision'], 'Recall': eval_dict['Recall'], 'F1 Score': eval_dict['F1 Score']}

# === New Baseline: Odds Only (except 2) ===
def odds_baseline(x_min, x_max, true_primes, tolerance=1):
    odds = [n for n in range(x_min, x_max + 1) if n % 2 == 1 or n == 2]
    eval_dict = evaluate_predictions(odds, true_primes, tolerance)
    return {'Precision': eval_dict['Precision'], 'Recall': eval_dict['Recall'], 'F1 Score': eval_dict['F1 Score']}

# === New Baseline: Von Mangoldt (cosine sum, no Hilbert/entropy) ===
def von_mangoldt_baseline(gamma_values, y_grid, N_grid, N_min, N_max, true_primes, k=1000, tolerance=1):
    # Standard explicit formula: sum_k cos(gamma_k * y)
    V_vm = np.sum([np.cos(g * y_grid) for g in gamma_values], axis=0)
    # Find peaks
    peak_indices = argrelextrema(V_vm, np.greater)[0]
    if len(peak_indices) > 0:
        peak_scores = V_vm[peak_indices]
        sorted_peak_indices = peak_indices[np.argsort(peak_scores)[::-1]]
        top_indices = sorted_peak_indices[:min(k, len(sorted_peak_indices))]
        predicted_N = np.round(N_grid[top_indices]).astype(int)
        predicted_N = np.unique(predicted_N[(predicted_N >= N_min) & (predicted_N <= N_max)])
    else:
        predicted_N = np.array([], dtype=int)
    eval_dict = evaluate_predictions(predicted_N, true_primes, tolerance)
    return {'Precision': eval_dict['Precision'], 'Recall': eval_dict['Recall'], 'F1 Score': eval_dict['F1 Score'], 'NumPred': len(predicted_N)}

# === Precision-vs-k Curve for LORE ===
def precision_recall_vs_k(symbolic_score, N_grid, N_min, N_max, N_primes, k_min=10, k_max=2000, step=10, tolerance=1):
    from scipy.signal import argrelextrema
    peak_indices = argrelextrema(symbolic_score, np.greater)[0]
    peak_scores = symbolic_score[peak_indices]
    sorted_peak_indices = peak_indices[np.argsort(peak_scores)[::-1]]
    ks = np.arange(k_min, k_max + 1, step)
    precisions = []
    recalls = []
    for k in ks:
        top_indices = sorted_peak_indices[:min(k, len(sorted_peak_indices))]
        predicted_N = np.round(N_grid[top_indices]).astype(int)
        predicted_N = np.unique(predicted_N[(predicted_N >= N_min) & (predicted_N <= N_max)])
        eval_dict = evaluate_predictions(predicted_N, N_primes, tolerance)
        precisions.append(eval_dict['Precision'])
        recalls.append(eval_dict['Recall'])
    return ks, precisions, recalls

# === Plotting Precision-vs-k ===
def plot_precision_vs_k(ks, precisions, recalls, out_path):
    plt.figure(figsize=(10,6))
    plt.plot(ks, precisions, label='Precision', color='blue')
    plt.plot(ks, recalls, label='Recall', color='orange', linestyle='--')
    plt.xlabel('Top k Peaks')
    plt.ylabel('Score')
    plt.title('LORE Precision and Recall vs Top-k Peaks')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# --------------------------
# Plotting Functions
# --------------------------

def plot_lore_score(N_grid, lore_score, N_primes, predicted_N_hits, predicted_N_misses, out_path):
    """Plots the LORE score vs N, highlighting primes and predictions."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(N_grid, lore_score, label='LORE Score', color='dodgerblue', linewidth=1.0, zorder=1)
    if N_primes:
        lore_at_primes = np.interp(N_primes, N_grid, lore_score)
        ax.scatter(N_primes, lore_at_primes, color='red', s=15, zorder=2, label=f'True Primes ({len(N_primes)})')
    if predicted_N_hits:
        lore_at_hits = np.interp(predicted_N_hits, N_grid, lore_score)
        ax.scatter(predicted_N_hits, lore_at_hits, color='lime', marker='o', s=40, zorder=3, label=f'True Positives ({len(predicted_N_hits)})', facecolors='none', edgecolors='lime', linewidths=1.5)
    if predicted_N_misses:
        lore_at_misses = np.interp(predicted_N_misses, N_grid, lore_score)
        ax.scatter(predicted_N_misses, lore_at_misses, color='magenta', marker='x', s=30, zorder=3, label=f'False Positives ({len(predicted_N_misses)})', linewidths=1.5)
    ax.set_xlabel("N (Integer Domain)")
    ax.set_ylabel("LORE Score (Normalized)")
    ax.set_title("LORE Symbolic Score vs N with Prime Validation")
    ax.legend(fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(N_grid.min(), N_grid.max())
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=150)
        print(f"  LORE score plot saved to {out_path}")
    except Exception as e: print(f"[ERROR] Failed to save LORE score plot: {e}")
    plt.close(fig)

def plot_prediction_distribution(true_positives, false_positives, bins, x_range, out_path):
    """Plots histogram of true and false positives across N."""
    fig, ax = plt.subplots(figsize=(12, 5))
    if true_positives:
        ax.hist(true_positives, bins=bins, range=x_range, alpha=0.7, label=f'True Positives ({len(true_positives)})', color='forestgreen')
    if false_positives:
        ax.hist(false_positives, bins=bins, range=x_range, alpha=0.7, label=f'False Positives ({len(false_positives)})', color='crimson')
    if not true_positives and not false_positives:
         ax.text(0.5, 0.5, 'No predictions to plot', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.set_title("Distribution of LORE Predictions Across Integer Domain")
    ax.set_xlabel("N (Integer Domain)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.set_xlim(x_range)
    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=150)
        print(f"  Prediction distribution plot saved to {out_path}")
    except Exception as e: print(f"[ERROR] Failed to save prediction distribution plot: {e}")
    plt.close(fig)

def plot_scientific_visuals(
    output_dir,
    N_grid,
    V0_y,
    envelope_score,
    curvature_score,
    frequency_score,
    entropy_alignment_score,
    energy_score,
    symbolic_score,
    N_primes,
    peak_indices,
    auroc,
    ap,
    rand_metrics,
    odds_metrics,
    vonmangoldt_metrics,
    auroc_shuf,
    ap_shuf,
    auroc_gue,
    ap_gue,
    kl_div_nats,
    mi_score,
    symbolic_score_shuf=None,
    symbolic_score_gue=None,
    train_test_results=None
):
    # 1. Forcing Field V0(y)
    plt.figure(figsize=(12,5))
    plt.plot(N_grid, V0_y, color='slateblue')
    plt.title('Forcing Field $V_0(y)$ vs N')
    plt.xlabel('N')
    plt.ylabel('$V_0(y)$')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'forcing_field_V0y.png'), dpi=150)
    plt.close()

    # 2. LORE Fields
    fields = [
        (envelope_score, 'Envelope', 'envelope_score'),
        (curvature_score, 'Curvature', 'curvature_score'),
        (frequency_score, 'Frequency', 'frequency_score'),
        (entropy_alignment_score, 'Entropy Alignment', 'entropy_score'),
        (energy_score, 'Energy', 'energy_score'),
    ]
    for arr, label, fname in fields:
        plt.figure(figsize=(12,5))
        plt.plot(N_grid, arr, label=label, color='dodgerblue')
        if N_primes:
            plt.scatter(N_primes, np.interp(N_primes, N_grid, arr), color='red', s=10, label='Primes', alpha=0.7)
        plt.title(f'{label} Field vs N')
        plt.xlabel('N')
        plt.ylabel(label)
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{fname}_field_vs_N.png'), dpi=150)
        plt.close()

    # 3. Histogram: LORE score at primes vs non-primes
    N_ints = np.arange(int(N_grid[0]), int(N_grid[-1])+1)
    lore_interp = np.interp(N_ints, N_grid, symbolic_score)
    is_prime = np.isin(N_ints, N_primes)
    plt.figure(figsize=(10,5))
    plt.hist(lore_interp[is_prime], bins=40, alpha=0.7, label='Primes', color='red', density=True)
    plt.hist(lore_interp[~is_prime], bins=40, alpha=0.5, label='Non-Primes', color='gray', density=True)
    plt.xlabel('LORE Score')
    plt.ylabel('Density')
    plt.title('Distribution of LORE Score at Primes vs Non-Primes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lore_score_hist_primes_vs_nonprimes.png'), dpi=150)
    plt.close()

    # 4. Histogram: Peak Scores
    if peak_indices is not None and len(peak_indices) > 0:
        plt.figure(figsize=(10,5))
        plt.hist(symbolic_score[peak_indices], bins=40, color='purple', alpha=0.7)
        plt.xlabel('LORE Score at Peaks')
        plt.ylabel('Count')
        plt.title('Distribution of LORE Scores at Detected Peaks')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lore_peak_score_histogram.png'), dpi=150)
        plt.close()

    # 5. ROC and PR Curves for LORE and baselines
    def plot_roc_pr(y_true, y_score, label, color, out_prefix):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(7,5))
        plt.plot(fpr, tpr, color=color, label=f'{label} (AUROC={roc_auc:.3f})')
        plt.plot([0,1],[0,1],'k--',alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{out_prefix}_roc_curve.png'), dpi=150)
        plt.close()
        plt.figure(figsize=(7,5))
        plt.plot(recall, precision, color=color, label=f'{label} (AP={pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{out_prefix}_pr_curve.png'), dpi=150)
        plt.close()
    # True/false labels for all N
    N_ints = np.arange(int(N_grid[0]), int(N_grid[-1])+1)
    y_true = np.isin(N_ints, N_primes).astype(int)
    # LORE
    lore_interp = np.interp(N_ints, N_grid, symbolic_score)
    plot_roc_pr(y_true, lore_interp, 'LORE', 'blue', 'lore')
    # Baselines (random, odds, von Mangoldt)
    # For random, odds, von Mangoldt, use their F1/precision/recall from metrics, but skip curves if not available
    # 6. Overlay: LORE score for true, shuffled, GUE gamma
    plt.figure(figsize=(12,6))
    plt.plot(N_grid, symbolic_score, label='LORE (True Gamma)', color='blue')
    if symbolic_score_shuf is not None:
        plt.plot(N_grid, symbolic_score_shuf, label='LORE (Shuffled Gamma)', color='orange', alpha=0.7)
    if symbolic_score_gue is not None:
        plt.plot(N_grid, symbolic_score_gue, label='LORE (GUE Gamma)', color='green', alpha=0.7)
    plt.xlabel('N')
    plt.ylabel('LORE Score')
    plt.title('LORE Score: True vs Shuffled vs GUE Gamma')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lore_score_overlay_controls.png'), dpi=150)
    plt.close()

    # 7. Bar plot: KL Divergence & MI
    plt.figure(figsize=(8,5))
    bars = ['LORE']
    kl_vals = [kl_div_nats]
    mi_vals = [mi_score]
    if auroc_shuf is not None:
        bars.append('Shuffled')
        kl_vals.append(np.nan)
        mi_vals.append(np.nan)
    if auroc_gue is not None:
        bars.append('GUE')
        kl_vals.append(np.nan)
        mi_vals.append(np.nan)
    plt.bar(np.arange(len(bars))-0.15, kl_vals, width=0.3, label='KL Div (nats)')
    plt.bar(np.arange(len(bars))+0.15, mi_vals, width=0.3, label='Mutual Info')
    plt.xticks(np.arange(len(bars)), bars)
    plt.ylabel('Value')
    plt.title('KL Divergence and Mutual Information')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kl_mi_barplot.png'), dpi=150)
    plt.close()

    # 8. Train/Test split visuals
    if train_test_results:
        for label, rng_min, rng_max, auroc_sub, ap_sub, nprimes in train_test_results:
            plt.figure(figsize=(10,4))
            plt.bar(['AUROC','AP'], [auroc_sub, ap_sub], color=['blue','orange'])
            plt.title(f'{label}: N=[{rng_min},{rng_max}], Primes={nprimes}')
            plt.ylim(0,1)
            plt.tight_layout()
            fname = f'train_test_{label.replace(" ","").replace("(","").replace(")","").replace("-","").lower()}.png'
            plt.savefig(os.path.join(output_dir, fname), dpi=150)
            plt.close()

    # 9. LORE score + von Mangoldt + primes overlay
    # Compute von Mangoldt field (cosine sum, normalized)
    V_vm = np.sum([np.cos(g * np.log(N_grid)) for g in np.linspace(1, 1000, 1000)], axis=0)
    V_vm = (V_vm - V_vm.min()) / (V_vm.max() - V_vm.min() + 1e-12)
    plt.figure(figsize=(14,6))
    plt.plot(N_grid, symbolic_score, label='LORE Score', color='blue', alpha=0.8)
    plt.plot(N_grid, V_vm, label='Von Mangoldt (cos sum, norm)', color='orange', alpha=0.6)
    if N_primes:
        plt.scatter(N_primes, np.interp(N_primes, N_grid, symbolic_score), color='red', s=10, label='Primes', alpha=0.7)
    plt.xlabel('N')
    plt.ylabel('Score')
    plt.title('LORE Score vs Von Mangoldt and Primes')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lore_vs_vonmangoldt_primes_overlay.png'), dpi=150)
    plt.close()

    # 10. Prime gap distribution vs predicted gap distribution
    if N_primes is not None and len(N_primes) > 2 and peak_indices is not None and len(peak_indices) > 2:
        prime_gaps = np.diff(sorted(N_primes))
        predicted_N = np.round(N_grid[peak_indices]).astype(int)
        predicted_N = np.unique(predicted_N[(predicted_N >= N_grid[0]) & (predicted_N <= N_grid[-1])])
        pred_gaps = np.diff(np.sort(predicted_N))
        plt.figure(figsize=(10,5))
        plt.hist(prime_gaps, bins=30, alpha=0.7, label='Prime Gaps', color='red', density=True)
        plt.hist(pred_gaps, bins=30, alpha=0.5, label='Predicted Gaps', color='blue', density=True)
        plt.xlabel('Gap')
        plt.ylabel('Density')
        plt.title('Prime Gap vs Predicted Gap Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prime_vs_predicted_gap_distribution.png'), dpi=150)
        plt.close()

    # 11. Cumulative count (π_LORE(N) vs π(N) vs R(N))
    N_ints = np.arange(int(N_grid[0]), int(N_grid[-1])+1)
    lore_interp = np.interp(N_ints, N_grid, symbolic_score)
    # π_LORE(N): cumulative count of predicted peaks up to N
    if peak_indices is not None and len(peak_indices) > 0:
        predicted_N = np.round(N_grid[peak_indices]).astype(int)
        predicted_N = np.unique(predicted_N[(predicted_N >= N_grid[0]) & (predicted_N <= N_grid[-1])])
        pi_lore = np.array([np.sum(predicted_N <= n) for n in N_ints])
    else:
        pi_lore = np.zeros_like(N_ints)
    pi_true = np.array([primepi(n) for n in N_ints])
    try:
        pi_li = np.array([li(n) for n in N_ints])
    except Exception:
        pi_li = np.zeros_like(N_ints)
    plt.figure(figsize=(12,6))
    plt.plot(N_ints, pi_true, label='π(N) (True)', color='red')
    plt.plot(N_ints, pi_lore, label='π_LORE(N) (Predicted)', color='blue')
    plt.plot(N_ints, pi_li, label='Logarithmic Integral li(N)', color='green', linestyle='--')
    plt.xlabel('N')
    plt.ylabel('Cumulative Count')
    plt.title('Prime Counting: True vs LORE vs Logarithmic Integral')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prime_counting_comparison.png'), dpi=150)
    plt.close()

    # 12. 2D heatmap of (N, LORE score) with primes
    from matplotlib.colors import LogNorm
    plt.figure(figsize=(14,5))
    plt.hexbin(N_grid, symbolic_score, gridsize=200, cmap='Blues', bins='log', mincnt=1)
    if N_primes:
        plt.scatter(N_primes, np.interp(N_primes, N_grid, symbolic_score), color='red', s=8, label='Primes', alpha=0.7)
    plt.xlabel('N')
    plt.ylabel('LORE Score')
    plt.title('2D Heatmap: N vs LORE Score (Primes Highlighted)')
    plt.colorbar(label='log10(Count)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_N_vs_LORE_score.png'), dpi=150)
    plt.close()

# --------------------------
# Main Execution Block
# --------------------------

if __name__ == '__main__':

    # === Configuration Parameters (EDIT THESE) ===
    config = {
        # --- Input Data ---
        "zeta_file": "zetazeros-50k.txt",  # Path to the zeta zeros file
        "num_zeros": 50000,                 # Number of zeta zeros to load and use (e.g., 100, 1000, 10000)

        # --- Domain Parameters ---
        "N_min": 2,                     # Minimum integer for the domain
        "N_max": 1000,                  # Maximum integer for the domain
        "grid_points": 50000,            # Number of points in the logarithmic grid y=log(N)

        # --- LORE Algorithm Parameters ---
        "entropy_window": 100,           # Window size (points) for local entropy
        "entropy_bins": 15,             # Number of bins for entropy histogram
        "weights": {                    # Weights for LORE score components
            'envelope': 0.9,
            'curvature': 3.5,
            'frequency': 0.9,
            'entropy': 3.5,
            'energy': 0.5
        },
        "peak_threshold": 0.0,          # Minimum LORE score for a point to be considered a peak

        # --- Prediction & Evaluation Parameters ---
        "top_N_predictions": 100,       # Number of top-scoring LORE peaks to select
        "tolerance": 0,                 # Tolerance (±N) for matching predicted N to true primes

        # --- Output Parameters ---
        "output_dir": "lore_standalone_results", # Directory to save results

        # --- Control/Robustness Experiments ---
        "edge_correction": None,        # None, 'tukey', or 'mirror'
        "run_shuffled_gamma": True,     # Run LORE with shuffled gamma
        "run_gue_gamma": True,          # Run LORE with synthetic GUE gamma
        "run_synthetic_target": True,   # Run LORE with synthetic (Hamming weight) target
        "run_train_test_split": True,   # Run train/test split for N ranges
        "taper": 'lorentz',             # Taper type: 'lorentz' or 'gaussian'
        "gamma_cutoff": None,            # Gamma cutoff for gaussian taper (if None, set to Nyquist freq)
        "grid_type": 'log',  # 'log', 'loglog', or 'wavelet'
    }
    # ===========================================

    start_time = time.time()
    os.makedirs(config["output_dir"], exist_ok=True)
    print(f"--- Starting LORE Analysis (Standalone) ---")
    print(f"Output directory: {config['output_dir']}")
    print(f"Parameters (from script):")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # --- 1. Load Zeta Zeros ---
    print("\n[Step 1/8] Loading Zeta Zeros...")
    gamma_values = load_zeta_zeros_from_txt(config["zeta_file"], config["num_zeros"])
    if gamma_values is None or len(gamma_values) == 0:
        print("[FATAL] No zeta zeros loaded. Aborting.")
        exit(1)
    actual_num_zeros = len(gamma_values)

    # --- 2. Setup Domain ---
    print("\n[Step 2/8] Setting up Domain...")
    N_min, N_max = config["N_min"], config["N_max"]
    grid_type = config.get("grid_type", 'log')
    if grid_type == 'loglog':
        # loglog grid: y = log(log(N)), N > e
        N_min = max(N_min, int(np.ceil(np.exp(1.01))))  # N > e
        y_grid = np.linspace(np.log(np.log(N_min)), np.log(np.log(N_max)), config["grid_points"], dtype=np.float64)
        N_grid = np.exp(np.exp(y_grid))
        dy = y_grid[1] - y_grid[0]
        print(f"  Using loglog grid: N = [{N_min}, {N_max}], y = loglog(N)")
    else:
        y_grid = np.linspace(np.log(N_min), np.log(N_max), config["grid_points"], dtype=np.float64)
        N_grid = np.exp(y_grid)
        dy = y_grid[1] - y_grid[0]
        print(f"  Using log grid: N = [{N_min}, {N_max}], y = log(N)")

    # --- 3. Compute Forcing Field V0(y) ---
    print("\n[Step 3/8] Computing Forcing Field V0(y)...")
    V0_y = compute_forcing_field(gamma_values, y_grid, config["taper"], config["gamma_cutoff"])
    # --- Edge Correction Option ---
    if config["edge_correction"] == 'tukey':
        print("  Applying Tukey window to V0_y before Hilbert transform...")
        V0_y = tukey_window_pad(V0_y, alpha=0.5)
    elif config["edge_correction"] == 'mirror':
        print("  Applying mirror padding to V0_y before Hilbert transform...")
        pad_width = min(1000, len(V0_y)//10)
        V0_y = mirror_pad(V0_y, pad_width=pad_width)
        y_grid = np.linspace(np.log(N_min), np.log(N_max), len(V0_y), dtype=np.float64)
        N_grid = np.exp(y_grid)

    # --- 4. LORE on True Primes (Main Run) ---
    print("\n[Step 4/8] Calculating Derived LORE Fields...")
    fields = calculate_lore_fields(y_grid, V0_y, config["entropy_window"], config["entropy_bins"])
    envelope_score, curvature_score, frequency_score, entropy_alignment_score, energy_score = fields

    if SELF_CALIBRATING_LORE:
        # --- MAIN RUN: Self-calibrating LORE ---
        print("\n[Step 5/8] Computing Composite LORE Score (Self-Calibrating)...")
        symbolic_score, lore_selfcal_details = compute_selfcal_lore_score(
            envelope_score, curvature_score, frequency_score, y_grid
        )
        print(f"  Self-calibrating exponents: {lore_selfcal_details['exponents']}")
        print(f"  Self-calibrating entropy window: {lore_selfcal_details['entropy_window']}")
    else:
        print("\n[Step 5/8] Computing Composite LORE Score (Manual Weights)...")
        score_weights = config["weights"]
        symbolic_score = compute_symbolic_score(fields, score_weights)
        lore_selfcal_details = None

    # --- 5. Prime Validation and Prediction ---
    print("\n[Step 6/8] Performing Prime Validation...")
    print(f"  Loading primes in range [{N_min}, {N_max}]...")
    N_primes = list(primerange(N_min, N_max + 1))
    if not N_primes: print("[WARN] No primes found in the specified range.")
    else: print(f"  Found {len(N_primes)} primes.")

    # Self-calibrating peak threshold selection
    peak_threshold = config["peak_threshold"]
    peak_threshold_method = "manual"
    # Force manual threshold (disable Otsu/self-calibration)
    print(f"  Manual peak threshold: {peak_threshold:.6g}")

    # Use the selected threshold for peak selection
    peak_indices = argrelextrema(symbolic_score, np.greater)[0]
    peak_indices = peak_indices[symbolic_score[peak_indices] > peak_threshold]

    if len(peak_indices) > 0:
        peak_scores = symbolic_score[peak_indices]
        sorted_peak_indices = peak_indices[np.argsort(peak_scores)[::-1]]
        actual_top_N_peaks = min(config["top_N_predictions"], len(sorted_peak_indices))
        top_indices = sorted_peak_indices[:actual_top_N_peaks]
    else:
        print(f"[WARN] No peaks found above threshold {peak_threshold}. No predictions generated.")
        top_indices = np.array([], dtype=int)
        actual_top_N_peaks = 0

    if len(top_indices) > 0:
        predicted_N = np.round(N_grid[top_indices]).astype(int)
        predicted_N = np.unique(predicted_N[(predicted_N >= N_min) & (predicted_N <= N_max)])
        print(f"  Identified {len(peak_indices)} peaks above threshold {peak_threshold}.")
        print(f"  Selected top {actual_top_N_peaks} peaks, resulting in {len(predicted_N)} unique integer predictions.")
    else: predicted_N = np.array([], dtype=int)

    eval_metrics = evaluate_predictions(predicted_N, N_primes, tolerance=config["tolerance"])
    rand_metrics = random_baseline(len(predicted_N), N_min, N_max, N_primes, tolerance=config["tolerance"])
    odds_metrics = odds_baseline(N_min, N_max, N_primes, tolerance=config["tolerance"])
    vonmangoldt_metrics = von_mangoldt_baseline(
        gamma_values, y_grid, N_grid, N_min, N_max, N_primes,
        k=len(predicted_N), tolerance=config["tolerance"])
    ks, precisions, recalls = precision_recall_vs_k(
        symbolic_score, N_grid, N_min, N_max, N_primes, k_min=10, k_max=2000, step=10, tolerance=config["tolerance"])
    precision_vs_k_path = os.path.join(config["output_dir"], 'lore_precision_vs_k.png')
    plot_precision_vs_k(ks, precisions, recalls, precision_vs_k_path)
    auroc, ap = compute_auroc_ap(symbolic_score, N_grid, N_min, N_max, set(N_primes))

    # --- Control 1: Shuffled Gamma ---
    if config["run_shuffled_gamma"]:
        print("\n[Control] Running LORE with shuffled gamma...")
        gamma_shuffled = shuffle_gamma(gamma_values, seed=123)
        V0_shuf = compute_forcing_field(gamma_shuffled, y_grid, config["taper"], config["gamma_cutoff"])
        if config["edge_correction"] == 'tukey':
            V0_shuf = tukey_window_pad(V0_shuf, alpha=0.5)
        elif config["edge_correction"] == 'mirror':
            pad_width = min(1000, len(V0_shuf)//10)
            V0_shuf = mirror_pad(V0_shuf, pad_width=pad_width)
        fields_shuf = calculate_lore_fields(y_grid, V0_shuf, config["entropy_window"], config["entropy_bins"])
        envelope_score_shuf, curvature_score_shuf, frequency_score_shuf, entropy_alignment_score_shuf, energy_score_shuf = fields_shuf
        if SELF_CALIBRATING_LORE:
            symbolic_score_shuf, _ = compute_selfcal_lore_score(
                envelope_score_shuf, curvature_score_shuf, frequency_score_shuf, y_grid
            )
        else:
            score_weights = config["weights"]
            symbolic_score_shuf = compute_symbolic_score(fields_shuf, score_weights)
        auroc_shuf, ap_shuf = compute_auroc_ap(symbolic_score_shuf, N_grid, N_min, N_max, set(N_primes))
    else:
        auroc_shuf, ap_shuf = None, None

    # --- Control 2: GUE Gamma ---
    if config["run_gue_gamma"]:
        print("\n[Control] Running LORE with synthetic GUE gamma...")
        gamma_gue = generate_gue_gamma(len(gamma_values), mean_spacing=np.mean(np.diff(gamma_values)), seed=456)
        V0_gue = compute_forcing_field(gamma_gue, y_grid, config["taper"], config["gamma_cutoff"])
        if config["edge_correction"] == 'tukey':
            V0_gue = tukey_window_pad(V0_gue, alpha=0.5)
        elif config["edge_correction"] == 'mirror':
            pad_width = min(1000, len(V0_gue)//10)
            V0_gue = mirror_pad(V0_gue, pad_width=pad_width)
        fields_gue = calculate_lore_fields(y_grid, V0_gue, config["entropy_window"], config["entropy_bins"])
        envelope_score_gue, curvature_score_gue, frequency_score_gue, entropy_alignment_score_gue, energy_score_gue = fields_gue
        if SELF_CALIBRATING_LORE:
            symbolic_score_gue, _ = compute_selfcal_lore_score(
                envelope_score_gue, curvature_score_gue, frequency_score_gue, y_grid
            )
        else:
            score_weights = config["weights"]
            symbolic_score_gue = compute_symbolic_score(fields_gue, score_weights)
        auroc_gue, ap_gue = compute_auroc_ap(symbolic_score_gue, N_grid, N_min, N_max, set(N_primes))
    else:
        auroc_gue, ap_gue = None, None

    # --- Control 3: Synthetic Target ---
    if config["run_synthetic_target"]:
        print("\n[Control] Running LORE on synthetic (Hamming weight) target...")
        synthetic_targets = synthetic_target_hamming_weight(N_min, N_max, weight=5)
        auroc_syn, ap_syn = compute_auroc_ap(symbolic_score, N_grid, N_min, N_max, set(synthetic_targets))
    else:
        auroc_syn, ap_syn = None, None

    # --- Control 4: Train/Test Split ---
    train_test_results = []
    if config["run_train_test_split"]:
        print("\n[Control] Running train/test split for N ranges...")
        # 1. Tune on 2-1000
        N_train_min, N_train_max = 2, 1000
        N_test1_min, N_test1_max = 1000, 10000
        N_test2_min, N_test2_max = 10000, 100000
        # Use current weights as tuned
        for (rng_min, rng_max, label) in [
            (N_train_min, N_train_max, 'Train (2-1000)'),
            (N_test1_min, N_test1_max, 'Test1 (1000-10000)'),
            (N_test2_min, N_test2_max, 'Test2 (10000-100000)')]:
            y_grid_sub = np.linspace(np.log(rng_min), np.log(rng_max), config["grid_points"], dtype=np.float64)
            N_grid_sub = np.exp(y_grid_sub)
            V0_sub = compute_forcing_field(gamma_values, y_grid_sub, config["taper"], config["gamma_cutoff"])
            if config["edge_correction"] == 'tukey':
                V0_sub = tukey_window_pad(V0_sub, alpha=0.5)
            elif config["edge_correction"] == 'mirror':
                pad_width = min(1000, len(V0_sub)//10)
                V0_sub = mirror_pad(V0_sub, pad_width=pad_width)
            fields_sub = calculate_lore_fields(y_grid_sub, V0_sub, config["entropy_window"], config["entropy_bins"])
            envelope_score_sub, curvature_score_sub, frequency_score_sub, entropy_alignment_score_sub, energy_score_sub = fields_sub
            if SELF_CALIBRATING_LORE:
                symbolic_score_sub, _ = compute_selfcal_lore_score(
                    envelope_score_sub, curvature_score_sub, frequency_score_sub, y_grid_sub
                )
            else:
                score_weights = config["weights"]
                symbolic_score_sub = compute_symbolic_score(fields_sub, score_weights)
            primes_sub = list(primerange(rng_min, rng_max + 1))
            auroc_sub, ap_sub = compute_auroc_ap(symbolic_score_sub, N_grid_sub, rng_min, rng_max, set(primes_sub))
            train_test_results.append((label, rng_min, rng_max, auroc_sub, ap_sub, len(primes_sub)))

    kl_div_nats = float('nan')
    mi_score = float('nan')
    try:
        if len(eval_metrics['TP List']) > 0:
            tp_hist, _ = np.histogram(eval_metrics['TP List'], bins=50, range=(N_min, N_max), density=True)
            num_bins = len(tp_hist)
            uniform_prob_per_bin = 1.0 / num_bins if num_bins > 0 else 0
            uniform_dist = np.full_like(tp_hist, uniform_prob_per_bin)
            kl_div_nats = np.sum(rel_entr(tp_hist + 1e-12, uniform_dist + 1e-12))

        if len(top_indices) > 0: score_threshold_val = symbolic_score[top_indices[-1]]
        else: score_threshold_val = np.inf
        lore_score_binary = (symbolic_score >= score_threshold_val).astype(int)
        prime_mask = np.zeros_like(y_grid, dtype=int)
        if N_primes:
            prime_indices_on_grid = [np.argmin(np.abs(y_grid - np.log(p))) for p in N_primes]
            prime_mask[prime_indices_on_grid] = 1
        mi_score = mutual_info_score(prime_mask, lore_score_binary)
    except Exception as e: print(f"[WARN] Could not calculate KL or MI: {e}")

    eval_metrics['KL Divergence (TP vs Uniform, nats)'] = kl_div_nats
    eval_metrics['Mutual Information (Score vs Primes)'] = mi_score

    # --- 6. Generate Plots ---
    print("\n[Step 8/8] Generating Plots...")
    plot_lore_score(
        N_grid, symbolic_score, N_primes,
        eval_metrics['TP List'], eval_metrics['FP List'],
        os.path.join(config["output_dir"], 'lore_score_validation_standalone.png')
    )
    plot_prediction_distribution(
        eval_metrics['TP List'], eval_metrics['FP List'],
        bins=max(20, (N_max - N_min) // 50),
        x_range=(N_min, N_max),
        out_path=os.path.join(config["output_dir"], 'prediction_distribution_standalone.png')
    )
    plot_scientific_visuals(
        config["output_dir"],
        N_grid,
        V0_y,
        envelope_score,
        curvature_score,
        frequency_score,
        entropy_alignment_score,
        energy_score,
        symbolic_score,
        N_primes,
        peak_indices,
        auroc,
        ap,
        rand_metrics,
        odds_metrics,
        vonmangoldt_metrics,
        auroc_shuf,
        ap_shuf,
        auroc_gue,
        ap_gue,
        kl_div_nats,
        mi_score,
        symbolic_score_shuf if 'symbolic_score_shuf' in locals() else None,
        symbolic_score_gue if 'symbolic_score_gue' in locals() else None,
        train_test_results
    )

    # --- 7. Generate Report ---
    print("\n--- LORE Analysis Standalone Report ---")
    summary = f"""
Parameters Used:
----------------
Zeta Zero File: {config['zeta_file']}
Number of Zeros Used: {actual_num_zeros} (Target: {config['num_zeros']})
Domain N: [{config['N_min']}, {config['N_max']}]
Log Grid Points: {config['grid_points']} (dy={dy:.6f})
Entropy Window: {config['entropy_window']} points
Entropy Bins: {config['entropy_bins']}
LORE Weights: {config['weights']}
Peak Threshold: {peak_threshold:.6g} ({peak_threshold_method})
Top N Predictions: {config['top_N_predictions']} (Actual Unique Integers: {eval_metrics['Total Unique Predictions']})
Prediction Tolerance: ±{config['tolerance']}

Performance Metrics:
--------------------
True Primes in Range: {eval_metrics['Total True Primes in Range']}
True Positives (TP - Hits): {eval_metrics['True Positives (Hits)']}
False Positives (FP - Misses): {eval_metrics['False Positives (Misses)']}
False Negatives (FN - Primes Missed): {eval_metrics['False Negatives (Primes Missed)']}
Distinct Primes Hit: {eval_metrics['Distinct Primes Hit']}

Precision (TP / Total Pred): {eval_metrics['Precision']:.4f}
Recall (Distinct Primes Hit / Total Primes): {eval_metrics['Recall']:.4f}
F1 Score (Harmonic Mean): {eval_metrics['F1 Score']:.4f}
AUROC: {auroc:.4f}
Average Precision (AP): {ap:.4f}

Analysis & Comparison:
----------------------
Random Baseline F1 Score: {rand_metrics['F1 Score']:.4f}
Odds Baseline: Precision={odds_metrics['Precision']:.4f}, Recall={odds_metrics['Recall']:.4f}, F1={odds_metrics['F1 Score']:.4f}
Von Mangoldt Baseline (top k={len(predicted_N)}): Precision={vonmangoldt_metrics['Precision']:.4f}, Recall={vonmangoldt_metrics['Recall']:.4f}, F1={vonmangoldt_metrics['F1 Score']:.4f}
KL Divergence (TP Dist. vs Uniform): {kl_div_nats:.4f} nats
Mutual Information (LORE Score Bin vs Prime Loc): {mi_score:.4f}

--- Robustness/Control Experiments ---
Shuffled Gamma: AUROC={auroc_shuf if auroc_shuf is not None else 'N/A'}, AP={ap_shuf if ap_shuf is not None else 'N/A'}
GUE Gamma: AUROC={auroc_gue if auroc_gue is not None else 'N/A'}, AP={ap_gue if ap_gue is not None else 'N/A'}
Synthetic Target (Hamming-5): AUROC={auroc_syn if auroc_syn is not None else 'N/A'}, AP={ap_syn if ap_syn is not None else 'N/A'}

--- Train/Test Split Results ---
"""
    if train_test_results:
        for label, rng_min, rng_max, auroc_sub, ap_sub, nprimes in train_test_results:
            summary += f"  {label}: N=[{rng_min},{rng_max}], Primes={nprimes}, AUROC={auroc_sub:.4f}, AP={ap_sub:.4f}\n"
    summary += f"""

Output Files:
-------------
Results Directory: {config['output_dir']}
Results CSV: lore_full_results_standalone.csv
LORE Score Plot: lore_score_validation_standalone.png
Prediction Dist. Plot: prediction_distribution_standalone.png
Precision-vs-k Plot: lore_precision_vs_k.png
"""
    print(summary)
    report_path = os.path.join(config["output_dir"], 'lore_summary_report_standalone.txt')
    try:
        with open(report_path, 'w') as f:
            params_to_save = config.copy()
            params_to_save['actual_num_zeros_used'] = actual_num_zeros
            params_to_save['dy_log_spacing'] = dy
            params_to_save['entropy_window'] = lore_selfcal_details['entropy_window'] if lore_selfcal_details else None
            params_to_save['exponents'] = lore_selfcal_details['exponents'] if lore_selfcal_details else None
            params_to_save['peak_threshold'] = peak_threshold
            params_to_save['peak_threshold_method'] = peak_threshold_method
            f.write("--- Parameters (JSON) ---\n")
            json.dump(params_to_save, f, indent=2, default=str)
            f.write("\n\n--- Summary Report ---\n")
            f.write(summary)
            f.write("\n--- Prediction Lists ---\n")
            f.write(f"TP List ({len(eval_metrics['TP List'])}): {eval_metrics['TP List']}\n")
            f.write(f"FP List ({len(eval_metrics['FP List'])}): {eval_metrics['FP List']}\n")
            f.write(f"FN List ({len(eval_metrics['FN List'])}): {eval_metrics['FN List']}\n")
    except Exception as e: print(f"[ERROR] Failed to save summary report: {e}")

    end_time = time.time()
    print(f"\n--- LORE Analysis Complete ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")