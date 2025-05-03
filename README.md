<div align="center">

# Symbolic Emergence Field Analysis (SEFA)

**Quantifying the Hidden Geometry of Emergence in Complex Data**

<div>
  <img src="https://github.com/user-attachments/assets/1d2237c5-0484-431b-91ed-669b08c3ea4f" alt="SEFA Perspective" width="512">
</div>

#### **Field-Resolved Symbolic Interference Pattern: a direct representation of SEFA’s latent resonance topology, not stylized, not symbolic, not hyperreal or forced; just the math expressing itself.**

</div>

---

## What is SEFA? A Framework for Measuring Structural Emergence

The natural world is full of patterns that emerge from simpler components - from the distribution of prime numbers to the collective behavior of neurons. These patterns aren't directly encoded in the underlying rules, yet they become recognizable structures at higher levels of organization. SEFA provides a quantitative method to identify and measure such emergent structures.

SEFA operates through a sequence of well-defined mathematical transformations:

1. **Field Construction:** We represent data as a continuous field by transforming discrete elements (like spectra or sequences) into a weighted superposition of oscillatory components.

2. **Multi-dimensional Feature Analysis:** At each point in this field, we measure four complementary structural properties:
   - **Amplitude (A):** The local magnitude of the field
   - **Curvature (C):** The second derivative of amplitude, capturing inflection points
   - **Frequency (F):** The rate of phase change, indicating local oscillatory behavior
   - **Entropy Alignment (E):** The degree of order in local amplitude patterns

3. **Information-Theoretic Calibration:** The framework objectively weights these features based on their information content. Features that show more global structure (lower entropy in their distribution) receive higher weights, eliminating subjective parameter tuning.

4. **Composite Measurement:** These weighted features combine through a geometric mean to produce a single SEFA score. This score is high only when multiple features simultaneously indicate the presence of structured information.

SEFA doesn't impose preconceived notions of what patterns should look like. Instead, it responds to inherent geometric and information-theoretic properties, making it applicable across different domains - from number theory to signal processing to experimental data analysis.

What distinguishes SEFA from other pattern-detection methods is its transparency: every step has a clear mathematical definition and interpretation, allowing us to understand exactly how and why certain structures are highlighted.

---

**Here is the logic and math flow of SEFA:**

```ascii
                       Driver Extraction
                       ──────────────────
                               │
                               ▼
                           Raw Data
                               │
                               ▼
                              FFT
                               │
                               ▼
                             {γₖ}
                               │
                               ▼
                       w(γₖ)=1/(1+γₖ²)
                               │
                        Field Construction
                       ───────────────────
                               │
                               ▼
                    V₀(y)=∑w(γₖ)cos(γₖy)
                               │
                        Feature Extraction
                       ───────────────────
                               │
            ┌────────────┬─────┴─────┬────────────┐
            │            │           │            │
            ▼            ▼           ▼            │
       Amplitude(A)    Hilbert    d²/dy²     Sliding Window
                         │           │            │
                         ▼           │            ▼
                        Z(y)         │        Entropy S(y)
                         │           │            │
                    ┌────┴────┐      │            ▼
                    │         │      │      E(y)=1-S(y)/max(S)
                    ▼         ▼      │            │
               Phase φ(y)     └──────┼────►Curvature(C)
                    │                │            │
                    ▼                │            │
                  dφ/dy              │            │
                    │                │            │
                    ▼                │            │
              Frequency(F)           │            │
                    │                │            │
                    │                │            │
                    │      Feature Normalization  │
                    │     ─────────────────────   │
                    │               │             │
            ┌───────┼────────┬──────┘             │
            │       │        │                    │
            ▼       ▼        ▼                    ▼
 F'(y)=F(y)/max|F|  │  C'(y)=C(y)/max|C|   E'(y)=E(y)/max|E|
            │       │        │                    │
            │       ▼        │                    │
            │  A'(y)=A(y)/max|A|                  │
            │       │        │                    │
            │       │        │                    │
            │       │  Self-Calibration           │
            │       │ ────────────────            │
            │       │        │                    │
       ┌────┼───────┼────────┼────────────────────┘
       │    │       │        │
       │    │       │        │
       ▼    ▼       ▼        ▼
compute I_F compute I_A compute I_C compute I_E
       │    │       │        │
       ▼    ▼       ▼        ▼
w_F=ln(B)-I_F  │  w_C=ln(B)-I_C  w_E=ln(B)-I_E
       │    │       │        │
       │    ▼       │        │
       │w_A=ln(B)-I_A        │
       │    │       │        │
       │    │       │        │
       └────┼───────┼────────┘
            │       │        │
            ▼       ▼        ▼
                 W_total=∑w_X
                      │
                      ▼
              α_X=4w_X/W_total
                      │
                      │
              Composite Score
             ───────────────
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
     α_A, A'(y)    α_C, C'(y)    α_F, F'(y)
        │             │             │
        │             │             │
        │             │             │
        └─────────────┼─────────────┘
                      │
                      ▼
                   α_E, E'(y)
                      │
                      ▼
         SEFA(y)=exp[∑α_X·ln(X'(y)+ε)]
                      │
                      │
             Physical Applications
            ───────────────────
                      │
        ┌─────────────┴──────────────┐
        │                            │
        ▼                            ▼
Wave Equation:               Quantum Mechanics: 
v(y)=v₀/(1+β·SEFA(y))        V(r)=V₀(r)+λ·SEFA(r)
```
<img src="https://github.com/user-attachments/assets/f9dd3e34-3f98-461e-bf47-f5904bd23cd9" alt="Visual" width="720">

## Key Features

*   **Self-Calibrating:** No hand-tuned parameters. Weights, thresholds, and window sizes are derived directly from the data itself using information theory.

*   **Domain-Agnostic:** While demonstrated here with number theory (zeta zeros & primes), SEFA can be applied to any signal or field where structure might emerge (physics, biology, finance, social networks, etc.).

*   **Interpretable:** Unlike many ML models, SEFA's components (A, C, F, E) and their weights are transparent and physically meaningful.

*   **Reproducible:** The code and data provided allow for full replication of the key experiments.

## Validation: The Zeta Zero / Prime Number Experiment

<img src="https://github.com/user-attachments/assets/f5369c36-2f21-4eda-85bc-5ed359966366" alt="SEFA Zeta-Prime Field Emergence" width="500">

To test SEFA rigorously, I applied it to one of the most fundamental datasets in mathematics: the non-trivial zeros of the Riemann zeta function. The hypothesis: could SEFA, *without knowing anything about primes*, detect regions correlated with prime numbers simply by analyzing the structure of a field constructed from the zeta zeros?

**The Results:**

*   SEFA demonstrated a statistically significant ability to identify prime-rich locations (AUROC ≈ 0.98 on the training range [2, 1000], ≈ 0.83 on the hold-out range [1000, 10000]).

*   Control experiments (shuffled zeros, synthetic targets) showed near-random performance, confirming the specificity of the result.

*   The method successfully identifies regions of high *structural coherence* that are strongly correlated with primes, acting as a detector of emergent symbolic patterns.

**(Important Note:** SEFA is **not** a primality test, nor an attempt to prove the Riemann Hypothesis. It's an exploratory tool for quantifying emergent structure correlated with arithmetic properties.)

*   For a detailed summary of the experiment and results, see `L.O.R.E_Paper.md`.
*   Raw output files and plots are in the `lore_standalone_results/` directory.
*   The full mathematical derivation is in `SEFA.md`.
*   A derivation exploring physical interpretations (wavefront shift, quantum energy shift) is in `SEFA_IGS.md`.

## Repository Contents

*   `README.md`: This file.
  
*   `SEFA.md`: The core mathematical derivation of the SEFA framework from first principles.

*   `SEFA_IGS.md`: Derivation exploring potential Informational Geometry / Physics interpretations.

*   `L.O.R.E_Paper.md`: A summary paper outlining the motivation, method, and results of the zeta/prime experiment (LORE was the internal project name).

*   `LORE FAQs.md`: Frequently asked questions about the LORE/SEFA concept and experiment.

*   `lore_demo.py`: A Python script demonstrating the core SEFA pipeline applied to the zeta zero data. Configuration for `lore_demo.py` is found at line [688](https://github.com/severian42/Symbolic-Emergence-Field-Analysis/blob/e679e2b2a63d6013e4621bf06b6fdf7fc28d40ea/lore_demo.py#L688) the main run execution, allowing adjustment of parameters (domain, number of zeros, etc.). *Note: Weights can be set here, but self-calibration is default.*

*   `zetazeros-50k.txt`: Text file containing the imaginary parts of the first 50,000 non-trivial zeta zeros (sourced from standard mathematical libraries/repositories).

*   `lore_standalone_results/`: Directory containing output files (plots, data summaries, network graphs) from a sample run of `lore_demo.py`.

*   `sefa_ml_model.py`: (Experimental) Contains code exploring potential machine learning integrations or comparisons with SEFA features.

## Getting Started

### Prerequisites

*   Python 3.7+
*   NumPy
*   SciPy (for Hilbert transform, signal processing)
*   Matplotlib (for plotting)
*   SymPy (for prime generation in demo)
*   Scikit-learn (for metrics like AUROC, AP)
*   Pandas (for data handling in demo)

You can install these using pip:
```bash
pip install requirements.txt
```

### Running the Demo

The primary demonstration script is `lore_demo.py`.

1.  **Configure (Optional):** Modify parameters in `lore_config.json` if desired. You can change the number range (`N_min`, `N_max`), the number of zeta zeros to use (`num_zeros`), etc. The default settings replicate the core experiment.

2.  **Run:** Execute the script from your terminal:
    ```bash
    python lore_demo.py
    ```

3.  **Output:** The script will:
    *   Load zeta zeros from `zetazeros-50k.txt`.
    *   Construct the field \(V_0(y)\) over the specified log-domain.
    *   Calculate the SEFA features (A, C, F, E).
    *   Self-calibrate the weights (\(\alpha, \beta, \gamma, \delta\)).
    *   Compute the composite SEFA score.
    *   Identify peaks in the SEFA score.
    *   Compare peak locations to prime numbers in the range.
    *   Print performance metrics (Precision, Recall, F1, AUROC, AP) to the console.
    *   Generate plots (like SEFA score vs. N, prediction distribution) and save them, along with data summaries, to a results directory (default: `lore_standalone_results`).

## Understanding the Configuration (`lore_config.json`)

This file allows you to control the execution of `lore_demo.py`:

*   `zeta_file`: Path to the zeta zeros data.
*   `num_zeros`: How many zeros to use for field construction.
*   `N_min`, `N_max`: Integer range for analysis.
*   `grid_points`: Number of points for the discretized log-domain `y`.
*   `entropy_window`, `entropy_bins`: Parameters for local entropy calculation (though window size self-calibrates by default).
*   `weights`: *Can* be used to manually override self-calibration (e.g., `{"envelope": 1.0, "curvature": 1.0, ...}`). If `null` or missing, self-calibration is used.
*   `top_N_predictions`: How many top SEFA peaks to consider as predictions.
*   `output_dir`: Where to save results.
*   ... (other parameters controlling plotting, baselines, etc.)

---

# Applying SEFA Across Domains: From Theory to Practice

One of SEFA's most powerful aspects is its domain-agnostic nature. While our primary demonstration focuses on the remarkable connection between zeta zeros and prime numbers, the framework's core principles can be applied to virtually any system where emergent structures might exist within complex data.

## Understanding SEFA's Universal Approach

At its essence, SEFA transforms raw data into a continuous field and then analyzes the geometric and information-theoretic properties of this field. This approach works because many forms of structural emergence share common characteristics:

- **Local amplitude variations** indicating regions of constructive interference
- **Sharp curvature** highlighting structural transitions
- **Frequency modulation** signaling pattern changes
- **Entropy gradients** revealing islands of order amid chaos

These properties appear across remarkably diverse systems - from quantum phenomena to biological patterns to financial market structures.

## Step-by-Step Application to New Domains

To apply SEFA to your specific problem:

1. **Identify your spectral drivers**: 
   - For time series: Fourier or wavelet components
   - For networks: Graph eigenvalues
   - For spatial data: Eigenmodes of relevant operators
   - For quantum systems: Energy levels or resonances

2. **Map your domain appropriately**:
   - Choose a continuous embedding that preserves the structure you're interested in
   - For discrete sets: Consider logarithmic (for multiplicative relationships) or other transformations

3. **Configure the analysis pipeline**:
   - Use `lore_demo.py` as a starting template
   - Modify the grid configuration to match your domain's scale
   - Adjust the feature extraction for domain-specific characteristics

4. **Interpret results contextually**:
   - SEFA highlights regions of potential structural significance
   - Domain expertise is crucial for interpreting what these regions represent

## Practical Applications

### Physics and Signal Processing
- **Spectral Analysis**: Identify resonance peaks in experimental data
- **Material Science**: Detect phase transitions and structural anomalies
- **Quantum Systems**: Analyze energy level distributions and identify coherent states

### Data Science and Machine Learning
- **Anomaly Detection**: Identify unusual patterns in high-dimensional data
- **Feature Engineering**: Use SEFA as a full SciKit ML model or even just a preprocessor for other ML models (see `sefa_ml_model.py`)
- **Clustering Enhancement**: Improve cluster detection by focusing on high-SEFA regions

### Biological Systems
- **Neural Activity**: Identify coordinated firing patterns in neural recordings
- **Genomic Analysis**: Detect functional regions in DNA sequences
- **Ecological Data**: Find emergent patterns in species distribution

### Finance and Economics
- **Market Microstructure**: Identify regimes of order/disorder in price data
- **Economic Time Series**: Detect emergent cycles and structural breaks
- **Network Analysis**: Find coherent subgraphs in economic/financial networks

## Extended Example: Analyzing Time Series Data

Consider a time series with complex, multi-scale patterns:

```python
# Example adaptation for time series analysis using the SEFA class
import numpy as np
from scipy.fft import rfft, rfftfreq # Use real FFT for real data
from scipy.signal import find_peaks
from sefa_code.sefa import SEFA, SEFAConfig # Import the actual SEFA components

# --- Configuration ---
# 0. Define SEFA configuration (optional, defaults can be used)
config = SEFAConfig(
    # Add any specific configurations needed for time series if different from defaults
    # e.g., p_features, entropy_window_size might need tuning
    p_features=5,         # Example: Number of features (adjust as needed)
    entropy_window_size=21 # Example: Window size (adjust based on data characteristics)
)
NUM_DRIVERS = 10 # Example: Number of top frequency components to use as drivers

# --- Data Preparation ---
# 1. Generate or load your time series data
sampling_rate = 100 # Hz (Example)
duration = 10       # seconds (Example)
time = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
# Example: A signal with a few dominant frequencies + noise
signal = (np.sin(2 * np.pi * 5 * time) +
          0.5 * np.sin(2 * np.pi * 15 * time) +
          0.3 * np.sin(2 * np.pi * 30 * time) +
          0.5 * np.random.randn(len(time)))
your_time_series_data = signal

# 2. Extract spectral drivers (frequencies) using FFT
yf = rfft(your_time_series_data)
xf = rfftfreq(len(your_time_series_data), 1 / sampling_rate) # Get the frequencies
# Select top K frequencies based on amplitude as drivers
spectrum_amplitudes = np.abs(yf)
# Exclude 0 Hz component if desired
top_k_indices = np.argsort(spectrum_amplitudes[1:])[-NUM_DRIVERS:] + 1
drivers_gamma = xf[top_k_indices] # Frequencies are the drivers

print(f"Using top {NUM_DRIVERS} frequencies as drivers: {np.round(drivers_gamma, 2)}")

# 3. Define the domain for analysis (map to time points)
ymin = time[0]
ymax = time[-1]
num_points = len(your_time_series_data) # Analyze at the original time resolution

# --- SEFA Analysis ---
# 4. Initialize and Run SEFA Pipeline
sefa_analyzer = SEFA(config=config)
sefa_analyzer.run_pipeline(
    drivers_gamma=drivers_gamma,
    ymin=ymin,
    ymax=ymax,
    num_points=num_points
)

# 5. Get Results
results = sefa_analyzer.get_results()
sefa_score = results['sefa_score']
processed_domain_y = results['processed_domain_y'] # This corresponds to the time points

# --- Interpretation ---
# 6. Find and interpret significant regions (e.g., using peak finding)
# Find peaks in the SEFA score - adjust prominence/height as needed
peaks, properties = find_peaks(sefa_score, prominence=np.std(sefa_score)) # Example threshold
significant_times = processed_domain_y[peaks]
peak_scores = sefa_score[peaks]

print(f"\nFound {len(significant_times)} significant time points:")
for t, s in zip(significant_times, peak_scores):
    print(f"  Time: {t:.3f} s, SEFA Score: {s:.4f}")

# Optional: Use SEFA's built-in thresholding
# mask = sefa_analyzer.threshold_score(method='percentile', percentile=98)
# significant_times_thresh = processed_domain_y[mask]
# print(f"\nFound {len(significant_times_thresh)} points above 98th percentile threshold.")

# --- Plotting (Optional Example) ---
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot original signal
color = 'tab:grey'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Original Signal', color=color)
ax1.plot(time, your_time_series_data, color=color, alpha=0.6, label='Original Signal')
ax1.tick_params(axis='y', labelcolor=color)

# Plot SEFA score on a secondary axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('SEFA Score', color=color)
ax2.plot(processed_domain_y, sefa_score, color=color, label='SEFA Score')
ax2.scatter(significant_times, peak_scores, color='black', marker='o', s=50, zorder=10, label=f'Significant Peaks ({len(significant_times)})')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
plt.title('SEFA Analysis of Time Series Data')
plt.show()
```

## Integration with Machine Learning

The `sefa_ml_model.py` file provides scikit-learn compatible components that allow you to:

1. Use SEFA as a full ML model or even just a feature extractor in other ML pipelines
2. Train models to recognize patterns with SEFA characteristics
3. Create hybrid systems that combine traditional and SEFA-based approaches

This integration makes SEFA particularly valuable for problems where purely statistical approaches miss important structural features.

## Limitations and Considerations

When applying SEFA to new domains, keep these points in mind:

- **Scale Sensitivity**: The appropriate scale for analysis depends on your data; test multiple grid resolutions
- **Computational Cost**: Higher resolution analyses require more computational resources
- **Interpretation**: SEFA highlights *potential* structure; domain knowledge is needed to validate significance
- **Validation**: Always use control experiments (shuffled data, synthetic targets) to confirm specificity

## Getting Started with Your Own Data

To begin exploring your dataset with SEFA:

1. Start with the simplest possible mapping of your data to a continuous domain
2. Use a moderate number of spectral drivers (1,000-5,000) initially
3. Test with a range of domain scales to find the right resolution
4. Compare SEFA results with known structures in your data (if available)
5. Refine your approach based on these initial insights

Remember that SEFA is an exploratory tool - it works best when combined with domain expertise and complementary analytical methods.

---

## Contributing

While this is primarily my personal research framework, I'm open to collaboration and suggestions. Feel free to open issues for bugs or feature ideas. If you wish to contribute code, please open an issue first to discuss the proposed changes.

## Citation

If you use SEFA or the results from the zeta/prime experiment in your work, please cite this repository and/or the associated `LORE BRIEF.md` document (treating it as a white paper/preprint).

```
Dillon, B. (2025). Symbolic Emergence Field Analysis (SEFA). Available from [https://github.com/[your-username]/sefa](https://github.com/severian42/Symbolic-Emergence-Field-Analysis-SEFA)
```

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Contact

For questions or discussions, you can reach me at beckettdillon42@gmail.com.

---
