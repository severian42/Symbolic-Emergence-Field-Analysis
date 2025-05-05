<div align="center">

# Symbolic Emergence Field Analysis (SEFA)

**Quantifying the Hidden Geometry of Emergence in Complex Data**

<div>
  <img src="https://github.com/user-attachments/assets/1cf0706a-941f-4e75-80ce-f013756d0db9" alt="SEFA Perspective" width="512">
</div>

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

## Repository Contents

*   `L.O.R.E_Paper.md`: A summary paper outlining the motivation, method, and results of the zeta/prime experiment (LORE was the internal project name).

*   `LORE FAQs.md`: Frequently asked questions about the LORE/SEFA concept and experiment.

*   `lore_demo.py`: A Python script demonstrating the core SEFA pipeline applied to the zeta zero data. Configuration for `lore_demo.py` is found at line [688](https://github.com/severian42/Symbolic-Emergence-Field-Analysis/blob/e679e2b2a63d6013e4621bf06b6fdf7fc28d40ea/lore_demo.py#L688) the main run execution, allowing adjustment of parameters (domain, number of zeros, etc.). *Note: Weights can be set here, but self-calibration is default.*

*   `zetazeros-50k.txt`: Text file containing the imaginary parts of the first 50,000 non-trivial zeta zeros (sourced from standard mathematical libraries/repositories).

*   `lore_standalone_results/`: Directory containing output files (plots, data summaries, network graphs) from a sample run of `lore_demo.py`.

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


## Citation

If you use SEFA or the results from the zeta/prime experiment in your work, please cite this repository and/or the associated documents (treating it as a white paper/preprint) and research codebase.

```
Dillon, B. (2025). Symbolic Emergence Field Analysis (SEFA). Available from [[https://github.com/[your-username]/sefa](https://github.com/severian42/Symbolic-Emergence-Field-Analysis-SEFA)](https://github.com/severian42/Symbolic-Emergence-Field-Analysis)
```

## License

Intellectual Property Notice – SEFA Framework

The SEFA (Symbolic Emergence Field Analysis) algorithm, codebase, and all mathematical concepts, including but not limited to:

- Entropy-weighted symbolic emergence scoring
- Normalized local signal decomposition (A, C, F, E features)
- Global alpha calibration via information deficit
- Log-based exponential feature synthesis and scoring
- Self-calibrated emergence detection for 1D, 2d, 3d and multi-dimensional time-series
- The SEFA scoring profile aggregation and symbolic peak analysis

are proprietary intellectual property of BeckettDillon42@gmail.com protected under the Business Source License 1.1 and U.S. copyright law.

No portion of this repository, code, or theoretical framework may be used in commercial products, research platforms, or derivative software without a separate, signed commercial license agreement.

All rights to SEFA, including algorithmic extensions, theoretical variants, and any SEFA-related scoring functions, are reserved by the author.

Any unauthorized commercial use, distribution, or integration will constitute IP infringement and is subject to legal action.

For licensing inquiries, contact: BeckettDillon42@gmail.com

(Patent Pending – United States, Provisional Filed: May 1, 2025)


The SEFA math and [White Paper](https://github.com/severian42/Symbolic-Emergence-Field-Analysis/tree/main/SEFA%20White%20Paper) is licensed under the [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/) License - see the LICENSE.md file for details.

## Contact

If you would like to discuss potential commercial licensing for your use case and industry, and for any other questions or discussions, you can reach me at BeckettDillon42@gmail.com

---
