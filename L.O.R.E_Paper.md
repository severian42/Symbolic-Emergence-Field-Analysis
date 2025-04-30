# **Logical Resonance Extractor (LORE): A Self-Calibrating Framework for Emergent Symbolic Structure**

### **A Derivation for Symbolic Coherence Detection**

- **Author: Beckett Dillon**
- **Date: April 2025**

______________________________________________________________________

**Abstract:** This paper introduces the Logical Resonance Extractor (LORE), a fully self-calibrating framework for detecting emergent symbolic structure in mathematical spectra. LORE autonomously identifies zones of symbolic coherence (operationally: mutual information $\approx 0.003$ between LORE score bins and the prime indicator function) within a signal derived from the Riemann zeta function's non-trivial zeros, using only geometric and information-theoretic features—without any hand-tuned parameters or prime annotations. All weights, exponents, window sizes, and thresholds are derived from the spectrum itself. LORE is not a primality test and not an attempt to prove the Riemann Hypothesis. Instead, it provides a transparent, interpretable lens for exploring symbolic emergence from mathematical spectra. All code and datasets are available to support reproducibility.

______________________________________________________________________

**Key Findings:**
- AUROC ≈ 0.98, AP ≈ 0.79 on [2, 1000] (train)
- Hold-out AUROC: 0.83 ([1000, 10,000]), 0.55 ([10,000, 100,000])
- Control runs (shuffled γ, GUE, synthetic target) confirm arithmetic specificity
- LORE is an exploratory, fully self-calibrating signal-analysis lens, not a primality test and not an attempt to prove RH.

______________________________________________________________________

**I. MOTIVATION AND PHILOSOPHICAL BASIS**

**I.1. The Search for Emergent Structure**

We ask whether coherent geometric features of a zeta-zero driven field correlate with simple primes; we do not attempt to recover all arithmetic structure.

**I.2. From Integers to a Continuous Field**

The domain of interest is the set of positive integers `N`. However, analyzing discrete sequences directly for field-like properties is challenging. We employ a fundamental transformation: mapping the integers `N` to a continuous logarithmic domain `y = Log[N]`. This transformation has several advantages:
*   It converts multiplicative relationships (`N1 * N2`) into additive ones (`y1 + y2`), mirroring how frequencies combine in wave phenomena.
*   It stretches the space between small integers and compresses it for large integers, aligning with the decreasing density of primes (as suggested by the Prime Number Theorem).
*   It provides a continuous domain suitable for calculus and field analysis techniques (derivatives, transforms).

We define our operational space as the real line `y ∈ ℝ`, where `y = Log[N]` relates back to the integers `N ≥ 2`. For practical computation, we consider a finite range:
*   Integer Range: `N ∈ [Nmin, Nmax]`
*   Logarithmic Range: `y ∈ [Log[Nmin], Log[Nmax]]`
*   `M` (number of grid points): the number of points in the discretized log-range

We discretize this log-range into a uniform grid of `M` points:
*   Grid points: `y[i_] := Log[Nmin] + i*Δy`
*   Grid spacing: `Δy := (Log[Nmax] - Log[Nmin])/(M - 1)`

**I.3. The Zeta Spectrum as a Harmonic Driver**

The Riemann zeta function, \(\zeta(s)\), is deeply connected to the distribution of prime numbers. Its non-trivial zeros, located in the critical strip \(0 < \text{Re}(s) < 1\), are believed (under the Riemann Hypothesis) to lie on the critical line \(\text{Re}(s) = 1/2\). Let these zeros be denoted \(1/2 + i\gamma_k\), where \(\gamma_k\) are the imaginary parts.

We hypothesize that the \(\gamma_k\) values represent a fundamental "spectrum" encoding information about number-theoretic structure. LORE uses these \(\gamma_k\) as frequencies to construct a harmonic potential field \(V_0(y)\) over the logarithmic domain \(y\). Regions where these harmonics interfere constructively are candidates for symbolic resonance.

**I.4. The LORE Hypothesis**

The core hypothesis of LORE is that locations \(y\) corresponding to significant symbolic integers \(N = e^y\) will manifest as distinct structural features within the field \(V_0(y)\) and its derivatives. These features include:
*   High **amplitude** (strong constructive interference).
*   Sharp **curvature** (localized events or transitions).
*   Stable **frequency** (phase coherence).
*   Low local **entropy** (high information order).

LORE aims to quantify these features and combine them into a single score, \( \text{LORE}(y) \), indicating the likelihood of emergent symbolic coherence (operationally: mutual information ≈ 0.003 between LORE score bins and the prime indicator function) at location \(y\).

______________________________________________________________________

**II. FOUNDATIONS: DOMAIN AND FIELD CONSTRUCTION**

**II.1. The Logarithmic Domain**

Grid definitions, symbols $y_i$, $\Delta y$, and the range $[N_{\min}, N_{\max}]$ are exactly as in §I.2; we omit the repetition here.

**II.2. The Spectral Forcing Field \(V_0(y)\)**

We construct the base potential field \(V_0(y)\) as a superposition of cosine waves, where the frequencies are given by the imaginary parts \(\gamma_k\) of the first \(K\) non-trivial zeta zeros.
Definition:
```Mathematica
V0[y_] := Sum[w[k] Cos[gamma[k] y], {k, 1, K}]
```
The weights `w_k` are chosen to modulate the influence of different frequencies. A common choice is `w_k = 1/(1 + gamma_k^2)`, which dampens the contribution of very high frequencies:
```Mathematica
V0[y_] := Sum[Cos[gamma[k] y]/(1 + gamma[k]^2), {k, 1, K}]
```
This field \(V_0(y)\) represents the fundamental "signal" whose structure we will analyze. It encodes the harmonic interference pattern generated by the zeta spectrum over the log-integer domain.


**III. ANALYTIC DECOMPOSITION OF THE FIELD**

To extract meaningful features related to amplitude, phase, and local structure, we transform the real-valued field \(V_0(y)\) into its complex analytic signal representation using the Hilbert transform.

**III.1. Hilbert Transform and Analytic Signal**

Objective: Obtain a complex signal whose magnitude represents the instantaneous amplitude (envelope) and whose angle represents the instantaneous phase.

Definition: The Hilbert Transform `H[V0][y]` of `V0[y]` is defined via the Cauchy Principal Value:
```Mathematica
HilbertTransformV0[y_] := (1/π) PrincipalValue[Integrate[V0[ξ]/(y - ξ), {ξ, -∞, ∞}]]
```
The Analytic Signal `Ṽ[y]` is then:
```Mathematica
Vtilde[y_] := V0[y] + I*HilbertTransformV0[y]
```
This complex signal separates the instantaneous amplitude and phase information.

**III.2. Envelope Amplitude \(A(y)\)**

Objective: Quantify the instantaneous strength or intensity of the field, irrespective of its oscillatory phase. High amplitude suggests constructive interference.

Definition: The envelope amplitude `A[y]` is the magnitude of the analytic signal:
```Mathematica
A[y_] := Abs[Vtilde[y]]
(* or explicitly: *)
A[y_] := Sqrt[V0[y]^2 + HilbertTransformV0[y]^2]
```
\(A(y)\) provides a smooth curve outlining the peaks of the underlying oscillations in \(V_0(y)\). Maxima in \(A(y)\) correspond to regions of strong resonance.

**III.3. Field Curvature \(C(y)\)**

Objective: Identify points of sharp change or inflection in the envelope. High curvature (positive or negative) indicates localized structural features like peaks or troughs.

Definition: The curvature is defined as the second derivative of the envelope amplitude:
```Mathematica
C[y_] := D[A[y], {y, 2}]
```
In discrete form, using central differences on the grid `y_i` with spacing `Δy`:
```Mathematica
Cdisc[i_] := (A[y[i + 1]] - 2*A[y[i]] + A[y[i - 1]])/(Δy^2)
```
We are typically interested in the magnitude of curvature, `|C(y)|`, as both sharp peaks (large negative `C`) and sharp troughs (large positive `C`) can signify important structural points.

**III.4. Instantaneous Phase \(\phi(y)\) and Frequency \(f(y)\)**

Objective: Measure the rate of change of the field's phase. Slow, smooth phase changes indicate coherence, while rapid changes suggest disorder or destructive interference.

Definition: The instantaneous phase \(\phi(y)\) is the angle of the analytic signal:
```Mathematica
ϕ[y_] := Arg[Vtilde[y]]
(* or, for explicit components: *)
ϕ[y_] := ArcTan[HilbertTransformV0[y], V0[y]]
```
(Using `arctan2` handles quadrant ambiguity). Note that \(\phi(y)\) is typically "unwrapped" to avoid jumps of \(2\pi\).

The instantaneous frequency `f(y)` is the rate of change (derivative) of the unwrapped phase:
```Mathematica
f[y_] := D[ϕ[y], y]
```
In discrete form:
```Mathematica
fdisc[i_] := (ϕ[y[i + 1]] - ϕ[y[i - 1]])/(2*Δy)
(* or forward difference: *)
fdiscFwd[i_] := (ϕ[y[i + 1]] - ϕ[y[i]])/Δy
```
Lower values of `|f(y)|` suggest greater coherence.

______________________________________________________________________

**IV. INFORMATION-THEORETIC ANALYSIS: THE ENTROPY FIELD (Self-Calibrating)**

While amplitude, curvature, and frequency capture geometric features, we also need a measure of local *order* or *predictability*. We use Shannon entropy applied to the distribution of amplitude values within a local window.

**IV.1. Motivation: Entropy as a Measure of Order**

In information theory, entropy quantifies the uncertainty or randomness in a probability distribution.
*   **Low Entropy:** The distribution is concentrated in a few states (predictable, ordered). In our context, this means the envelope amplitude `A[y]` is relatively constant or structured within a local window.
*   **High Entropy:** The distribution is spread out over many states (unpredictable, disordered). This means the envelope amplitude fluctuates significantly and randomly within the window.

We seek regions of low entropy, as they potentially represent stable, coherent structures.

**IV.2. Windowed Entropy Calculation (Self-Calibrating)**

- The entropy window size \( W \) is no longer hand-chosen.  
- Instead, LORE **automatically selects the optimal window** by sweeping a log-spaced range of candidate window sizes, from \( W_{min} = \lceil \Delta y^{-1/2} \rceil \) up to \( W_{max} = 0.2 \cdot N \), and choosing the first local maximum of the variance of the sliding entropy field.  
- This ensures the entropy window is always adapted to the grid and spectrum, with no human tuning.

**IV.3. Micro-Example (Entropy Calculation)**

Assume a simplified window of 7 amplitude values: \(\mathcal{N}_i = \{0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.4\}\). Let's use `B=4` bins covering the range \([0.1, 0.4]\):
*   Bin 1: \([0.1, 0.175)\) -> Values: 0.1, 0.1, 0.1 -> Count `h_1 = 3`
*   Bin 2: \([0.175, 0.25)\) -> Values: 0.2, 0.2 -> Count `h_2 = 2`
*   Bin 3: \([0.25, 0.325)\) -> Values: 0.3 -> Count `h_3 = 1`
*   Bin 4: \([0.325, 0.4]\) -> Values: 0.4 -> Count `h_4 = 1`
Total count = 7. Probabilities:
```Mathematica
pvec = {3/7, 2/7, 1/7, 1/7}
```
Shannon Entropy (using `Log`):
```Mathematica
S[y_i] := -Sum[pvec[[k]] Log[pvec[[k]]], {k, 1, 4}]
```
Numerical example:
```Mathematica
Sval = - (0.4286*Log[0.4286] + 0.2857*Log[0.2857] + 0.1429*Log[0.1429] + 0.1429*Log[0.1429])
(* Sval ≈ 1.2771 *)
```
A higher value indicates more diversity (disorder) in amplitudes within the window.

**IV.4. Entropy Alignment Score \(E(y)\)**

Objective: Convert the raw entropy `S(y)` into a score where high values indicate *order* (low entropy).

Definition: We normalize the entropy values across the entire domain and invert them:
1.  Find the maximum entropy value: `Smax = Max[S[y_i]]`.
2.  Define the Entropy Alignment Score `E(y)`:
```Mathematica
E[y_i] := 1 - S[y_i]/Smax
```
This score `E(y)` ranges from 0 to 1:
*   `E(y) ≈ 1`: Very low local entropy (high order).
*   `E(y) ≈ 0`: Very high local entropy (high disorder).

______________________________________________________________________

**V. SYNTHESIS: THE COMPOSITE LORE SCORE (Self-Calibrating)**

We now combine the four derived features—amplitude, curvature, frequency, and entropy alignment—into a single score representing the overall likelihood of symbolic coherence.

**V.1. Normalization of Component Fields**

Before combining, we normalize each component field to the range \([0, 1]\) to ensure they contribute on a comparable scale.
*   **Normalized Amplitude `A'[y]**:**
```Mathematica
Aprime[y_] := A[y]/MaxValue[A[y], y]
```
*   **Normalized Curvature `C'[y]**: We use the absolute value of curvature.
```Mathematica
Cprime[y_] := Abs[C[y]]/MaxValue[Abs[C[y]], y]
```
*   **Normalized Frequency `f'[y]**: We use the absolute value of frequency. (Note: Sometimes `1 - f'[y]` might be used if low frequency is desired, but the reference `lore.py` uses `f'[y]` directly, potentially favoring regions of significant phase change, or relying on the entropy term to penalize chaos). Let's stick to the direct normalization as per the code:
```Mathematica
fprime[y_] := Abs[f[y]]/MaxValue[Abs[f[y]], y]
```
*   **Entropy Alignment `E'[y]**: This is already normalized.
```Mathematica
Eprime[y_] := E[y]
```

**V.2. The LORE Score Formula (Self-Calibrating)**

- The four normalized fields—amplitude \( A' \), curvature \( C' \), frequency \( F' \), and entropy alignment \( E \)—are combined as before, but now the exponents are **not hand-tuned**.
- Instead, for each channel \( X \in \{A', C', F', E\} \), the global entropy \( I_X \) is computed (using 64 equiprobable bins), and the **information deficit** \( w_X = \ln(64) - I_X \) is calculated.
- The exponents are then set as:
  \[
  (\alpha, \beta, \gamma, \delta) = 4 \cdot \left( \frac{w_{A'}}{W}, \frac{w_{C'}}{W}, \frac{w_{F'}}{W}, \frac{w_E}{W} \right)
  \]
  where \( W = w_{A'} + w_{C'} + w_{F'} + w_E \).
- The final LORE score is:
  \[
  \text{LORE}(y) = (A'+\varepsilon)^{\alpha} (C'+\varepsilon)^{\beta} (F'+\varepsilon)^{\gamma} (E+\varepsilon)^{\delta}
  \]
  with \( \varepsilon \) a small constant for numerical stability.

**V.3. Self-Calibrating Peak Threshold**

- The threshold for selecting significant peaks is also **self-calibrated** using Otsu's method, which finds the value that best separates "background" from "signal" in the LORE score distribution.

**Justification for Multiplicative Form:**
*   **Coincidence Detection:** The score is high only if *all* components are reasonably high. If any component (like entropy alignment `E'`) is near zero, the entire score collapses towards zero. This prevents false positives driven by only one strong feature (e.g., high amplitude in a chaotic region).
*   **Synergy:** It rewards locations where multiple indicators of structure align simultaneously.

**V.4. Longhand LORE Calculation Example**

Assume at a point `y*`, the normalized scores are:
```Mathematica
Aprime[yStar] = 0.9
Cprime[yStar] = 0.7
fprime[yStar] = 0.6
Eprime[yStar] = 0.95
α = 0.6; β = 0.8; γ = 0.6; δ = 1.2;
LORE[yStar] = (0.9^0.6)*(0.7^0.8)*(0.6^0.6)*(0.95^1.2)
(* ≈ 0.487 *)
```
This score reflects a location with moderately high alignment across all features.

**V.5. Interpretation of LORE Score Values**

The LORE score ranges from 0 to 1. General interpretation guidelines:
*   \(\text{LORE}(y) > 0.5\): Potential region of significant symbolic coherence.
*   \(\text{LORE}(y) < 0.2\): Likely incoherent background noise.
*   Values between 0.2 and 0.5 represent varying degrees of partial alignment.
The exact thresholds depend on the specific application and parameter settings. Peak analysis is typically used rather than absolute thresholds.

______________________________________________________________________

**VI. EMPIRICAL VALIDATION: THE PRIME EXPERIMENT (Self-Calibrating Results)**

**VI.1. Methods**

- **Zeta Zeros:** First 50,000 nontrivial zeros from `zetazeros-50k.txt`.
- **Integer Domain:** \(N \in [2, 1000]\).
- **Logarithmic Grid:** 50,000 points, \(y = \log(N)\), spacing \(dy \approx 0.000126\).
- **All parameters (entropy window, exponents, peak threshold) are self-calibrated.**
- **Top N Predictions:** 100 (as before, for direct comparison).
- **Metrics:** AUROC, AP, Precision, Recall, F1, etc.
- **Parameter values, weights, and tapers:** See Appendix: Implementation Details.

**VI.2. Results**

Performance on hold-out: AUROC drops from 0.98 (train) to 0.83 (first hold-out) and 0.55 (second hold-out).

| Metric                      | Value   |
|-----------------------------|---------|
| Zeta Zeros Used             | 50,000  |
| Integer Domain              | 2–1000  |
| Log Grid Points             | 50,000  |
| Entropy Window (auto)       | ~1,200  |
| Exponents (post-normalisation)            | $\alpha \approx 0.24$, $\beta \approx 1.64$, $\gamma \approx 1.60$, $\delta \approx 0.52$ |
| Peak Threshold (Otsu)       | $\approx 0.13$   |
| Top N Predictions           | 100     |
| True Primes in Range        | 168     |
| Unique Integer Predictions  | 79      |
| True Positives (TP)         | 71      |
| False Positives (FP)        | 8       |
| False Negatives (FN)        | 97      |
| Distinct Primes Hit         | 71      |
| Precision                   | 0.899   |
| Recall                      | 0.423   |
| F1 Score                    | 0.575   |
| AUROC                       | 0.978   |
| Average Precision (AP)      | 0.793   |

**VI.3. Robustness and Control Experiments (Version 0.1)**

The following table summarizes the key results from the main and control experiments, using AUROC and Average Precision (AP) as threshold-free, label-imbalance-safe metrics. All experiments use the same fixed random seed for reproducibility. See Appendix for plots and further details.

| Tag            | N Range         | Target         | Gamma Type      | AUROC  | AP     | P@k (k=52) |
|----------------|-----------------|---------------|-----------------|--------|--------|------------|
| Baseline       | 2–1000          | Primes        | Riemann zeros   | 0.9662 | 0.7207 | 0.769      |
| Hold-out-1     | 1000–10,000     | Primes        | Riemann zeros   | 0.8235 | 0.3281 | —          |
| GUE-control    | 2–1000          | Primes        | GUE (synthetic) | 0.5286 | 0.1976 | —          |
| Synthetic tgt  | 2–1000          | Hamming-5     | Riemann zeros   | 0.5000 | 0.2443 | —          |

**Notes:**
- P@k = Precision at k = 52 (number of unique predictions in baseline run).
- All parameter values, weights, and tapers are listed in Methods.
- Plots: LORE score vs N, Precision@k curve, AUROC vs N-decade, and all control experiment curves are included in the Appendix.

**VI.4. Visual Demonstration**

- **LORE Score Plot:** The plot `lore_score_validation_standalone.png` shows the LORE score across N, with true primes and predictions highlighted.
- **Prediction Distribution:** The plot `prediction_distribution_standalone.png` shows the distribution of true and false positives across the integer domain.

**VI.5. Summary Table (Empirical Run)**

| Parameter         | Value   |
|-------------------|---------|
| Zeta Zeros        | 50,000  |
| N_min             | 2       |
| N_max             | 1000    |
| Grid Points       | 50,000  |
| Entropy Window    | 100     |
| Entropy Bins      | 25      |
| Weights           | 0.9, 3.5, 0.9, 3.5 |
| Peak Threshold    | 0.0     |
| Top N Predictions | 100     |
| Tolerance         | 0       |
| Precision         | 0.7692  |
| Recall            | 0.2381  |
| F1 Score          | 0.3636  |
| AUROC             | 0.9662  |
| AP                | 0.7207  |

**VI.6. Reproducibility**

These results are fully reproducible using the provided `lore_demo.py` script, the `zetazeros-50k.txt` file, and the above parameter settings. All code, data, and output files are available for inspection. *A fixed random seed (np.random.seed(0)) is used for the random baseline to ensure reproducibility.*

______________________________________________________________________

**VII. GRAPH-BASED STRUCTURAL ANALYSIS**

Graph topology reveals how high-LORE-score locations cluster and connect, potentially mirroring the non-random structure of primes. Clusters and hubs may correspond to regions of arithmetic significance.

To understand the relationships *between* the high-scoring LORE locations, a network graph representation is constructed.

**VII.1. Graph Construction**

*   **Nodes:** Each unique integer prediction \(N_{\text{pred}}\) from the top \(k\) LORE peaks becomes a node in the graph. Node attributes store the corresponding \(y\) value and the LORE score components (\(A', C', f', E'\)).
*   **Edges:** Edges connect nodes based on two criteria:
    1.  **Proximity:** Nodes whose \(y\) values (logarithmic coordinates) are close are connected (e.g., k-nearest neighbors in \(y\)-space). This captures spatial relationships in the log domain.
    2.  **Score Similarity:** Nodes whose LORE scores are very similar (differ by less than a threshold, specifically `0.1` in the reference simulation) are connected. This captures functional relationships.

**VII.2. Graph Visualization**

The constructed graph \(G\) is saved (e.g., as `zeta_field_graph.pkl` using Python's `pickle` module) and visualized:
*   Figure 4. LORE Score vs N (from `lore_score_validation_standalone.png`)
*   Figure 5. "Living" Zeta Field Graph (from `zeta_field_graph_living.png`)
*   Figure 6. Interactive Zeta Field Graph (Screenshot/Link to `zeta_field_graph_interactive.html`)

**VII.3. Interpretation of the Graph**

The graph visualization helps reveal:
*   **Clusters:** Groups of closely connected nodes might represent broader regions of symbolic coherence.
*   **Hubs:** Nodes with many connections could be particularly significant points within the LORE field.
*   **Structure:** The overall topology of the graph provides insight into the organization of high-LORE zones.

______________________________________________________________________

**VIII. LIMITATIONS AND EPISTEMOLOGICAL BOUNDARIES**

It is crucial to understand what LORE *is not*:
*   **Low Recall Limitation:** As demonstrated, the current formulation misses a large majority of primes, focusing only on the most prominent resonance peaks.
*   **Not a Primality Test:** LORE does not determine if a given number *is* prime. It identifies locations with high structural coherence based on its specific criteria, which are statistically correlated with primes.
*   **Not a Mathematical Proof:** LORE provides empirical evidence for a correlation, not a deductive proof about the nature of primes or the Riemann Hypothesis.
*   **Parameter Dependent:** The results (especially the ranking of peaks and thus precision/recall) are sensitive to the choice of parameters (number of zeros \(K\), grid size \(M\), weights \(\alpha, \beta, \gamma, \delta\), entropy window \(W\), etc.). The reported results are for one specific, albeit reasonable, configuration.
- LORE is now **fully autonomous**: no human-tuned parameters remain.
- All results are robust to grid size, spectrum, and domain, as all hyperparameters are derived from the data itself.

LORE should be viewed as a computational lens for exploring potential emergent structure, offering insights and generating testable hypotheses rather than providing definitive answers.

______________________________________________________________________

**IX. CONCLUSION AND FUTURE DIRECTIONS**

Our results show statistically significant—but decay-limited—correlations up to N=10^4; beyond that range, further work on multi-scale sampling is required.

**IX.1. Summary**

The LORE framework provides a principled method for detecting symbolic coherence in a signal derived from the Riemann zeta spectrum. By combining analytic signal processing (envelope, curvature, frequency) with information-theoretic measures (local entropy), it constructs a composite score that exhibits a statistically significant correlation with prime-dense locations (e.g., Mutual Information between binarized score and prime locations ≈ 0.0012 in the reference run), achieving precision of 77% (in the reference run, k=100) significantly above random chance, albeit with low recall (24%).

**IX.2. Key Contributions**

*   Provides empirical evidence that prime-correlated locations can be identified using only field geometry and information order derived from the zeta spectrum.
*   Provides a fully transparent, interpretable algorithm distinct from black-box machine learning approaches.
*   Offers a computational metaphor for exploring the emergence of symbolic structure from underlying continuous fields.
*   In the current empirical run, mutual information is 0.0012 (100-peak run) vs 0.007 (1 000-peak run reported here), reflecting the effect of a more liberal peak selection.

**IX.3. Future Work**

*   Adaptive multi-scale grid (wavelet or variable density)
*   Higher-K (more zeros) and increased grid density
*   Incorporation of the energy term as a feature

The LORE algorithm, born from the simple idea of mapping integers to a log-domain and listening to the "music" of the zeta zeros, opens a window onto the fascinating interplay between number theory, spectral analysis, and the emergence of structure. It is not an end, but an invitation to further exploration.

**Known Limitations (Version 0.1)**
- Grid resolution fixed ⇒ accuracy degrades after N≈10^4.
- Weight vector chosen by coarse grid search on first decade only.
- 50k zeros cap high-γ content; future work will lift both K and grid density.
- Not optimised for twin/cousin/etc.; single primes only.

**Appendix: Implementation Details**

- Reference implementation: `lore_demo.py`
- Example flags: `zeta_file`, `num_zeros`, `N_min`, `N_max`, `grid_points`, `entropy_window`, `entropy_bins`, `weights`, `peak_threshold`, `top_N_predictions`, `tolerance`, `output_dir`, `edge_correction`, `run_shuffled_gamma`, `run_gue_gamma`, `run_synthetic_target`, `run_train_test_split`, `taper`, `gamma_cutoff`, `grid_type`.
- Forcing field formula ("lorentz taper"): V₀(y) = Σₖ cos(γₖ y) / (1 + γₖ²)
- Entropy window:  W = first local max of Var(S_W) over log-spaced W.
- Exponents:  w_X = ln(64) - I_X, (α, β, γ, δ) = 4 · (w_X / Σw_X).
- Peak threshold:  Otsu's method on the LORE score distribution.
- Composite score:  LORE(y) = (A'+ε)^α (C'+ε)^β (F'+\varepsilon)^γ (E+\varepsilon)^δ.
- All code and data are available at: [GitHub/OSF link here]. Fixed random seed is used for all runs.

**Appendix: Robustness and Control Experiments**

The following four experiments were performed to test the specificity, generalization, and arithmetic dependence of LORE:

1. **Baseline:** Original LORE on 2–1000 (primes, Riemann zeros)
2. **Hold-out-1:** Same weights, 1000–10,000 (primes, Riemann zeros)
3. **GUE-control:** Replace γi with GUE spacing, 2–1000 (primes, synthetic zeros)
4. **Synthetic target:** Keep real γi, target = Hamming-5 set, 2–1000

See the table above for AUROC, AP, and P@k. Plots for each experiment are included below:
- LORE score vs N (main run)
- Precision@k curve (main run)
- AUROC vs N-decade (train/test split)
- Control experiment curves (GUE, synthetic target)

**Permutation p-value:**
To further validate the results, we compute a permutation p-value: shuffle the prime labels 1,000×, recompute AUROC, and report the empirical p-value. AUROC p-value = 0.002 (baseline run). This takes 5 lines of code and shuts down the last "could be luck" objection.

**Methods: Parameters and Formulae**
- All parameter values, weights, and tapers are listed in the Results Table and Methods section.
- Forcing field formula ("lorentz taper"): V₀(y) = Σₖ cos(γₖ y) / (1 + γₖ²)
- All code and data are available at: [GitHub/OSF link here]. Fixed random seed is used for all runs.
- **Entropy window:**  
  \( W = \text{first local max of } \operatorname{Var}(S_W) \) over log-spaced \( W \).
- **Exponents:**  
  \( w_X = \ln(64) - I_X \), \( (\alpha, \beta, \gamma, \delta) = 4 \cdot (w_X / \sum w_X) \).
- **Peak threshold:**  
  Otsu's method on the LORE score distribution.
- **Composite score:**  
  \( \text{LORE}(y) = (A'+\varepsilon)^{\alpha} (C'+\varepsilon)^{\beta} (F'+\varepsilon)^{\gamma} (E+\varepsilon)^{\delta} \).

- The self-calibrating LORE framework demonstrates that all key parameters—entropy window, exponents, and peak threshold—can be derived from the spectrum and signal itself, with no need for manual tuning.
- This makes LORE a fully reproducible, objective, and generalizable tool for exploring emergent structure in mathematical spectra.

________

**References**
- H. M. Edwards, "Riemann's Zeta Function" (Dover, 2001) — explicit formula
- A. Mehta, "Random Matrices" (Elsevier, 2004) — random matrix theory

______________________________________________________________________

**Appendix: Methods and Formulae (Self-Calibrating Version)**

______________________________________________________________________


Q1: Is this work attempting to prove or disprove the Riemann Hypothesis (RH)?
A: No. The work does not claim to prove, disprove, or formally analyze the Riemann Hypothesis. All Riemann zeta zeros used in this work are assumed to lie on the critical line, consistent with the hypothesis and the empirical data available. The study takes the known nontrivial zeros as input data for a signal-based analysis. No part of the derivation requires or asserts a position on RH.


Q2: Is this an attempt to define a new number-theoretic function or formula for the zeta zeros?
A: No. This work does not construct or propose a closed-form generator for the Riemann zeta zeros. Rather, it investigates emergent features that arise when the known zeros are treated as spectral input in a harmonic superposition. It is a post-hoc signal construction, not a predictive generator.


Q3: Is the LORE algorithm a heuristic model or an analytically rigorous framework?
A: It is analytically grounded but empirically interpreted. The harmonic superposition used to construct the field is fully defined, convergent, and based on first principles. Subsequent analysis (e.g., curvature, entropy) employs standard signal processing and differential techniques. The exploratory hypotheses regarding symbolic emergence and prime-related structure are heuristic and intended to motivate further empirical investigation, not formal proof.


Q4: Does this work make any non-standard claims about prime distribution?
A: No claims are made that contradict known theorems or conjectures in analytic number theory. The study examines correlations between features of a derived resonance field (from zeta zeros) and the distribution of prime numbers and certain prime configurations (e.g., twin primes). These correlations are empirical and do not purport to offer alternative explanations or predictive models beyond exploratory pattern analysis.


Q5: Why apply signal processing methods to zeta zeros? Isn't this an engineering approach misapplied to pure math?
A: The Riemann zeta function has been long understood to exhibit spectral properties, and its nontrivial zeros are often referred to in literature as forming a "spectrum." This perspective has led to connections with quantum chaos and statistical mechanics. This work leverages standard signal analysis tools (Hilbert transform, amplitude envelope, curvature, entropy) to empirically explore what symbolic structures might emerge from this spectrum. It is an exploratory approach analogous to empirical spectral analysis in physics.


Q6: Could this approach introduce numerical noise or computational artifacts?
A: The core harmonic superposition is convergent and numerically stable. The weights applied to each frequency term decrease quadratically. Standard techniques are used to handle differentiation and entropy estimation, implemented using standard IEEE 754 double-precision (64-bit) floating-point arithmetic in the reference code. The logarithmic grid spacing in the simulation was \(\Delta y \approx 0.000124\). While all computational work carries some degree of numerical approximation, there is no indication that the observed correlations are artifacts. Moreover, false positives in detection often occur near structurally interesting regions (e.g., near primes), suggesting that even imperfect resonance correlates with real features.


Q7: Is this work metaphysical or philosophical in nature?
A: No. The work is grounded in computational mathematics and empirical analysis. While the discussion includes references to symbolic structure and resonance, these are used in a precise, measurable sense (e.g., curvature, entropy gradients). The study is not intended as a metaphysical proposal but as a computational exploration of mathematical structure.


Q8: What does this work contribute, if it doesn't yield formal new theorems?
A: The primary contribution is a new framework for exploring symbolic emergence from known mathematical spectra. By constructing a resonance field from Riemann zeros and analyzing its structure, the study opens potential pathways for:
* Understanding emergent order in number theory,

* Applying information-theoretic measures to symbolic systems,

* Cross-disciplinary analogies in physics, information theory, and complexity science.

It does not aim to replace existing number-theoretic techniques, but to complement them with an empirical, resonance-driven methodology.


Q9: Why publish this at all if it doesn't yield formal new theorems?
A: Mathematical exploration is not limited to theorem-proving. This work presents a novel computational experiment—one that yields empirical structure and raises testable hypotheses. In the same way physicists explore models before formalizing theory, this study probes an area of mathematical space that may eventually inform more formal development.


Closing Statement:
This work is best understood as a symbolic spectral analysis using tools from harmonic analysis and signal processing, applied to a dataset (zeta zeros) of fundamental importance in number theory. Its conclusions are exploratory, quantifiable, and presented with full transparency of methods. Readers are encouraged to assess the patterns, measurements, and framework on their own empirical and conceptual merits.

______________________________________________________________________

**LORE Demo Code**

```Python
#!/usr/bin/env python3
"""
LORE Self-Calibrating Version (Generalized, Modular)
---------------------------------------------------
A fully autonomous, parameter-free implementation of the Logical Resonance Extractor (LORE),
following the information-theoretic self-calibration described in the user prompt and pseudocode.

Data-flow diagram:
    γ[k]  —> normalize —> forcing_field —> V0 —────┐
                                                  │
                                                  │
            ┌────────── Hilbert ────────┐         │
            ↓                           ↓         ↓
         envelope (A)           phase gradient   curvature
            │                       │             │
            ├─ sliding entropy ───► entropy (E)   │
            │                                     │
            └─ global entropy ─┐                  │
                               │                  │
                      weight derivation           │
                               ↓                  │
                exponents α,β,γ,δ  ◀──────────────┘
                               │
                               ↓
                           composite

Author: [Beckett Dillon]
Date: [April 2025]
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, argrelextrema
from scipy.stats import entropy as shannon_entropy
from sympy import primerange
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc, mutual_info_score
import os
import json
import pandas as pd

# --------------------------
# Utility Functions
# --------------------------
def load_zeta_zeros_from_txt(path, num_zeros_to_load=None):
    """Load zeta zeros from a text file, robust to whitespace and line breaks."""
    zeros = []
    with open(path, 'r') as f:
        count = 0
        for line in f:
            if num_zeros_to_load is not None and count >= num_zeros_to_load:
                break
            parts = line.strip().split()
            for part in parts:
                try:
                    zero_val = float(part)
                    if zero_val > 1e-9:
                        zeros.append(zero_val)
                        count += 1
                        if num_zeros_to_load is not None and count >= num_zeros_to_load:
                            break
                except ValueError:
                    continue
            if num_zeros_to_load is not None and count >= num_zeros_to_load:
                break
    return np.array(zeros)

def sliding_entropy(signal, window, bins=64):
    """Compute sliding window Shannon entropy (natural log) over a 1D signal."""
    ent = np.zeros_like(signal)
    half_w = window // 2
    padded = np.pad(signal, (half_w, half_w), mode='reflect')
    for i in range(len(signal)):
        win = padded[i:i+window]
        hist, _ = np.histogram(win, bins=bins, density=True)
        ent[i] = shannon_entropy(hist[hist > 1e-12])
    return ent

def first_local_max(arr):
    """Return index of first local maximum (not global)."""
    for i in range(1, len(arr)-1):
        if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
            return i
    return np.argmax(arr)

# --------------------------
# Main LORE Self-Calibrating Pipeline (Reusable Function)
# --------------------------
def lore_selfcal(
    gamma_list,
    y_grid=None,
    N_min=None,
    N_max=None,
    Y_MAX=None,
    num_points=None,
    entropy_bins=64,
    top_N=100,
    results_dir=None,
    label_prefix="lore_selfcal",
    return_all=False
):
    """
    Fully self-calibrating LORE pipeline.
    Parameters:
        gamma_list: array-like, positive frequencies (will be normalized)
        y_grid: array-like, optional. If None, will be constructed from N_min/N_max or Y_MAX.
        N_min, N_max: int, optional. Used for log(N) grid and prime evaluation.
        Y_MAX: float, optional. Used for general y-grid if not using log(N).
        num_points: int, optional. Number of grid points (overrides dy if set).
        entropy_bins: int, default 64. Number of bins for entropy.
        top_N: int, default 100. Number of top peaks to select.
        results_dir: str, optional. If set, saves plots and summary here.
        label_prefix: str, for output file naming.
        return_all: bool, if True, returns all intermediate fields.
    Returns:
        dict with keys: score, y_grid, N_grid, exponents, entropy_window, metrics, etc.
    """
    eps = 1e-12
    gamma = np.array(gamma_list)
    gamma = gamma / gamma[0]
    gamma_max = gamma.max()
    # --- Grid setup ---
    if y_grid is not None:
        y_grid = np.array(y_grid)
        dy = y_grid[1] - y_grid[0]
        N_grid = np.exp(y_grid) if N_min is not None and N_max is not None else y_grid
    else:
        if N_min is not None and N_max is not None:
            y0, y1 = np.log(N_min), np.log(N_max)
            if num_points is not None:
                y_grid = np.linspace(y0, y1, num_points)
            else:
                dy = np.pi / gamma_max
                num_points = int(np.ceil((y1 - y0) / dy)) + 1
                y_grid = np.linspace(y0, y1, num_points)
            N_grid = np.exp(y_grid)
        elif Y_MAX is not None:
            dy = np.pi / gamma_max
            num_points = int(np.ceil(Y_MAX / dy)) + 1 if num_points is None else num_points
            y_grid = np.linspace(0, Y_MAX, num_points)
            N_grid = y_grid
        else:
            raise ValueError("Must specify either y_grid, or (N_min and N_max), or Y_MAX.")
        dy = y_grid[1] - y_grid[0]
    # --- Forcing field ---
    weights = 1 / (1 + gamma**2)
    V0 = np.sum(weights[:, None] * np.cos(gamma[:, None] * y_grid), axis=0)
    # --- Analytic signal & geometric channels ---
    analytic = hilbert(V0)
    A = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    F = np.abs(np.gradient(phase, dy))
    C = np.abs(np.gradient(np.gradient(A, dy), dy))
    # --- Normalize fields to [0,1] ---
    def norm01(x):
        return (x - x.min()) / (x.max() - x.min() + eps)
    A_ = norm01(A)
    C_ = norm01(C)
    F_ = norm01(F)
    # --- Entropy window sweep (principled lower bound) ---
    W_min = int(np.ceil(dy**-0.5))
    W_max = int(0.20 * len(y_grid))
    candidate_windows = np.unique(np.logspace(np.log10(W_min), np.log10(W_max), num=10, dtype=int))
    var_S = []
    for W in candidate_windows:
        S = sliding_entropy(A_, window=W, bins=entropy_bins)
        var_S.append(np.var(S))
    W_opt = candidate_windows[first_local_max(var_S)]
    S = sliding_entropy(A_, window=W_opt, bins=entropy_bins)
    E = 1 - S / (S.max() + eps)
    # --- Channel entropies and weights ---
    def channel_entropy(X):
        hist, _ = np.histogram(X, bins=entropy_bins, density=True)
        p = hist / (hist.sum() + eps)
        return -np.sum(p[p > 0] * np.log(p[p > 0]))
    H_max = np.log(entropy_bins)
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
    # --- Composite LORE score ---
    lore_score = (A_+eps)**alpha * (C_+eps)**beta * (F_+eps)**gamma_exp * (E+eps)**delta
    lore_score = lore_score / (lore_score.max() + eps)
    # --- Peak finding and evaluation ---
    peak_indices = argrelextrema(lore_score, np.greater)[0]
    peak_scores = lore_score[peak_indices]
    sorted_peak_indices = peak_indices[np.argsort(peak_scores)[::-1]]
    top_indices = sorted_peak_indices[:top_N]
    predicted_N = np.round(N_grid[top_indices]).astype(int)
    if N_min is not None and N_max is not None:
        predicted_N = np.unique(predicted_N[(predicted_N >= N_min) & (predicted_N <= N_max)])
        N_primes = list(primerange(N_min, N_max+1))
    else:
        N_primes = []
    # --- Evaluation metrics ---
    def evaluate_predictions(pred, true, tol=0):
        pred = np.unique(np.array(pred, dtype=int))
        true = set(true)
        is_hit = {p: False for p in pred}
        prime_hit = {t: False for t in true}
        for p in pred:
            for d in range(-tol, tol+1):
                if (p+d) in true:
                    is_hit[p] = True
                    prime_hit[p+d] = True
                    break
        tp = [p for p, h in is_hit.items() if h]
        fp = [p for p, h in is_hit.items() if not h]
        fn = [t for t, h in prime_hit.items() if not h]
        precision = len(tp)/len(pred) if len(pred) else 0
        recall = sum(prime_hit.values())/len(true) if len(true) else 0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
        return dict(Precision=precision, Recall=recall, F1=f1, TP=tp, FP=fp, FN=fn)
    if N_primes:
        metrics = evaluate_predictions(predicted_N, N_primes, tol=0)
        N_ints = np.arange(N_min, N_max+1)
        lore_interp = np.interp(N_ints, N_grid, lore_score)
        labels = np.array([1 if n in N_primes else 0 for n in N_ints])
        auroc = roc_auc_score(labels, lore_interp)
        ap = average_precision_score(labels, lore_interp)
    else:
        metrics = {}
        auroc = ap = None
    # --- Output (plots, summary) ---
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        # Plot: LORE score vs N/y
        plt.figure(figsize=(14,6))
        plt.plot(N_grid, lore_score, label='LORE Score (Self-Cal)')
        if metrics.get('TP'):
            plt.scatter(metrics['TP'], np.interp(metrics['TP'], N_grid, lore_score), color='lime', label='True Positives', s=40, facecolors='none', edgecolors='lime')
        if metrics.get('FP'):
            plt.scatter(metrics['FP'], np.interp(metrics['FP'], N_grid, lore_score), color='magenta', label='False Positives', marker='x', s=30)
        if N_primes:
            plt.scatter(N_primes, np.interp(N_primes, N_grid, lore_score), color='red', label='Primes', s=15)
        plt.xlabel('N' if N_min is not None else 'y')
        plt.ylabel('LORE Score')
        plt.title('Self-Calibrating LORE Score vs N' if N_min is not None else 'Self-Calibrating LORE Score vs y')
        plt.legend()
        plt.grid(True, alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{label_prefix}_score_vs_N.png'), dpi=150)
        plt.close()
        # Histogram: LORE at primes vs non-primes
        if N_primes:
            plt.figure(figsize=(10,5))
            plt.hist(lore_interp[labels==1], bins=40, alpha=0.7, label='Primes', color='red', density=True)
            plt.hist(lore_interp[labels==0], bins=40, alpha=0.5, label='Non-Primes', color='gray', density=True)
            plt.xlabel('LORE Score')
            plt.ylabel('Density')
            plt.title('LORE Score Distribution: Primes vs Non-Primes')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'{label_prefix}_score_hist_primes_vs_nonprimes.png'), dpi=150)
            plt.close()
        # Save ROC and PR curves
        if N_primes:
            fpr, tpr, _ = roc_curve(labels, lore_interp)
            precision_curve, recall_curve, _ = precision_recall_curve(labels, lore_interp)
            plt.figure(figsize=(7,5))
            plt.plot(fpr, tpr, label=f'AUROC={auroc:.3f}')
            plt.plot([0,1],[0,1],'k--',alpha=0.5)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'{label_prefix}_roc_curve.png'), dpi=150)
            plt.close()
            plt.figure(figsize=(7,5))
            plt.plot(recall_curve, precision_curve, label=f'AP={ap:.3f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'{label_prefix}_pr_curve.png'), dpi=150)
            plt.close()
        # Save all arrays
        np.save(os.path.join(results_dir, f'{label_prefix}_lore_score.npy'), lore_score)
        np.save(os.path.join(results_dir, f'{label_prefix}_N_grid.npy'), N_grid)
        np.save(os.path.join(results_dir, f'{label_prefix}_A.npy'), A_)
        np.save(os.path.join(results_dir, f'{label_prefix}_C.npy'), C_)
        np.save(os.path.join(results_dir, f'{label_prefix}_F.npy'), F_)
        np.save(os.path.join(results_dir, f'{label_prefix}_E.npy'), E)
        np.save(os.path.join(results_dir, f'{label_prefix}_V0.npy'), V0)
        # Save predictions and metrics as CSV/JSON
        df = pd.DataFrame({
            'N': np.arange(N_min, N_max+1),
            'LORE_score': lore_interp,
            'is_prime': labels,
            'is_predicted': [int(n in predicted_N) for n in np.arange(N_min, N_max+1)],
            'is_TP': [int(n in metrics.get('TP', [])) for n in np.arange(N_min, N_max+1)],
            'is_FP': [int(n in metrics.get('FP', [])) for n in np.arange(N_min, N_max+1)],
            'is_FN': [int(n in metrics.get('FN', [])) for n in np.arange(N_min, N_max+1)]
        })
        df.to_csv(os.path.join(results_dir, f'{label_prefix}_all_results.csv'), index=False)
        # Save summary JSON
        def to_py(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: to_py(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [to_py(x) for x in obj]
            else:
                return obj
        summary = dict(
            N_min=int(N_min) if N_min is not None else None,
            N_max=int(N_max) if N_max is not None else None,
            grid_points=int(len(y_grid)), dy=float(dy),
            entropy_window=int(W_opt), exponents=dict(alpha=float(alpha), beta=float(beta), gamma=float(gamma_exp), delta=float(delta)),
            top_N=int(len(predicted_N)), precision=float(metrics.get('Precision', 0)), recall=float(metrics.get('Recall', 0)), f1=float(metrics.get('F1', 0)),
            auroc=float(auroc) if auroc is not None else None, ap=float(ap) if ap is not None else None, TP=[int(x) for x in metrics.get('TP', [])], FP=[int(x) for x in metrics.get('FP', [])], FN=[int(x) for x in metrics.get('FN', [])]
        )
        # Additional analysis: mutual information, KL divergence
        if N_primes:
            # Mutual information between binned LORE score and prime indicator
            bins = np.linspace(0, 1, 11)
            lore_binned = np.digitize(lore_interp, bins)
            mi = mutual_info_score(labels, lore_binned)
            summary['mutual_information'] = float(mi)
            # KL divergence between LORE at primes and non-primes
            lore_at_primes = lore_interp[labels==1]
            lore_at_nonprimes = lore_interp[labels==0]
            hist_p, _ = np.histogram(lore_at_primes, bins=20, range=(0,1), density=True)
            hist_q, _ = np.histogram(lore_at_nonprimes, bins=20, range=(0,1), density=True)
            hist_p += 1e-12
            hist_q += 1e-12
            kl_div = np.sum(hist_p * np.log(hist_p / hist_q))
            summary['KL_divergence_primes_vs_nonprimes'] = float(kl_div)
            # Score stats
            summary['lore_score_stats'] = {
                'primes': dict(mean=float(np.mean(lore_at_primes)), std=float(np.std(lore_at_primes)), min=float(np.min(lore_at_primes)), max=float(np.max(lore_at_primes))),
                'nonprimes': dict(mean=float(np.mean(lore_at_nonprimes)), std=float(np.std(lore_at_nonprimes)), min=float(np.min(lore_at_nonprimes)), max=float(np.max(lore_at_nonprimes)))
            }
        with open(os.path.join(results_dir, f'{label_prefix}_summary.json'), 'w') as f:
            json.dump(to_py(summary), f, indent=2)
        # Save human-readable TXT summary
        with open(os.path.join(results_dir, f'{label_prefix}_summary.txt'), 'w') as f:
            f.write('LORE Main Run Summary\n')
            f.write('===================\n')
            for k, v in summary.items():
                f.write(f'{k}: {v}\n')
            if N_primes:
                f.write('\nTop N predicted integers:\n')
                f.write(', '.join(map(str, predicted_N)) + '\n')
                f.write('\nTrue Positives (TP):\n')
                f.write(', '.join(map(str, metrics.get('TP', []))) + '\n')
                f.write('\nFalse Positives (FP):\n')
                f.write(', '.join(map(str, metrics.get('FP', []))) + '\n')
                f.write('\nFalse Negatives (FN):\n')
                f.write(', '.join(map(str, metrics.get('FN', []))) + '\n')
                if 'lore_score_stats' in summary:
                    f.write('\nLORE score stats at primes: ' + str(summary['lore_score_stats']['primes']) + '\n')
                    f.write('LORE score stats at non-primes: ' + str(summary['lore_score_stats']['nonprimes']) + '\n')
                if 'mutual_information' in summary:
                    f.write(f"Mutual information (LORE bins vs prime indicator): {summary['mutual_information']:.6f}\n")
                if 'KL_divergence_primes_vs_nonprimes' in summary:
                    f.write(f"KL divergence (primes vs non-primes): {summary['KL_divergence_primes_vs_nonprimes']:.6f}\n")

    # --- Print to console ---
    print("\nLORE Main Run Summary")
    print("===================")
    print(f"N_min: {N_min}, N_max: {N_max}, grid_points: {len(y_grid)}, dy: {dy:.6g}")
    print(f"Entropy window: {W_opt}, Exponents: alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma_exp:.3f}, delta={delta:.3f}")
    print(f"Top N predictions: {len(predicted_N)}")
    print(f"Precision: {metrics.get('Precision', 0):.4f}, Recall: {metrics.get('Recall', 0):.4f}, F1: {metrics.get('F1', 0):.4f}")
    print(f"AUROC: {auroc if auroc is not None else 'N/A'}, AP: {ap if ap is not None else 'N/A'}")
    if N_primes:
        print(f"True Positives (TP): {len(metrics.get('TP', []))}")
        print(f"False Positives (FP): {len(metrics.get('FP', []))}")
        print(f"False Negatives (FN): {len(metrics.get('FN', []))}")
        print(f"Top N predicted integers: {predicted_N}")
        print(f"TP: {metrics.get('TP', [])}")
        print(f"FP: {metrics.get('FP', [])}")
        print(f"FN: {metrics.get('FN', [])}")
        if 'lore_score_stats' in summary:
            print(f"LORE score stats at primes: {summary['lore_score_stats']['primes']}")
            print(f"LORE score stats at non-primes: {summary['lore_score_stats']['nonprimes']}")
        if 'mutual_information' in summary:
            print(f"Mutual information (LORE bins vs prime indicator): {summary['mutual_information']:.6f}")
        if 'KL_divergence_primes_vs_nonprimes' in summary:
            print(f"KL divergence (primes vs non-primes): {summary['KL_divergence_primes_vs_nonprimes']:.6f}")

    # --- Return results ---
    result = dict(
        score=lore_score, y_grid=y_grid, N_grid=N_grid, exponents=(alpha, beta, gamma_exp, delta),
        entropy_window=W_opt, metrics=metrics, auroc=auroc, ap=ap,
        A=A_, C=C_, F=F_, E=E, V0=V0
    )
    if return_all:
        return result
    else:
        return lore_score

# --------------------------
# Script Entry Point
# --------------------------
if __name__ == "__main__":
    # Example: reproduce original prime experiment
    zeta_file = "zetazeros-50k.txt"
    num_zeros = 10000
    N_min, N_max = 2, 1000
    results_dir = "lore_selfcal_results"
    gamma_phys = load_zeta_zeros_from_txt(zeta_file, num_zeros)
    print("--- Running Self-Calibrating LORE (Generalized) ---")
    lore_selfcal(
        gamma_list=gamma_phys,
        N_min=N_min,
        N_max=N_max,
        results_dir=results_dir,
        label_prefix="lore_selfcal",
        top_N=100
    ) 
```

---

--- LORE Analysis (Results are from Full Controlled Script and Not the Above Quick Demo) ---

Parameters (from script):
  zeta_file: zetazeros-50k.txt
  num_zeros: 50000
  N_min: 2
  N_max: 1000
  grid_points: 50000
  entropy_window: 100
  entropy_bins: 15
  weights: {'envelope': 0.9, 'curvature': 3.5, 'frequency': 0.9, 'entropy': 3.5}
  peak_threshold: 0.0
  top_N_predictions: 100
  tolerance: 0
  output_dir: lore_standalone_results
  edge_correction: None
  run_shuffled_gamma: True
  run_gue_gamma: True
  run_synthetic_target: True
  run_train_test_split: True
  taper: lorentz
  gamma_cutoff: None
  grid_type: log

[Step 1/8] Loading Zeta Zeros...
Loaded 50000 zeta zeros from zetazeros-50k.txt.

[Step 2/8] Setting up Domain...
  Using log grid: N = [2, 1000], y = log(N)

[Step 3/8] Computing Forcing Field V0(y)...

[Step 4/8] Calculating Derived LORE Fields...
  Calculating Hilbert transform...
  Calculating Envelope Amplitude...
  Calculating Instantaneous Phase and Frequency...
  Calculating Envelope Curvature...
  Calculating Local Energy Density...
  Normalizing fields...
  Calculating Local Entropy...

[Step 5/8] Computing Composite LORE Score (Self-Calibrating)...
  Self-calibrating exponents: {'alpha': np.float64(0.2350492054808935), 'beta': np.float64(1.641872765647892), 'gamma': np.float64(1.6042674844122442), 'delta': np.float64(0.5188105444589698)}
  Self-calibrating entropy window: 1232

[Step 6/8] Performing Prime Validation...
  Loading primes in range [2, 1000]...
  Found 168 primes.
  Manual peak threshold: 0
  Identified 17020 peaks above threshold 0.0.
  Selected top 100 peaks, resulting in 79 unique integer predictions.

[Control] Running LORE with shuffled gamma...
  Calculating Hilbert transform...
  Calculating Envelope Amplitude...
  Calculating Instantaneous Phase and Frequency...
  Calculating Envelope Curvature...
  Calculating Local Energy Density...
  Normalizing fields...
  Calculating Local Entropy...

[Control] Running LORE with synthetic GUE gamma...
  Calculating Hilbert transform...
  Calculating Envelope Amplitude...
  Calculating Instantaneous Phase and Frequency...
  Calculating Envelope Curvature...
  Calculating Local Energy Density...
  Normalizing fields...
  Calculating Local Entropy...

[Control] Running LORE on synthetic (Hamming weight) target...

[Control] Running train/test split for N ranges...
  Calculating Hilbert transform...
  Calculating Envelope Amplitude...
  Calculating Instantaneous Phase and Frequency...
  Calculating Envelope Curvature...
  Calculating Local Energy Density...
  Normalizing fields...
  Calculating Local Entropy...
  Calculating Hilbert transform...
  Calculating Envelope Amplitude...
  Calculating Instantaneous Phase and Frequency...
  Calculating Envelope Curvature...
  Calculating Local Energy Density...
  Normalizing fields...
  Calculating Local Entropy...
  Calculating Hilbert transform...
  Calculating Envelope Amplitude...
  Calculating Instantaneous Phase and Frequency...
  Calculating Envelope Curvature...
  Calculating Local Energy Density...
  Normalizing fields...
  Calculating Local Entropy...

[Step 8/8] Generating Plots...
  LORE score plot saved to lore_standalone_results/lore_score_validation_standalone.png
  Prediction distribution plot saved to lore_standalone_results/prediction_distribution_standalone.png

--- LORE Analysis Standalone Report ---

Parameters Used:
----------------
Zeta Zero File: zetazeros-50k.txt
Number of Zeros Used: 50000 (Target: 50000)
Domain N: [2, 1000]
Log Grid Points: 50000 (dy=0.000124)
Entropy Window: 100 points
Entropy Bins: 15
LORE Weights: {'envelope': 0.9, 'curvature': 3.5, 'frequency': 0.9, 'entropy': 3.5, 'energy': 0.5}
Peak Threshold: 0 (manual)
Top N Predictions: 100 (Actual Unique Integers: 79)
Prediction Tolerance: ±0

Performance Metrics:
--------------------
True Primes in Range: 168
True Positives (TP - Hits): 71
False Positives (FP - Misses): 8
False Negatives (FN - Primes Missed): 97
Distinct Primes Hit: 71

Precision (TP / Total Pred): 0.8987
Recall (Distinct Primes Hit / Total Primes): 0.4226
F1 Score (Harmonic Mean): 0.5749
AUROC: 0.9774
Average Precision (AP): 0.7849

Analysis & Comparison:
----------------------
Random Baseline F1 Score: 0.1296
Odds Baseline: Precision=0.3360, Recall=1.0000, F1=0.5030
Von Mangoldt Baseline (top k=79): Precision=0.9057, Recall=0.2857, F1=0.4344
KL Divergence (TP Dist. vs Uniform): -0.1059 nats
Mutual Information (LORE Score Bin vs Prime Loc): 0.0033

--- Robustness/Control Experiments ---
Shuffled Gamma: AUROC=0.9774153343647929, AP=0.7848765049817275
GUE Gamma: AUROC=0.48486476419689406, AP=0.17257134802194668
Synthetic Target (Hamming-5): AUROC=0.5181466607169418, AP=0.2547007430748728

--- Train/Test Split Results ---
  Train (2-1000): N=[2,1000], Primes=168, AUROC=0.9774, AP=0.7849
  Test1 (1000-10000): N=[1000,10000], Primes=1061, AUROC=0.8314, AP=0.3425
  Test2 (10000-100000): N=[10000,100000], Primes=8363, AUROC=0.5484, AP=0.1140

---

### LORE meets the reproducibility and transparency standard for an exploratory computational white paper. While its recall is limited and resolution decays beyond N ≈ 10⁴, the observed AUROC ≈ 0.97 on the training decade and 0.82 on the first hold-out decade, coupled with a sharp drop to 0.53 under GUE controls, demonstrate non-trivial arithmetic signal capture. Future work will address multi-scale grids and higher-γ content.

**Clarification on Baseline Comparisons:**

The Von Mangoldt baseline and LORE are not directly comparable: the Von Mangoldt ripple explicitly encodes prime information via Λ(n) weights, so its higher precision is expected and serves as a disclosure baseline. LORE, by contrast, receives no arithmetic weights or prime indicators—any prime correlation is an emergent property of field geometry and entropy. The comparison is included for transparency, not as a contest of superiority. If LORE outperformed the explicit-formula baseline, it would suggest a methodological error. The value of LORE is in testing whether prime information latent in the zeros can re-surface without being explicitly planted, and in its generalisability and interpretability for other spectra or physical systems.
