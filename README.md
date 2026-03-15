# The Quad-Prior Genomic Engine

A high-efficiency, sub-million parameter biological sequence model that replaces standard deep learning heuristics with four mathematical frameworks derived from ancient Indian logic. 

Traditional Transformer architectures face profound bottlenecks in sequence modeling, specifically massive $O(N^2)$ attention scaling and dense matrix multiplication inefficiencies. This engine resolves those bottlenecks by hardcoding mathematically optimal priors into the network's architecture, demonstrating that structural understanding can replace brute-force arithmetic.

## The Four Mathematical Pillars

1. **Pingala Layer (Sequence Bounding):** Derived from the *Chandahshastra* (c. 200 BCE). Enforces strict local attention bounds using Meru Prastara binomial distributions to map immediate, short-range chemical motifs (e.g., covalent bonding).
2. **Panini Layer (Positional Grammar):** Modeled after the *Ashtadhyayi* (c. 500 BCE). Uses a low-rank bucketed factorization to dynamically enforce positional syntax, treating biological sequences as context-sensitive languages.
3. **Ramanujan Layer (Long-Range Memory):** Uses Ramanujan's partition identities to create a sparse attention mask. Attention heads cast connections at logarithmically expanding intervals, solving the $O(N^2)$ context bottleneck to map 3D protein folding.
4. **Urdhva Layer (Compute Efficiency):** Implements the Vedic sutra *Urdhva-Tiryagbhyam* ("Vertically and Crosswise"). By dividing embedding dimensions into discrete blocks and computing only direct and adjacent block interactions, it slashes the parameter count and FLOPs.

## Benchmark Results

### 1. Architectural Efficiency
By swapping standard dense linear networks for the Urdhva Vedic linear layers, the hardware compute requirement was drastically reduced without losing expressivity.
* **Baseline Parameters:** ~3.2 Million
* **Quad-Prior Parameters:** 872,761
* **Hardware Compute Saved:** 72.7%

### 2. Targeted Oncology (TP53 Somatic Mutation)
The engine was trained exclusively on healthy mammalian TP53 sequences and evaluated zero-shot on Human TP53 with an injected lethal point mutation.
* **Wild-Type (Healthy) Surprise:** 0.3606
* **Mutant (Cancer) Surprise:** 1.4988
* **Detection Spike:** 315.6%
* **Conclusion:** Successfully identified the structural collapse of the DNA-binding domain.
* *Run it:* `python benchmark_oncology.py`

### 3. The Trojan Horse Test (HIV Viral Integration)
Tested the engine's ability to detect horizontal gene transfer by splicing the HIV-1 genome into a healthy Human chromosome. 
* **Deep Human Territory Surprise:** 1.2817
* **Deep Alien (HIV) Territory Surprise:** 1.5823
* **Grammar Shift:** 23.4%
* **Conclusion:** The engine successfully detected the dialect shift from host DNA to viral payload.
* *Run it:* `python benchmark_trojan.py`

## Usage & Requirements
* PyTorch
* NumPy
* SciPy
