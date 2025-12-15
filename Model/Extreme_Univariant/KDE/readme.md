# Kernel Density Estimation for Anomaly Detection (KDE)

---

## Table of Contents

This article focuses specifically on one of the second category: Kernel Density Estimation (KDE) — a non-parametric method used to estimate the probability density function of a dataset.
- you can imagine KDE as an advanced top/bottom 1% Outlier detection.
- Follow this [notebook](https://colab.research.google.com/drive/1qC-Gry8py_Icl0V8zNedlIeX3HFEKHuY#scrollTo=pvMDx5tXhGd0) for a step-by-step explanation of the article.


1. [Understanding KDE: From Business Case to Mathematical Intuition](#understanding-kde-from-business-case-to-mathematical-intuition)  
   - [2.1 Business Use Case](#21-business-use-case-detecting-anomalies-in-signal-to-noise-ratio-snr)  
   - [2.2 Theoretical Foundation](#22-theoretical-foundation-estimating-the-data-generator)  
   - [2.3 From Histogram to KDE](#23-from-histogram-to-kde-smoothing-the-distribution)  
   - [2.4 Mathematical Intuition](#24-mathematical-intuition-behind-kde)  
3. [KDE Parameters](#kde-parameters)  
   - [3.1 Kernel Function](#31-kernel-function)  
   - [3.2 Bandwidth Parameter](#32-bandwidth-parameter)  
4. [KDE-Based Anomaly Detection: Step-by-Step Guide](#kde-based-anomaly-detection-step-by-step-guide)  
5. [Threshold Selection in KDE Anomaly Detection](#threshold-selection-in-kde-anomaly-detection)  
6. [Limitations of KDE](#limitations-of-kde)  
7. [Conclusion](#conclusion)  
8. [Resources](#resources)

---


---

## Understanding KDE: From Business Case to Mathematical Intuition

### 2.1 Business Use Case: Detecting Anomalies in Signal-to-Noise Ratio (SNR)

Suppose you have time series measurements of **Signal-to-Noise Ratio (SNR)** and want to determine whether the latest value is anomalous compared to history.

<img width="1400" height="552" alt="image" src="https://github.com/user-attachments/assets/337524c0-915e-44d3-9e43-52273f5e81b7" />


Instead of using rigid thresholds like the 1st or 99th percentile, KDE provides a more **flexible, data-driven** approach by estimating where values are **dense** (common) vs. **sparse** (rare).

### 2.2 Theoretical Foundation: Estimating the Data Generator

KDE starts with the assumption that your observed data points are samples from an unknown underlying probability distribution — a “data generator” you want to estimate.

A simple analogy:
- Throwing a die produces outcomes from a hidden generator (the die).
- To judge if a new value is “weird,” you first need to understand the generator behind past values.

### 2.3 From Histogram to KDE: Smoothing the Distribution


A histogram estimates a distribution by counting points within bins. But histograms are **sensitive** to:
- Bin widths
- Bin alignments (small shifts can change the shape a lot)

<img width="1400" height="1046" alt="image" src="https://github.com/user-attachments/assets/62ff429d-e865-41c9-8216-a1077e3373ab" />


KDE fixes this by:
- Placing a **kernel** (a small “bump” or “block”) on **every data point**
- Summing these kernels to form a smooth density estimate

You can think of the progression like:
- Histogram (discrete, bin-based)
- Tophat kernels (block smoothing)
- Gaussian kernels (smooth, continuous density)

### 2.4 Mathematical Intuition Behind KDE

The KDE density at a location \(y\) is the sum of contributions from each observed point \(x_i\).  
Each point adds a small “bump,” and KDE aggregates these bumps into a continuous density estimate.

At a high level:
- **More nearby points** → higher density
- **Few nearby points** → lower density (potential anomaly region)

---

## KDE Parameters

KDE is a **non-parametric** model:

- **Non-parametric ≠ no parameters**
- It means **no assumption about a fixed distributional form** (e.g., not forcing Gaussianity)
- Complexity grows with the data: **more data → more structure**

### ✅ Parametric vs. Non-parametric (Quick Contrast)

**Parametric models** (e.g., Gaussian distribution, linear regression):
- Assume a fixed functional form
- Have a fixed number of parameters (e.g., Gaussian has \(\mu\) and \(\sigma\))

**Non-parametric models** (e.g., KDE, k-NN, decision trees):
- Do not assume a global distribution shape
- “Learn structure” by accumulating patterns from data

### 3.1 Kernel Function

The **kernel** is not an assumption about the data’s distribution.  
It’s the **smoothing mechanism** describing how each data point contributes to the final density.

Analogy: **Paintbrushes, not templates**
- Parametric model: you pick a stencil (“everything is Gaussian”)
- KDE: you paint a soft brushstroke at every point (Gaussian, tophat, triangular, etc.)
- The final shape comes from all brushstrokes combined

|  |  |
|---|---|
| <img src="https://github.com/user-attachments/assets/07cb909f-3402-4bef-be01-4fd7f48c7ab1" width="420" /> | <img src="https://github.com/user-attachments/assets/310a5b2a-0b1f-49fc-941f-47577a0aa9c3" width="420" /> |


<img width="1400" height="834" alt="image" src="https://github.com/user-attachments/assets/d4ba4c5e-6a52-43ec-b97a-7f9d10eb0d85" />


### 3.2 Bandwidth Parameter

**Bandwidth** controls the bias–variance trade-off:

- Large bandwidth → smoother density (higher bias)
- Small bandwidth → more wiggly density (higher variance)

Bandwidth is often the **most important** hyperparameter for KDE anomaly detection.

<img width="1400" height="834" alt="image" src="https://github.com/user-attachments/assets/1d299355-eddf-4fd6-a034-47fd6ccc7d43" />


---

## KDE-Based Anomaly Detection: Step-by-Step Guide

Below is a simple step-by-step workflow using `sklearn.neighbors.KernelDensity`.

### 1) Fit the KDE model

```python
from sklearn.neighbors import KernelDensity

kde = KernelDensity(kernel="gaussian", bandwidth=0.5)
kde.fit(X_train)  # X_train shape: (n_samples, 1) or (n_samples, n_features)
```

### 2) Visualize the KDE curve (optional but recommended)

```python
import numpy as np

x_vals = np.linspace(X_train.min() - 1, X_train.max() + 1, 1000).reshape(-1, 1)
log_dens = kde.score_samples(x_vals)
dens = np.exp(log_dens)
```

<img width="1400" height="693" alt="image" src="https://github.com/user-attachments/assets/83b25807-fc84-4be6-9a2c-d0d60ae34eac" />


### 3) Compute training densities and choose a density threshold

```python
log_dens_train = kde.score_samples(X_train)
dens_train = np.exp(log_dens_train)

threshold = np.quantile(dens_train, 0.01)  # bottom 1% density cutoff
```

### 4) Score new observations and flag anomalies

```python
log_dens_new = kde.score_samples(X_test)
dens_new = np.exp(log_dens_new)[0]

prediction = "Novelty" if dens_new < threshold else "Normal"
```

---

## Threshold Selection in KDE Anomaly Detection

### 5.1 Why KDE Works for Anomaly Detection

KDE outputs a density estimate: “How likely is a value, given the learned distribution?”

A point in a **low-density** region is rare under the estimated distribution → likely anomaly.

### 5.2 Importance of Thresholds

KDE itself produces a smooth density function, but you still need a rule for decision-making:

- density < threshold → anomaly
- density ≥ threshold → normal

This is similar to z-scores, but instead of distance from the mean in standard deviations, you use **probability density**.

### 5.3 Choosing the Threshold

A practical approach:
1. Compute densities for all training points
2. Choose a **quantile cutoff** (e.g., 1st percentile)

You can tune the percentile based on:
- False positive tolerance
- Desired sensitivity
- Business costs of misses vs. false alarms

---

## Limitations of KDE

### 6.1 KDE Fails to Capture Seasonality and Temporal Context

KDE assumes data is **i.i.d.** and ignores time dependencies.

For time series with seasonality (hour-of-day, day-of-week patterns), KDE may flag values as anomalies that are normal for a specific time context.

**Mitigations (common approaches):**
- Run KDE separately per time bucket (e.g., hour-of-day KDE)
- Add time/context features (if using multivariate KDE)
- Use sequence-aware models for temporal anomalies

### 6.2 Sensitivity to Narrow Variance

<img width="1343" height="783" alt="image" src="https://github.com/user-attachments/assets/7362f803-9322-43ab-a905-4330372be7ba" />


When the data distribution is extremely tight, small deviations can be pushed into low-density regions and get flagged as anomalies.



### 6.3 Training Contamination (Anomalies in Training Data)

<img width="1400" height="553" alt="image" src="https://github.com/user-attachments/assets/e0b49a3a-d815-45e1-8546-a8d0176f208d" />


If anomalies are present in training data, KDE will learn them as normal:
- tails become heavier
- density threshold becomes less strict
- detecting future anomalies becomes harder


### 6.4 Sensitivity to Extreme Outliers

<img width="1400" height="706" alt="image" src="https://github.com/user-attachments/assets/a235267b-5d2b-4edb-8d8b-83d89a662c29" />


Because KDE places a kernel on every point, extreme outliers can distort the estimated density shape and threshold, even if those outliers are spurious.


---

## Conclusion

Kernel Density Estimation transforms anomaly detection from “guessing what might be rare” into estimating what *truly is rare* under the learned distribution.

By modeling the density in feature space:
- **High density** → common/normal behavior  
- **Low density** → rare behavior → candidate anomalies  

The key operational decision is the **density threshold** (often a bottom-quantile cutoff).  
KDE works especially well when the data doesn’t follow neat parametric distributions, but it can be computationally heavy and sensitive to bandwidth — so tuning and careful training data selection matter.

---

## Resources

- **GitHub (code referenced in this article):**  
  `Anomaly_Detection_toolkit/KDE/FeaturewiseKDENoveltyDetector.py`

- **Colab notebook walkthrough:**  
  https://colab.research.google.com/drive/1qC-Gry8py_Icl0V8zNedlIeX3HFEKHuY#scrollTo=WXniSVfCznS_

---

### License / Usage Note (optional)
If you’re publishing this README in a repo, consider adding:
- a LICENSE file
- installation steps
- a minimal “Quick Start” example using your detector class
