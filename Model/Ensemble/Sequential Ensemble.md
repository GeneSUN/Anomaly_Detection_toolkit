# Ensemble Methods for Outlier Detection (2): Sequential Ensemble-Outlier Contamination and Model Selection

This article explores how **sequential ensemble methods** can reduce **bias** in unlabeled outlier detection by **iteratively pruning** suspected anomalies from training data.

Related articles:
- *Ensemble Methods for Outlier Detection* (from supervised learning to unsupervised learning)  
- *Ensemble Methods for Outlier Detection (1): Independent, Model-Centric Ensemble* (hyperparameter tuning for unsupervised learning)

---

## Table of Contents

- [Why ensemble?](#why-ensemble)
- [Bias reduction by data-centric pruning](#bias-reduction-by-data-centric-pruning)
  - [Iterative outlier removal](#iterative-outlier-removal)
  - [Algorithm: IterativeOutlierRemoval](#algorithm-iterativeoutlierremovaldata-set-d)
- [Example: Sequential pruning with an autoencoder](#example-sequential-pruning-with-an-autoencoder)
- [Conclusion](#conclusion)

---

## Why ensemble?

Outlier detection methods typically assume the model is trained on **normal** data.

<img width="800" height="318" alt="image" src="https://github.com/user-attachments/assets/b5da2126-b7ad-4fac-aab2-c12bfe90d4f8" />


In practice, training sets are often **contaminated** with outliers. When too many outliers are present:

- the model can “learn” anomalies as normal behavior,
- true outliers become harder to identify,
- detection results become biased and less accurate.

**Key motivation:**  
By removing contaminating outliers from the training set, we reinforce the assumption of clean training data, reduce bias, and improve detection accuracy.

<img width="800" height="391" alt="image" src="https://github.com/user-attachments/assets/b9ba75b2-4be0-4c69-a2b3-14631695a1c9" />


---

## Bias reduction by data-centric pruning

Bias often arises when a model’s assumptions don’t match reality (e.g., fitting a linear model to nonlinear data).

In outlier detection, a core assumption many models make is:

> The training data is clean (anomaly-free).

This assumption is rarely true. Training on contaminated data violates the assumption, producing biased results and reducing detection quality.

A common remedy is **iterative outlier removal**: repeatedly train a detector on progressively cleaner data.

---

### Iterative outlier removal

```
Algorithm: IterativeOutlierRemoval(Data Set D)
------------------------------------------------
1. Initialize D_current = D
2. Repeat:
      a. Apply algorithm A to D_current to build model M
      b. Score all points in D_current using M to identify outliers O
      c. Remove O from D_current
   Until convergence or reaching the maximum number of iterations
3. Return the set of detected outliers O
```
<img width="371" height="597" alt="image" src="https://github.com/user-attachments/assets/d54ce635-e8db-415b-a134-25f98689f9e2" />


This systematically reduces contamination in the training set, improving the model’s notion of “normal.”

---

### Algorithm: IterativeOutlierRemoval(Data Set D)

```text
Algorithm: IterativeOutlierRemoval(Data Set D)
------------------------------------------------
1. Initialize D_current = D
2. Repeat:
      a. Apply algorithm A to D_current to build model M
      b. Score all points in D_current using M to identify outliers O
      c. Remove O from D_current
   Until convergence or reaching the maximum number of iterations
3. Return the set of detected outliers O
```

---

## Example: Sequential pruning with an autoencoder

A [practical demonstration](https://colab.research.google.com/drive/1LqBRw-p1OCP7VJ0Qn4qAA3i2H58_UCky) uses an **autoencoder** for anomaly detection (see the related autoencoder article if needed):

Workflow described:

1. **Round 1:** Train on the full dataset (normal + outliers).  
   - Some samples show extremely high anomaly scores.
  <img width="800" height="376" alt="image" src="https://github.com/user-attachments/assets/205b75b7-18be-49bc-951b-de186c7740d0" />


  
2. **Prune:** Remove the highest-scoring outliers from the training data.
3. **Round 2:** Retrain the model on the cleaner dataset.  
   - Fewer outliers appear, and scores are less extreme.
    <img width="800" height="396" alt="image" src="https://github.com/user-attachments/assets/c5844a13-a543-4d10-919a-0e260314168a" />

     
4. **Re-score removed outliers:** Score the removed series using the Round 2 model and compare to Round 1.  
   - The separation between outliers and inliers becomes more pronounced in Round 2.
    
    <img width="708" height="470" alt="image" src="https://github.com/user-attachments/assets/4a99f445-0210-4849-9436-444fa632f892" />


Why this helps:

- When the model is trained, it assumes training data is clean and learns “normal patterns.”
- If many outliers are included, the model may treat them as normal → future true outliers may be missed.
- If only a few outliers exist, the model can better learn normal behavior → deviations are more likely to be flagged.

---

## Conclusion

Sequential ensemble methods provide a powerful remedy to the bias and instability of unsupervised outlier detection.

By iteratively pruning anomalies:

- the model’s notion of “normal” becomes more accurate,
- anomaly identification becomes more confident,
- results become easier to interpret and justify.

Beyond empirical benefits, this strategy advances practical unsupervised detection by:

- Adapting ensemble principles to **unlabeled** scenarios with no ground truth
- Reducing **bias** (removing contaminating outliers) and potentially **variance** (aggregating across iterations)
- Offering a blueprint for cleaner, more interpretable models in real-world noisy settings

This is especially relevant in applications such as **sensor monitoring**, **cybersecurity**, and **manufacturing**.
