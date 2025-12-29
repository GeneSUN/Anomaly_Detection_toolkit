# Ensemble Methods for Outlier Detection (1): Independent, Model-Centric Ensemble-Hyperparameter tuning for unsupervised learning


This article explores **independent/parallel ensemble approaches** for outlier detection in unsupervised learning:

- Proposes a **model-centric ensemble** by combining results across different configurations and/or different models  
- Provides a practical way to **boost reliability** and **reduce variance** when labels are unavailable

<img width="615" height="207" alt="image" src="https://github.com/user-attachments/assets/841db567-d714-4fe8-9d7c-aa6f4688a9ab" />


Related articles:

- *Ensemble Methods for Outlier Detection* (from supervised learning to unsupervised learning)  
- *Ensemble Methods for Outlier Detection (2): Sequential Ensemble* (outlier contamination and model selection)

A [Google Colab notebook](https://colab.research.google.com/drive/1npz391Gw1sC5XnYVy_1GsIqVlfrpttVM) is referenced to reproduce the workflow end-to-end.

---

## Table of Contents

- [Why ensemble in unsupervised learning?](#why-ensemble-in-unsupervised-learning)
- [1. Model-related ensemble methods](#1-model-related-ensemble-methods)
  - [1.1 Parametric ensembles](#11-parametric-ensembles)
  - [1.2 Randomized-initialization ensembles](#12-randomized-initialization-ensembles)
  - [1.3 Model ensembles](#13-model-ensembles)
- [2. Feature bagging: a data-centric ensemble](#2-feature-bagging-a-data-centric-ensemble)
- [3. Isolation Forests: an ensemble-centric view](#3-isolation-forests-an-ensemble-centric-view)
- [Summary](#summary)

---

## Why ensemble in unsupervised learning?

Selecting the best hyperparameters (or even the best model) is one of the major challenges in unsupervised learning.

- In **supervised learning**, you can compare hyperparameters using evaluation metrics against labels.
- In **unsupervised learning**, without labels, it’s difficult to know which setting is best.

A practical solution is to **ensemble**:

> Instead of betting on a single “best” setting, we leverage them all.

For example, you can run the *same* model with multiple hyperparameter settings and combine their scores to reduce instability and improve reliability.


---

## 1. Model-related ensemble methods

In a model-centric ensemble, base detectors can be built in different ways:

1. **Model ensembles:** use completely different algorithms  
2. **Parametric ensembles:** vary hyperparameters of one algorithm  
3. **Randomized-initialization ensembles:** rerun the same algorithm with different random seeds  

---

### 1.1 Parametric ensembles

<img width="720" height="207" alt="image" src="https://github.com/user-attachments/assets/1323735e-30cd-4353-9d04-eaa2d451021a" />


For clustering-based methods like **DBSCAN**, parameters such as `eps` and `min_samples` must be specified in advance. In many cases, finding optimal values is challenging, and best settings can vary across datasets, often requiring manual tuning.

A practical strategy:

- Run multiple DBSCAN configurations (or other parametric variants)
- Compute outlier scores from each run
- Combine them (commonly by averaging)

> Averaging outlier scores from several parameterized clusterings can reduce sensitivity to any single choice.

<img width="720" height="614" alt="image" src="https://github.com/user-attachments/assets/8608fdbd-be26-430f-9234-e946fb374ebf" />


Reference mentioned in the article:

- A. Emmott, S. Das, T. Dietterich, A. Fern, and W. Wong.  
  *Systematic Construction of Anomaly Detection Benchmarks from Real Data.* arXiv:1503.01158, 2015.

---

### 1.2 Randomized-initialization ensembles

<img width="720" height="182" alt="image" src="https://github.com/user-attachments/assets/8a38e78a-83f3-414e-b10a-c10accd99aef" />


Many algorithms (e.g., **k-means**) are sensitive to random initialization. Different random seeds can produce different cluster assignments and thus different outlier scores — a source of **model-centric variance**.

<img width="720" height="607" alt="image" src="https://github.com/user-attachments/assets/d722000d-b76d-4aed-ba6e-21c2253c8f76" />


A simple way to reduce this variance:

- run the same algorithm multiple times with different random seeds  
- aggregate scores (e.g., average)

This often improves stability because extreme runs are “smoothed out” by the ensemble.

---

### 1.3 Model ensembles

Different outlier detection models capture different anomaly patterns, each with strengths and limitations.




Example intuition described:

- A **density-based** detector may struggle with “far-but-tight” anomaly groups.  
  If anomalies cluster tightly, the region can still be high-density, and a density method might label them as normal.

<img width="720" height="624" alt="image" src="https://github.com/user-attachments/assets/4f502678-ed50-4294-8756-bc32e0fec945" />

> The outliers cluster together, knn would regard the high density area as normal

- A **cluster-based** detector can be sensitive to noise and may misclassify scattered points as anomalies or miss a compact anomalous cluster depending on how clustering behaves.


<img width="720" height="624" alt="image" src="https://github.com/user-attachments/assets/e3313d12-a3b3-4332-b963-c7400aae5f40" />

> clustering method failed to recognize the outlier, since there are other points have much higher score.


**Why ensemble helps:**  
By combining multiple detectors, one model’s blind spots may be covered by another model’s strengths — yielding a more stable, averaged performance across diverse anomaly scenarios.

<img width="720" height="605" alt="image" src="https://github.com/user-attachments/assets/2958416b-a6d3-4098-bd9f-8b87af3275a9" />

---

## 2. Feature bagging: a data-centric ensemble

Feature bagging is a data-centric ensemble approach, common in **high-dimensional** outlier detection.

Basic idea:

1. For a dataset with **D** dimensions, randomly pick **r** dimensions.
2. Run a base detector on this smaller feature subset.
3. Repeat multiple times.
4. Combine the outlier scores from each run (often **max** or **average**).

<img width="1100" height="341" alt="image" src="https://github.com/user-attachments/assets/f90a9234-594c-4040-a139-8fa910652180" />


This introduces diversity through different feature subspaces and can improve robustness when many features are noisy or redundant.

---

## 3. Isolation Forests: an ensemble-centric view

Isolation Forest can be viewed as an ensemble method (in spirit similar to Random Forests), with two independent layers of randomness:

- 1️⃣ Random sampling of data → which points a tree sees
- 2️⃣ Random splitting inside the tree → how isolation happens

It works in three main steps:

### 3.1 Random subsampling

Each isolation tree is built from a random subset of the data:
- reduces computation
- increases diversity across trees

<img width="720" height="165" alt="image" src="https://github.com/user-attachments/assets/70fe63d2-2dd7-4f95-a81a-a6ca2da77e7e" />


### 3.2 Recursive random partitioning

Each tree recursively splits data by:
- randomly choosing a feature
- randomly choosing a split value

<img width="720" height="139" alt="image" src="https://github.com/user-attachments/assets/7366e1db-a335-44c4-9956-23f66b50ed1d" />



Why it works for anomalies:
- anomalies often lie in sparse regions
- they are easier to isolate, so they require fewer splits

### 3.3 Path length averaging

For each point, compute its average path length across all trees (how many splits it takes to isolate the point).

- **Outliers** → shorter average path length (isolated early)
- **Normal points** → longer path length

<img width="546" height="435" alt="image" src="https://github.com/user-attachments/assets/e86de209-b621-41ea-a8df-a8eb53b3c58e" />



By averaging across many trees, Isolation Forest produces a robust anomaly score emphasizing points that consistently isolate early.

---

## Summary

Independent/parallel ensembles help counteract **model-centric variance** — instability due to arbitrary choices like:

- random initialization
- hyperparameter settings
- algorithm selection

By averaging results across these variations, ensembles reduce extreme errors and improve consistency — especially in unlabeled environments where finding a single perfect setup is impractical.
