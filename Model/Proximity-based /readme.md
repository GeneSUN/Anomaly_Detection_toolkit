# Multivariate and Proximity-Based Detection: K-means, KNN, LOF, and DBSCAN


---

## Table of Contents

1. [Proximity-based outlier detection (high-level idea)](#proximity-based-outlier-detection-high-level-idea)
2. [Introduction](#1-introduction)
   - [Cluster-Based Methods](#1a-cluster-based-methods)
   - [Distance-Based Methods](#1b-distance-based-methods)
   - [Density-Based Methods](#1c-density-based-methods)
3. [Comparison](#2-comparison)
   - [Global vs. local analysis](#global-vs-local-analysis)
4. [Optimization considerations](#3-optimization-considerations)
5. [Wrap-up](#wrap-up)


---

## Proximity-based outlier detection (high-level idea)

**Proximity-based methods** identify a point as an outlier if it lies in a **sparsely populated** region of the feature space.

<img width="566" height="714" alt="image" src="https://github.com/user-attachments/assets/fa26fab7-243a-4b63-8340-48ae76c1ea8b" />


There are three main ways to define “proximity”:

1. **Cluster-Based Methods**
2. **Distance-Based Methods**
3. **Density-Based Methods**

---

## 1. Introduction

### 1a) Cluster-Based Methods

**Idea:** Outlierness is measured by how far a point is from its nearest cluster center.
<img width="800" height="665" alt="image" src="https://github.com/user-attachments/assets/615f827b-f282-4b96-904b-107226d033ff" />

- Outlierness concept: **distance from assigned cluster center**
- Typical algorithm: **K-means**

```text
outlierness = distance(point - assigned_cluster_center)
```

<img width="600" height="610" alt="image" src="https://github.com/user-attachments/assets/2ca8f47f-2d55-4d2b-bee5-a2c10c45d9ce" />


Implementation reference:
- [Proximity-based/KMeansOutlierDetector.py](https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/Proximity-based%20/KMeansOutlierDetector.py)

---

### 1b) Distance-Based Methods

**Idea:** Each point’s anomaly score is based on distances to its **k nearest neighbors**.
<img width="1200" height="333" alt="image" src="https://github.com/user-attachments/assets/07a42f12-4073-401d-9af4-c5fd0814dd9c" />


Common scoring variants: Distance to the **k-th nearest neighbor**
```text
outlierness = average(distance(point - neighbor_1),
                      distance(point - neighbor_2),
                      ...,
                      distance(point - neighbor_k))
```
- **Average** distance to all k neighbors
- **Harmonic mean** distance to all k neighbors


Intuition:
- Points that are **farther** from their neighbors are more likely to be outliers.



Implementation reference:
- [Proximity-based/KNNOutlierDetector.py](https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/Proximity-based%20/KNNOutlierDetector.py)


---

### 1c) Density-Based Methods

**Idea:** Measure a point’s **local density** by looking at neighbor distances in its neighborhood.  
A classic example is **LOF (Local Outlier Factor)**.

High-level steps (intuition):
1. Compute average neighbor distance in a local region
2. Take the inverse as a proxy for local density
3. Compare local density to neighbors’ densities
4. Use that ratio-like comparison as the **LOF score**

<img width="1200" height="547" alt="image" src="https://github.com/user-attachments/assets/cd8d3667-0cb8-4ec4-9ca2-3406091695a5" />


Interpretation:
- If a point’s local density is **much lower** than its neighbors’, it is more likely to be an outlier.

Reference:
- https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/Proximity-based%20/LOFOutlierDetector.py
- https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/Proximity-based%20/DBSCANOutlierDetector.py

---

## 2. Comparison

Each method has strengths and limitations. A helpful way to summarize their differences is:

<img width="800" height="693" alt="image" src="https://github.com/user-attachments/assets/0a087d72-7ece-4fca-ba35-1567aff2ad7f" />

<img width="800" height="693" alt="image" src="https://github.com/user-attachments/assets/be86be64-a911-4678-a8f6-efa97194881b" />


### Global vs. local analysis

- **Purely global** (Cluster-Based Methods) can miss **localized outliers**.
- **Purely local** (Density-Based Methods) can miss **small clusters of outliers**.
- **Distance-Based Methods** sit in between — a balance of global and local.

As discussed in earlier parts of this series, unsupervised learning doesn’t have an absolute “good” or “bad” the way supervised learning does (where accuracy/precision can objectively compare models). Without labels, there’s no ground truth — you evaluate whether a method works for the outliers that matter in your business scenario.

**Practical takeaway:**  
To find the best proximity-based approach, try multiple methods and pick the one that best captures the outliers you care about.

---

## 3. Optimization considerations

Proximity-based outlier detection can be computationally expensive, especially when:

- Full pairwise distances are computed
- Data is high-dimensional
- Very fine-grained local analysis is required

Why it can be expensive:
- Many methods rely on pairwise distance computations, which can require up to **O(N²)** comparisons for **N** points.
- As N grows, computation grows rapidly and can become impractical for large datasets.


### Cell-Based Pruning

<img width="676" height="682" alt="image" src="https://github.com/user-attachments/assets/c47cad33-2117-4d9f-9917-3712d4813839" />

🟥 Red Dot: Query Point This is the point for which we want to find the k-nearest neighbors (k = 3).

🟩 Green Box: Query Cell The grid partitions the space into square cells (in this case, 2x2 units). The query point falls into the green-highlighted cell.

🟧 Orange Dots: Candidate Points We only search for neighbors in the query cell and its 8 surrounding neighbors (3x3 block), drastically reducing the number of comparisons.

🔵 Blue Dots: Actual k-Nearest Neighbors After searching among candidate points, we find the 3 closest ones.

### Sampling-based Pruning

<img width="676" height="682" alt="image" src="https://github.com/user-attachments/assets/aa5f1416-1759-4fd3-81a6-ab0e6f780027" />

🔴 Red Dot: Query Point This is the target point for which we want to find the nearest neighbors.

🟠 Orange Dots: Sampled Points

Instead of scanning all 500 points, we randomly sample 10% of the data (≈ 50 points). This prunes the search space, greatly reducing computation.

🔵 Blue Dots: Approximate k-NN (from Sample)

We run k-NN on just the sampled subset to get approximate neighbors.

🟢 Green Hollow Dots: True k-NN (from Full Data)

For reference, we compute the true neighbors from the full dataset. You can see there’s a small difference, which is the trade-off for efficiency.
## Wrap-up

All three approaches revolve around the same core themes:

- **Locality**
- **Sparsity**

Their main differences are in **how proximity is defined and computed**.

In practice, many algorithms blend ideas across categories (e.g., combining density and distance signals).
