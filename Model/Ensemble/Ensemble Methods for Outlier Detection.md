# Ensemble Methods for Outlier Detection  
## From supervised learning to unsupervised learning

---

## Table of Contents

- [Introduction](#introduction)  
- [1. Why Ensemble: Bias-Variance](#1-why-ensemble-bias-variance)  
  - [Variance](#variance)  
  - [Bias](#bias)  
- [2. Categorization of Ensemble Methods](#2-categorization-of-ensemble-methods)  
  - [Independent vs. Sequential](#independent-vs-sequential)  
  - [Model-Centric vs. Data-Centric](#model-centric-vs-data-centric)  
- [3. Model Combination for Outlier Ensembles](#3-model-combination-for-outlier-ensembles)  
  - [Score normalization](#score-normalization)  
  - [Combination strategies](#combination-strategies)  
- [Limitations and trade-offs](#limitations-and-trade-offs)  
- [Conclusion](#conclusion)  
  - [Pros](#pros)  
  - [Cons](#cons)  
  - [Mitigation strategies](#mitigation-strategies)  

---

## Introduction

Ensemble methods merge the outputs of multiple algorithms to produce a stronger and more reliable result. 
- In supervised learning, ensembles are widely used (Bagging, Random Forests, Boosting).
- In unsupervised outlier detection, ensembles are less frequently discussed — but often just as valuable.

<img width="1100" height="304" alt="image" src="https://github.com/user-attachments/assets/a19edfac-383e-47e8-8f32-82abfc6a322a" />


> No single algorithm can catch every type of anomaly—each detector has blind spots. By combining multiple detectors, we can build a system that’s stronger than any one model alone.


---

## 1. Why Ensemble: Bias-Variance

The primary goal of ensemble learning is to balance **bias** and **variance**.

- Choosing an inappropriate model increases **bias**, leading to systematic errors and underfitting.
- Overfitting to training data increases **variance**, reducing generalization to new data.

The examples below build intuition for where bias/variance come from, and why ensembles help — especially in unsupervised outlier detection.

### Variance

**Variance = error due to different outcomes from different training sets.**

<img width="720" height="300" alt="image" src="https://github.com/user-attachments/assets/6ffbc1cf-caa2-45aa-8685-a9fcdebabb67" />


Imagine sampling two different sets of points from the same underlying distribution and using both to compute the outlier score for the *same* test point.

- In one sample, the test point might have fewer nearby neighbors → higher outlier score.
- In another sample, it might have more neighbors → lower outlier score.

That inconsistency is **variance**: the sensitivity of the model to the particular sample it sees.

**High variance ⇒ unstable model, fluctuating results.**  
Outlier ensembles can reduce variance by averaging over multiple models trained on different views/subsets of the data.

### Bias

**Bias happens when model assumptions don’t match reality.**
<img width="720" height="357" alt="image" src="https://github.com/user-attachments/assets/9c732914-0f49-4e81-ba3f-6fede66c6720" />


Example intuition:

- The true data relationship is curved (nonlinear).
- A linear model is applied.
- Because a straight line cannot “bend,” it will systematically miss in certain regions.

That systematic mismatch creates **bias**.

---

## 2. Categorization of Ensemble Methods

Ensemble methods can be categorized along two axes:

1. **Independent vs. Sequential**
2. **Model-Centric vs. Data-Centric**

<img width="720" height="335" alt="image" src="https://github.com/user-attachments/assets/9dc5453d-31b6-4ec0-9220-956d2a0fef8b" />

<img width="1100" height="594" alt="image" src="https://github.com/user-attachments/assets/ea6d119a-f7d1-429a-b2fb-6ddc4b9b114c" />


### Independent vs. Sequential

- **Independent Ensemble:** models run independently / in parallel, then outputs are combined.
- **Sequential Ensemble:** models run successively (one after another), where later steps depend on earlier outputs.

### Model-Centric vs. Data-Centric

- **Model-Centric:** ensemble by varying the model side  
  - different algorithms  
  - different hyperparameter settings  
  - randomized runs of the same algorithm  

- **Data-Centric:** ensemble by varying the data side (often with the same model)  
  - sampling subsets of the data  
  - random projections  
  - reweighting points or dimensions  
  - adding noise  

These strategies offer different ways to trade off bias/variance and improve robustness.

---

## 3. Model Combination for Outlier Ensembles

Once you obtain scores from multiple detectors, you need a strategy to combine them.

### Score normalization

**Score normalization is the first step** in almost any outlier ensemble, because different algorithms output scores on different scales (and sometimes with different score directions).

<img width="720" height="463" alt="image" src="https://github.com/user-attachments/assets/4f7c0124-4837-4426-a715-f68ecaf0d3b2" />


A common approach:
- normalize each detector’s scores into a comparable range such as **[0, 1]**,
- then aggregate the normalized scores.

### Combination strategies

Once scores are comparable, common aggregation methods include:

- **Score averaging**  
  Normalize each model’s score to `[0, 1]`, then average.  
- **Max score**  
  Take the maximum score across models (conservative: flags anomalies if *any* model is confident).  
- **Weighted voting**  
  Assign weights by reliability/validation, then compute weighted combination.  
- **Majority vote**  
  Each model votes anomaly vs. normal; final label is majority.
  
---

## Limitations and trade-offs

Ensembles improve stability and robustness by combining multiple views of the data/model space, often reducing bias and/or variance.

The major limitation is **computational cost**:
- Running 2+ models increases training and inference time
- Resource usage rises (CPU/GPU/memory)

However, this can often be mitigated by:
- parallelization
- distributed computing frameworks (Spark, Dask, Ray)
- GPU acceleration

The core trade-off is:

- **Accuracy/robustness** (from ensemble diversity)  
  vs.  
- **Efficiency/scalability** (managed via infrastructure and design)

---

## Conclusion

### Pros

- **Higher accuracy & robustness:** diverse models capture different aspects of the data (reducing bias/variance).
- **Better generalization:** ensembles tend to be more consistent across datasets.
- **Resilience to noise:** combining learners smooths random fluctuations.

### Cons

- **Increased computational cost:** multiple models require more resources.
- **Deployment complexity:** more models to maintain, monitor, and update.
- **Potential latency issues:** real-time systems may struggle with added inference overhead.

### Mitigation strategies

- **Parallelize** training/inference across models.
- Use **distributed frameworks** (Spark, Dask, Ray) or **GPU acceleration**.
- Apply **model distillation** to compress an ensemble into a single faster model after validation.
