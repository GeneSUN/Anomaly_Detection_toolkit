# Autoencoders for Time Series Anomaly Detection: A Visual and Practical Guide

This article covers:

- **Conceptual overview:** how autoencoders work (intuition + methodology)  
- **Model construction:** building an LSTM-based autoencoder (multiple approaches)  
- **Practical code & notebooks:** scripts + Colab workflow from training to visualization  
- **Result interpretation:** how to visualize and explain anomalies statistically and graphically  

## Table of Contents

- [Autoencoders for Time Series Anomaly Detection: A Visual and Practical Guide](#autoencoders-for-time-series-anomaly-detection-a-visual-and-practical-guide)
  - [1. Intuition behind autoencoders](#1-intuition-behind-autoencoders)
  - [2. Autoencoder methodology](#2-autoencoder-methodology)
  - [3. Building an LSTM autoencoder](#3-building-an-lstm-autoencoder)
    - [3.1 Build from scratch in PyTorch (manual)](#31-build-from-scratch-in-pytorch-manual)
    - [3.2 Use time series libraries (PyTorch Forecasting + Lightning)](#32-use-time-series-libraries-pytorch-forecasting--lightning)
    - [3.3 Use PyOD (recommended)](#33-use-pyod-recommended)
    - [3.4 Full pipeline class (preprocess + train + visualize)](#34-full-pipeline-class-preprocess--train--visualize)
  - [4. Model results: anomaly score](#4-model-results-anomaly-score)
    - [4.1 From time series to anomaly score](#41-from-time-series-to-anomaly-score)
    - [4.2 From anomaly score to outlier label](#42-from-anomaly-score-to-outlier-label)
  - [5. Visualization](#5-visualization)
    - [5.1 The challenge (unsupervised interpretability)](#51-the-challenge-unsupervised-interpretability)
    - [5.2 A practical approach](#52-a-practical-approach)
      - [5.2.1 Plot anomaly score distribution](#521-plot-anomaly-score-distribution)
      - [5.2.2 Plot normal vs abnormal time series](#522-plot-normal-vs-abnormal-time-series)
      - [5.2.3 Mean and spread](#523-mean-and-spread)
  - [6. Higher-dimension time series](#6-higher-dimension-time-series)
  - [🧠 Conclusion](#-conclusion)


---

## 1. Intuition behind autoencoders

Imagine you have a collection of English sentences. Most are well-written, but some contain obvious grammar mistakes. You want a method to catch the flawed sentences.

<img width="1542" height="776" alt="image" src="https://github.com/user-attachments/assets/707ab089-b75e-4b43-a6cf-1031cabba3a6" />


Think of an autoencoder like a two-step translation:

1. Translate English → Spanish (**encoding / compression**)  
   - The compressed representation is the **latent space**.
2. Translate Spanish → English (**decoding / reconstruction**)  

What happens?

- If a sentence is well-written, the back-and-forth translation keeps it mostly unchanged.
- If a sentence has grammar mistakes, the translation can **exaggerate** the errors, producing a worse reconstruction.



---

## 2. Autoencoder methodology

Autoencoder-based anomaly detection works like this:

<img width="1508" height="253" alt="image" src="https://github.com/user-attachments/assets/49b5d44b-97df-4c71-9d52-87e9550afcbe" />


**Step 1:** Take a series (or window/vector) as input.  
**Step 2:** Reconstruct the original series via encode → decode.  
**Step 3:** Compare input vs. reconstruction and compute a **reconstruction error** (e.g., Mean Squared Error).

In practice:

- Each window/series is mapped to **one anomaly score**.
- Detecting anomalous time series becomes detecting anomalous **scalar values**.
- Windows with the **largest reconstruction errors** are considered anomalies.

<img width="592" height="393" alt="image" src="https://github.com/user-attachments/assets/ee6b2a44-366a-4a6e-9f5d-049f251e7237" />


---

## 3. Building an LSTM autoencoder

In the repository, three approaches are presented for building an autoencoder for time series data:

### 3.1 Build from scratch in PyTorch (manual)

- Notebook: [AutoEncoder/lstm_autoencoder_Manual.ipynb](https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/AutoEncoder/lstm_autoencoder_Manual.ipynb)
- https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/AutoEncoder/LSTM-Autoencoder-Tutorial-for-Sequential-Data.pdf

### 3.2 Use time series libraries (PyTorch Forecasting + Lightning)

- Notebook: [AutoEncoder/lstm_autoencoder_pytorch_forecasting.ipynb](https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/AutoEncoder/lstm_autoencoder_pytorch_forecasting.ipynb?source=post_page-----021d4b9c7909---------------------------------------)

### 3.3 Use PyOD (recommended)

PyOD is a well-encapsulated library for outlier detection with a modular design and convenient APIs.

Example:

```python
from pyod.models.auto_encoder_torch import AutoEncoder

model = AutoEncoder(
    hidden_neurons=[144, 4, 4, 144],
    hidden_activation="relu",
    epochs=20,
    batch_norm=True,
    learning_rate=0.001,
    batch_size=32,
    dropout_rate=0.2,
)
```

- Notebook: [AutoEncoder/PyOD_AE.ipynb](https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/AutoEncoder/PyOD_AE.ipynb?source=post_page-----021d4b9c7909---------------------------------------)

### 3.4 Full pipeline class (preprocess + train + visualize)

A class designed to include data preprocessing, training, and (most importantly) visualization:

- https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/AutoEncoder/AutoencoderAnomalyDetector.py
- https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/AutoEncoder/MultiTimeSeriesAutoencoder.py
- A [Colab notebook](https://colab.research.google.com/drive/174QBd3_2k3e88UyC__jLLG45Ukk-PMBx) is also available for an end-to-end walkthrough.

---

## 4. Model results: anomaly score

### 4.1 From time series to anomaly score

After training, the model reconstructs each time series window and produces a reconstruction error as the anomaly score.

A typical workflow:

- Start with a long time series
- Segment it into multiple fixed-length windows (e.g., **24-hour** windows)
- Reconstruct each window
- Convert reconstruction error to an **anomaly score** per window

<img width="2000" height="744" alt="image" src="https://github.com/user-attachments/assets/bace55e3-9650-4761-a2e7-a498ce96ece6" />


How to interpret the score:

- **Low score** → model is familiar with this pattern (likely normal)  
- **High score** → model struggles to reconstruct it (potential anomaly / unusual behavior)

### 4.2 From anomaly score to outlier label

Next, convert **numeric anomaly scores** into **binary labels**:

> Is this window an outlier or not?

Common rule-based options:

- **Percentile threshold:** mark as anomaly if score > percentile (e.g., 95th)  
- **Standard deviation rule:** anomaly if score > mean + *k*·σ (e.g., mean + 2σ)

<img width="615" height="452" alt="image" src="https://github.com/user-attachments/assets/14ccad61-8777-4e8e-a86b-d3727112615b" />


**Practical use:**  
Once you have scores across windows, you can flag abnormal periods and investigate or trigger alerts.

---

## 5. Visualization

### 5.1 The challenge (unsupervised interpretability)

- Autoencoders are trained **without anomaly labels**, so during training we don’t know which windows are “normal” or “abnormal.”  
- After training, the model doesn’t output a true label either—it outputs an **anomaly score** (reconstruction error), which only tells us how well it can reconstruct a window.

So the real work starts after scoring: 
1) to turn scores into “normal vs. anomaly”, and  
2) **explain** why a making such decision
   
This matters for non-technical stakeholders. 
- Methods like KNN/LOF/KDE are easier to explain (“far away” or “low density”),
  <img width="700" height="534" alt="image" src="https://github.com/user-attachments/assets/b9e31759-a7d1-4e22-9d7e-892c8e965e29" />

- while  deep learning models, like autoencoders are harder,  because their reasoning happens in a hidden **latent space**, so visualization becomes key.


### 5.2 A practical approach

A practical way to interpret results:

- Skip complex math
- **Show** differences between normal and abnormal windows
  - For example, show what a typical low anomaly score time series looks like versus one with a high score.  
- Compare low-score time series vs high-score time series

This “visual contrast” helps stakeholders understand what the model considers normal vs anomalous.

Use the [Colab notebook](https://colab.research.google.com/drive/174QBd3_2k3e88UyC__jLLG45Ukk-PMBx?source=post_page-----021d4b9c7909---------------------------------------#scrollTo=YOsuWLmcR4Dz) to demonstrate the following:

#### 5.2.1 Plot anomaly score distribution

Plot the anomaly score distribution to quickly understand:

- what range looks normal,
- what range looks abnormal,
- how separated the two regimes are.

<img width="1400" height="686" alt="image" src="https://github.com/user-attachments/assets/a49ae7af-ec0f-426e-92c8-70e0e35a26b9" />


#### 5.2.2 Plot normal vs abnormal time series


Compare representative examples:

- A **low-score** window (high confidence normal)
  - Constant time series (e.g., values roughly within 10–30) tend to be normal
<img width="1000" height="814" alt="image" src="https://github.com/user-attachments/assets/f17a0ea3-cb73-42d1-8270-77c8d9ea5fc3" />

- A **high-score** window (high confidence abnormal)
  - Highly fluctuating series tend to be abnormal

<img width="1000" height="818" alt="image" src="https://github.com/user-attachments/assets/aab5673d-49f5-488d-b891-708208a6a21c" />


You can also select samples by score range, for example:

- **High confidence abnormal:** score in [10, 20] (based on the distribution)
<img width="2000" height="815" alt="image" src="https://github.com/user-attachments/assets/d933bffe-443a-4106-ade1-547e93b8ac11" />


- **High confidence normal:** score in [4, 6] (based on the distribution)

<img width="2000" height="812" alt="image" src="https://github.com/user-attachments/assets/802f25fc-828d-468b-83a8-e2f591bfc559" />


In the described examples:

- High-confidence normal ranges roughly **12.5 to 27.5**
- Abnormal windows may swing dramatically (e.g., **20 to 50**)

#### 5.2.3 Mean and spread

A helpful summary visualization is mean + spread (variability):

- Normal data might range ~14 to 24
- Abnormal data might range ~15 to 30, with spikes up to ~40

<img width="2000" height="968" alt="image" src="https://github.com/user-attachments/assets/43eee5ae-03a1-4c1d-bacb-85d541cdd265" />


---

## 6. [Higher-dimension time series](https://colab.research.google.com/drive/1jXEZjyxvN-5LGE6Oe-FTHwPYru6N343Y)

So far, we’ve focused on one-dimensional time series with multiple samples:

- Shape: `torch.Size([n_samples, 1, Time])`

For high-dimensional time series (multiple features over time), the input becomes:

- Shape: `torch.Size([n_samples, features, Time])`

You can think of each `[features, Time]` sample like a **grayscale 2D image**:

- x-axis = time
- y-axis = features

The autoencoder learns patterns across **both time and features** — and detects anomalies in this “image-like” structure.

<img width="755" height="711" alt="image" src="https://github.com/user-attachments/assets/371487f6-7bac-4f47-b4e4-391face0df68" />
- https://colab.research.google.com/drive/1jXEZjyxvN-5LGE6Oe-FTHwPYru6N343Y

### Visualization becomes harder

In 1D, each sample is a line plot.  
In multi-feature time series, each sample becomes a **heatmap**, and:

<img width="400" height="751" alt="image" src="https://github.com/user-attachments/assets/bd45da92-7af0-484e-9055-eff625dfd2d9" />


- comparing heatmaps is less intuitive,
- spotting “what is abnormal” is harder to explain clearly.

**Bottom line:** feasible to detect anomalies in high-dimensional time series, but significantly harder to interpret and communicate.

---

## 🧠 Conclusion

- Autoencoders are effective for unsupervised anomaly detection in time series.
- They learn “normal” patterns and detect anomalies without labeled data.
- Their black-box nature makes interpretation harder — especially for high-dimensional inputs.
- Visualization is key: comparing original vs reconstructed signals and contrasting normal vs abnormal windows helps explain results.
- With clear interpretation, autoencoders become a powerful and flexible anomaly detection tool.
