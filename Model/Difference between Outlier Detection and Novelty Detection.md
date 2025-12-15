# Difference between Outlier Detection and Novelty Detection: clean/dirty Assumption

## Definition
According [Sklearn](https://scikit-learn.org/stable/modules/outlier_detection.html), outlier and novelty detection can be summarized as:

> Outlier detection: The training and testing are the same, it contains outliers which are defined as observations that are far from the others.

<img width="720" height="311" alt="image" src="https://github.com/user-attachments/assets/1a7fd5cc-a4b6-4f34-8096-5e05063463bc" />

> Novelty detection: Trainning and Testing are different. The training data is not polluted by outliers, and used as reference to detect whether a new observation is an outlier.

<img width="720" height="311" alt="image" src="https://github.com/user-attachments/assets/cdddd815-aaf5-41a6-bfe5-53c571bcb5a8" />

### Streaming vs. Batch

You can think **Novelty Detection** as **Streaming Detection**:

Every historical record is normal, we used them to detect whether new, incoming data points deviate from the established normal pattern.

<img width="720" height="349" alt="image" src="https://github.com/user-attachments/assets/97c279cf-bd0b-4fa6-a61a-0ab92a012cc3" />

On the other hand, outlier detection is more like a batch detection task. The dataset contain a mixture of normal (inlier) and abnormal (outlier) samples, but without explicit labels. The goal is to identify which points deviate significantly from the majority.

<img width="720" height="302" alt="image" src="https://github.com/user-attachments/assets/04b9f34e-ab73-4c0d-98ce-b8916bc9e4d2" />

### Assumption:

The biggest difference rely on the clean/dirty Assumption:

- Novelty detection assumes a clean training set, often using one-class models (e.g., One-Class SVM, autoencoders trained only on normal data).
- Outlier detection does not assume clean data, and is often more robust to noise, but more prone to uncertainty in defining what is “normal”.

## Example:

- https://colab.research.google.com/drive/1A68Q-GfxEBiA2dCdPWSXJK8pL_ronhyZ
- https://colab.research.google.com/drive/1qC-Gry8py_Icl0V8zNedlIeX3HFEKHuY?

The two notebooks provide examples demonstrating the difference between Outlier Detection and Novelty Detection, using Kernel Density Estimation (KDE) and ARIMA.

Outlier Detection: Each point is evaluated against the entire dataset. Outliers are those that deviate significantly in a global context.

<img width="720" height="281" alt="image" src="https://github.com/user-attachments/assets/a21cd073-6379-46b5-94bd-83cf8869021f" />

Novelty Detection: Each new point is evaluated only against the previous history.

<img width="720" height="538" alt="image" src="https://github.com/user-attachments/assets/c894720a-5ef1-4968-8a7f-d9c31b59a6c9" />


If we collect the true novelty labels across consecutive series, the results differ from the outlier case:
- Some points that are not extreme in the global distribution are still flagged as abnormal
- Because they represent unusual behavior compared to their immediate past.


<img width="720" height="358" alt="image" src="https://github.com/user-attachments/assets/16fa9609-84bd-4a07-b169-b1bb410d7d5b" />


## Conclusion:
This repository supports both Outlier Detection and Novelty Detection.
You can control the behavior with the parameter new_idx:

- new_idx = "all" → Perform Outlier Detection (entire dataset used).
- new_idx = slice(-1, None) → Perform Novelty Detection (detect only the last point).
- new_idx = slice(-n, None) → Detect the last n points as novelties.

