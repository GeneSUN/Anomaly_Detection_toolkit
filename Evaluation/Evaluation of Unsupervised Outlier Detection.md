# 🧪 Evaluation of Unsupervised Outlier Detection
**Understanding unsupervised learning beyond the supervised mindset**

---

## ✅ Why evaluation is difficult

Evaluating outlier detection algorithms is inherently challenging due to:

- **The lack of ground-truth labels** (i.e., known outliers are often unavailable)
- **The inherent rarity of outliers**

---

## ✅ How to solve this "The lack of ground-truth labels"

**1. Label them**

The intuitive solution is to manually label anomalies.  Even if you only label a small subset, you can compute metrics, then compare models and validate the results.  

**2. Which one to Label**

These labels are not based on real-world ground truth — **you create them**. So before labeling, a critical question must be asked:

```
What is an anomaly?
Why is this data point an anomaly? Why not the others?
```

**3. Outlier as Label**


In unsupervised anomaly detection, we define an anomaly as: > **“Something that significantly deviates from the majority.”**

- simple case: define outlier as top 1%; if you are the top 1%, i label you.
- More advanced
- <img width="560" height="590" alt="image" src="https://github.com/user-attachments/assets/82a5189a-6e8c-47e2-84d1-170214f176b7" />

**4. Outlier as Label**

There is a huge question!!!, **when you use an unsupervised method, you already know how good it will perform**

use the top 1% as an example: it is not about "if this method can find the top 1%", is about "if top 1% is the anomaly you want"!!!



So the question is not simply: **“Did the model find anomalies?”**, Because of course it did — that’s what it’s designed to do.

The real question is:

**“Are these anomalies we actually care about?”**


### **Analogy: supervised vs unsupervised**

**Supervised learning**
Supervised learning is to adjust the model based on the label.  It is kind of like cooking your own food based on what you want:

- “Winter is coming, so I want a warm soup.”
- “My stomach doesn’t feel well, so no spicy food.”
- “Seattle has no sunshine, so I should eat some fish for Vitamin D.”

You customize the dish based on your personal requirements.  Your **requirements are the labels**, and the **dish is the model output**.


**Unsupervised learning**
Unsupervised learning, on the other hand, is like going to a restaurant:

- This place might be Mediterranean (Italian/Spanish/French),
- or Japanese/Chinese/Korean,
- or a BBQ/steak/burger place,
- or just fast food.

Each restaurant has its own specialty, and they are good at what they cook.

So the evaluation is not:

❌ **“Can this restaurant make delicious Italian or Chinese food?”**

The real question is:

✅ **“Do I want to eat this kind of food today?”**



**Back to anomaly detection**

What really matters is not:

❌ **“How successful can you detect this kind of anomaly?”**

But rather:

✅ **“My model can detect this kind of anomaly — do you actually want to detect this type of anomaly?”**

---


1. KDE/Gaussian can detect value significantly different from majority

<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/4d174e12-12e1-44fc-a90d-2cc0d4d4f99b" />


2. Time series can detect significantly different from history pattern

3. Combine KDE and Time series together, we can detect significantly drop and extremly low

4. Multi-e

5. the series is either magnitudely or pattern different from majority.

6. Linear, the linear combination is significantly different from others



the first one, is launch a platform, and provide a click confirm button. so user can validate if it is abnormal or not. so we get real-world example.

the second one is check the click rate of this platform.




