# 🧪 Evaluation of Unsupervised Outlier Detection
**Understanding unsupervised learning beyond the supervised mindset**

---

## ✅ Why evaluation is difficult

Evaluating outlier detection algorithms is inherently challenging due to:

- **The lack of ground-truth labels** (i.e., known outliers are often unavailable)
- The inherent rarity of outliers

---

### "The lack of ground-truth labels"
**In supervised anomaly detection**
1. the anomalies come from the real world, so their labels already exist (1 = anomaly, 0 = normal).
2. we use the labels to train and tune a model. understand the underlying mechanism/association of why those anomalies happened:
3. evaluation is to measure how accurately the model captures the anomaly mechanism
   
<img width="539" height="116" alt="Untitled" src="https://github.com/user-attachments/assets/c45cf489-9432-4d42-b2bc-ed37205eebe3" />


**For unsupervised learning,**
1. we don't have label, so we have no clue of anomaly mechanism.
2. we use business judgement to define possible anomaly mechanism
3. based on the defined possible anomaly mechanism, we find the matching model

<img width="1265" height="375" alt="Untitled" src="https://github.com/user-attachments/assets/af73b88f-cfd0-49f1-808d-5f6b2dd55e28" />

**There is no evaluatlion in Unsupervised Learning!!!**

---

## Mechanism/Pattern -> Model

We need to detect an anomaly which is signficantly dropped from previous history, but the history follow a cyclic seasonablity. To detect that, we end up using an ensemble model.

<img width="680" height="707" alt="image" src="https://github.com/user-attachments/assets/63a88bfa-2ccb-4017-aa7e-7b9ede6402aa" />

**1. KDE/Gaussian can detect value significantly different from majority**

<img width="489" height="390" alt="image" src="https://github.com/user-attachments/assets/4d174e12-12e1-44fc-a90d-2cc0d4d4f99b" />

**2. Time series can detect significantly different from history pattern**

<img width="806" height="470" alt="image" src="https://github.com/user-attachments/assets/93d88140-431c-442c-89d1-198bd43a3079" />

**3. Combine KDE and Time series together, we can detect significantly drop and extremly low**

<img width="282" height="149" alt="image" src="https://github.com/user-attachments/assets/2151865a-eb5c-49ed-a77f-91ff2698ef6b" />

<img width="1100" height="653" alt="image" src="https://github.com/user-attachments/assets/8f463b3a-0727-4290-896e-16766aa9db0b" />


### Other Patterns

- Multi-variant Outlier
- the series is either magnitudely or pattern different from majority.
- Linear, the linear combination is significantly different from others

## Beyond Metrics: Practical Ways to Evaluate Unsupervised Anomaly Detection

**1. User validation:**
   
Launch a platform that shows detected anomalies and provides a “confirm” button.
When users confirm whether a case is truly abnormal, we obtain real-world labeled examples.

**2. Engagement metrics:**

Measure how frequently users interact with the platform (click rate).
A higher confirmation or review rate indicates that the detected anomalies are meaningful and useful.

---

## How to build a unsupervised learning outlier/novelty project

After we launched the platform, my manager shared it with other teams. Soon many people started reaching out, asking if we could help identify abnormal behavior in their customer groups. For example, one team had customers complaining about unstable 4G/5G, another saw unusual device reboots. They handed me lists of customers and asked:

**“Can your model detect what’s abnormal about these people compared to everyone else?”**

At first, this sounds like a reasonable request.
But when I ran the analysis, I found extremely difficult to detect unusual. we got 50+ features, and when you refer anomaly, you mean comparing to the majority, or specific model/priceplan?

**so I explain to them using an analogy**

Imagine I’m an engineer who has designed a medical device that detects heartbeat anomalies and blood pressure anomalies, etc. The device works perfectly for those signals.

Now doctors reaching out to me, start sending groups of patients and ask:

- Can you detect who has cancer?
- Can you detect diabetes?
- Can you detect infertility?

But I’m not a doctor, and my device is not a universal illness detector.

You as doctor, need to detect the associated pattern, and i will help you design the device which can detect such pattern.

<img width="1293" height="375" alt="Untitled" src="https://github.com/user-attachments/assets/3e6ac719-402f-42eb-9275-a4e497aaeab1" />


**From unsupervised to semi-supervised to supervised**



