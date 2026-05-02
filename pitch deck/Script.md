# SentinAL — Final Hackathon Presentation Script
## *Indian Audience Version | Natural + Slightly Witty + Judge-Friendly*

> **Use this as your final speaking script.**
> Tone: confident, practical, natural, with **light puns / smart lines** to keep it engaging.
> Ideal duration: **6–8 minutes**

---

# **SLIDE 1 — TITLE / INTRODUCTION**

**What to say:**

Good morning everyone.
I am Yash Nagrale, Suraj, Shubham and Akshay
And we are **Team 2Infinity**, and our project is **SentinAL**.

SentinAL is a **privacy-first, edge-deployed clinical AI platform** designed to detect health deterioration **before it becomes critical**.

Our selected problem statement was **PS2**, which focuses on predicting **patient physiological deterioration using time-series vital sign data**.

But while working on this, we realized one important thing:

In healthcare, problems rarely come **one at a time like assignment questions**.

A high-risk patient may also develop:
- diabetic wound complications
- or even neurological emergencies like stroke

So instead of building just one isolated model, we built **SentinAL** as a **connected multi-modal healthcare monitoring system**.

You can think of it as:

> **not just one alarm, but a smarter clinical warning system.**

And yes…  
that’s why we named it **SentinAL**.

Because it doesn’t just analyze —  
it **stands guard**.

---

# **SLIDE 2 — THE PROBLEM**

**What to say:**

The core problem we wanted to solve is that **clinical deterioration is often detected too late**.

In many cases, warning signs appear **hours before a crisis**, but they are missed due to:

- delayed monitoring
- limited staff
- lack of specialist access
- and difficulty in interpreting early warning patterns

This becomes even more important in settings like:

- **rural healthcare**
- **home-care monitoring**
- and **resource-constrained environments**

We also looked at connected clinical risks:

- diabetic foot complications often go unnoticed until they become severe
- and stroke diagnosis depends heavily on **speed**

And in healthcare, as we all know:

> **“We’ll check later” is not always a medically safe strategy.**

So the real problem is not lack of medical knowledge.

The real problem is lack of:

## **early, accessible, and practical decision-support**

And that is exactly where we positioned SentinAL.

---

# **SLIDE 3 — OUR SOLUTION**

**What to say:**

Our solution is **SentinAL**, which combines **three AI modules into one unified platform**.

### **Module 1: PS2 — Vital Signs Monitor**
This predicts physiological deterioration using **time-series patient data**

### **Module 2: PS1 — Diabetic Foot Wound Grader**
This classifies wound severity using **foot wound images**

### **Module 3: PS5 — CT Stroke Detector**
This detects possible stroke using **brain CT scan images**

In addition to prediction, the platform also supports:

- specialist recommendation
- urgency mapping
- nearby hospital finder
- and local clinical explanation through LLM support

So in simple terms:

> **SentinAL helps identify risk early, support triage, and guide timely intervention.**

Basically:

> **less guesswork, more guidance.**

---

# **SLIDE 4 — SYSTEM ARCHITECTURE**

**What to say:**

The architecture of SentinAL is designed to be:

- **privacy-first**
- **edge-friendly**
- and **practical for deployment**

The overall workflow is:

## **Input → Preprocessing → AI Inference → Risk Output → Referral Support**

Depending on the module, the input can be:

- a **CSV file of vitals**
- a **foot wound image**
- or a **brain CT scan**

Each input goes through preprocessing, then the corresponding model performs inference, and the final output includes:

- risk level
- confidence
- urgency
- and referral guidance

One key strength of SentinAL is that the AI runs **locally**, which means sensitive patient inference does not need to depend on cloud prediction.

Because in healthcare:

> **latency is annoying, but privacy leaks are worse.**

So we wanted the system to be both **fast and respectful of patient data**.

---

# **SLIDE 5 — MODULE DEEP DIVE: PS2**

**What to say:**

Our **main and primary module** is **PS2**, which focuses on **physiological deterioration prediction**.

The input is a **vital signs time-series dataset**, where each patient has measurements recorded across time.

The model does not just look at one reading in isolation.

Instead, it tries to understand:

- how the patient is changing over time
- whether multiple vitals are worsening together
- and whether those changes indicate early deterioration

For this, we used a:

## **Temporal Transformer**

The model takes:

- **12-hour sliding windows**
- with **34 engineered clinical features**

and predicts whether the patient is at risk of deterioration.

This module was trained on:

- **293K rows**
- grouped into **216K temporal windows**
- across **7,000 patients**

So in short:

> **we are not just checking if a patient looks bad now —  
we are checking if they are trending in the wrong direction.**

That is far more clinically meaningful.

---

# **SLIDE 6 — WHY TEMPORAL TRANSFORMER**

**What to say:**

We specifically chose a **Temporal Transformer** for PS2 because this is not just a tabular classification problem.

It is a **sequence learning problem**.

In healthcare, the more important question is often not:

> “What is the heart rate right now?”

but rather:

> “What has been happening over the last several hours?”

Transformers are useful here because they can learn:

- which time steps matter most
- which variables interact across time
- and what patterns are associated with deterioration

Compared to a simple row-wise model, this gives much stronger **temporal understanding**.

Our architecture uses:

- **3 encoder layers**
- **8 attention heads**
- and a classifier head for final prediction

with a total model size of around:

## **434K parameters**

So no, we did not bring a **rocket launcher to solve a fever chart**.

The model is actually **lightweight enough to be practical**, while still being powerful enough to learn temporal patterns.

---

# **SLIDE 7 — PS1 & PS5 MODULES**

**What to say:**

We then extended SentinAL with two connected modules inspired by **PS1** and **PS5**.

### **PS1 — Diabetic Foot Wound Grader**
This module takes a **foot wound image** and classifies it into one of the **Wagner grades**, which helps estimate wound severity.

It uses:

## **EfficientNet-B0**

along with **CLIP-based image validation**, so the system can first check whether the uploaded image is actually relevant.

So ideally, it avoids situations where someone uploads a shoe selfie and expects a medical prediction.

---

### **PS5 — Stroke Detection**
This module takes a **brain CT scan image** and classifies it as:

- **Normal**
- or **Stroke**

This also uses:

## **EfficientNet-B0**

and supports rapid image-based screening.

These two modules were added because they represent **connected complications and escalation pathways** in high-risk patients.

So PS2, PS1, and PS5 are not random add-ons.

They are different ways in which **the same vulnerable patient may clinically deteriorate**.

---

# **SLIDE 8 — DATA & FEATURES**

**What to say:**

One of the most important parts of our project was not only model selection —

but also **feature engineering and data processing**.

For PS2, we engineered **34 clinically meaningful features** from the raw data.

These include:

- **Pulse Pressure**
- **Mean Arterial Pressure**
- **Shock Index**
- **qSOFA-related indicators**
- inflammatory markers
- organ function indicators
- and temporal trend features

This was also inspired by a mentor suggestion:

## **“features from a feature”**

That insight was actually very valuable.

Because instead of feeding the model only raw numbers, we tried to extract:

> **more medically meaningful signals from those numbers**

So instead of just giving AI data,

we tried to give it a bit more **clinical common sense**.

The pipeline was:

## **Ingest → Assign → Engineer → Encode → Scale → Window → Balance → Train → Evaluate → Deploy**

---

# **SLIDE 9 — KEY DIFFERENTIATORS**

**What to say:**

What makes SentinAL different is that it is not just **three separate models stitched together in a hurry before deadline**.

It is designed as a **usable healthcare decision-support system**.

Its key differentiators are:

- **Privacy-first local inference**
- **Specialist recommendation**
- **Nearby hospital finder**
- **Clinical explanation support**

This means after prediction, the system can also help answer:

- What is the urgency?
- Which specialist is needed?
- Where should the patient go next?

That makes the platform much more practical in a real-world triage setting.

Because prediction alone is useful.

But:

> **prediction + next-step guidance is far more actionable.**

---

# **SLIDE 10 — TECH STACK**

**What to say:**

We built SentinAL using a practical and deployment-friendly open-source stack.

Our main technologies include:

- **PyTorch** for deep learning
- **TorchVision** for image backbones
- **HuggingFace Transformers** for CLIP validation
- **scikit-learn** for preprocessing and evaluation
- **Pandas and NumPy** for data handling
- **Streamlit** for the interface
- and **Ollama** for local LLM support

We wanted the stack to be:

- fast to build
- reproducible
- and suitable for prototype deployment

Basically:

> **strong enough for the hackathon, but sensible enough for the real world.**

---

# **SLIDE 11 — BENCHMARKS / RESULTS**

**What to say:**

Coming to the results:

### **PS2 — Vital Signs Prediction**
- **AUROC: 0.996**

### **PS1 — Wound Grading**
- **Accuracy: 97.05%**
- **F1 Score: 0.970**

### **PS5 — Stroke Detection**
- **AUROC: 0.982**
- **Accuracy: 92.2%**

These results were achieved using:

- class balancing strategies
- feature engineering
- careful preprocessing
- and model-specific optimization

Now of course, in healthcare:

> **good metrics are important — but trustworthy behavior matters even more.**

So we do not present this as:

> “AI that replaces doctors.”

We present this as:

> **AI that supports earlier, smarter clinical decision-making.**

And that is the more responsible position.

---

# **SLIDE 12 — USER EXPERIENCE**

**What to say:**

We also focused on making the platform easy to use.

The user workflow is:

1. Select module  
2. Upload data  
3. Run AI analysis  
4. View alert / risk output  
5. Get specialist recommendation  
6. Find nearby hospitals

The interface is designed to be simple enough for:

- clinicians
- students
- researchers
- and potentially field-level healthcare workers

Because even the smartest model becomes useless if the UI feels like:

> **“Please sacrifice three weekends to understand this dashboard.”**

So we kept it simple, usable, and practical.

---

# **SLIDE 13 — IMPACT & ROADMAP**

**What to say:**

The long-term vision of SentinAL is to support:

- **earlier detection**
- **better triage**
- **privacy-preserving AI**
- and **healthcare accessibility**

In the future, this can be extended through:

- ECG arrhythmia detection
- X-ray-based screening
- multilingual clinical summaries
- mobile deployment
- and federated learning across hospitals

So our goal is not only to build one hackathon model —

but to build the foundation of a:

## **scalable AI healthcare support platform**

In short:

> **today it is a prototype — tomorrow it can become a much broader clinical assistant ecosystem.**

---

# **SLIDE 14 — CONCLUSION**

**What to say:**

To conclude:

SentinAL is a **unified clinical AI system** that combines:

- physiological deterioration prediction
- diabetic wound grading
- and stroke screening

into one connected platform.

Our main focus was to move healthcare from:

## **reacting to emergencies**

toward:

## **predicting and responding earlier**

Because in healthcare, even a few hours of early warning can make a major difference.

And that is the core idea behind SentinAL.

> **It doesn’t wait for the emergency — it watches for the warning.**

Thank you.

---

# **BEST OPENING LINE**
Use this exactly if you want:

> **Good morning everyone. We are Team 2Infinity, and our project is SentinAL — a privacy-first, edge-deployed clinical AI platform designed to detect health deterioration before it becomes critical.**

---

# **BEST CLOSING LINE**
Use this exactly:

> **SentinAL doesn’t wait for the emergency — it watches for the warning.**

---

# **FINAL DELIVERY TIPS FOR TOMORROW**

## **How to speak**
- Speak **slightly slower**
- Keep your tone **clear and confident**
- Don’t rush technical slides
- Keep eye contact with judges
- Don’t read like a robot
- Explain like **you built it and understand it**

## **How to use humor**
Use the puns lightly.
Pause after them.
Let them land naturally.

Don’t make it a comedy set.
Just make it **human and memorable**.

## **What to avoid**
Avoid saying things like:

- “revolutionary”
- “near-perfect”
- “doctor replacement”
- “fully hospital-ready”
- “clinically approved”

Instead say:

- **strong prototype**
- **decision-support system**
- **clinically meaningful**
- **privacy-conscious**
- **designed for practical deployment**

That is much smarter under judge questioning.

---

# **ULTRA-SHORT BACKUP VERSION (IF THEY CUT YOUR TIME)**

Good morning everyone. We are Team 2Infinity, and our project is SentinAL.

SentinAL is an AI-powered healthcare early warning system designed to detect patient deterioration before it becomes critical.

We selected PS2 as our main problem statement and extended it with PS1 and PS5 to create a connected multi-modal healthcare monitoring platform.

Our core PS2 module uses time-series vital sign data, engineered clinical features, and a Temporal Transformer to predict deterioration risk.

We also added:
- a wound severity classifier using EfficientNet-B0
- and a CT-based stroke detector using EfficientNet-B0

The goal of SentinAL is simple:

> **to identify risk early enough for meaningful intervention.**

Thank you.
