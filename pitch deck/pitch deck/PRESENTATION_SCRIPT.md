# SentinAl — Presentation Script

> Estimated total time: **8-10 minutes** (adjust per hackathon rules)
> Tip: Practice this 3-4 times. Speak slowly on the metrics — they're your strongest proof points.

### Problem Statements Addressed:
- **PS1:** AI-Based Prediction of High-Risk Plantar Pressure Zones for Prevention of Diabetic Foot Ulcers
- **PS2:** AI-Based Early Warning System for Patient Physiological Deterioration Using Vital Sign Time-Series Data
- **PS5:** AI-Based Brain Stroke Detection and Lesion Segmentation from CT Scans

---

## SLIDE 1: TITLE (30 seconds)

**[Stand confidently. Pause. Make eye contact.]**

> "Imagine you're an ICU nurse. It's 3 AM. You're monitoring six patients simultaneously. One of them is about to go into cardiac arrest in six hours — but nothing on the monitor looks alarming *yet*."
>
> "What if an AI could see what human eyes can't — and warn you hours in advance?"
>
> "That's **SentinAl** — a unified clinical AI platform that predicts patient deterioration, grades diabetic wounds, and detects strokes — all running **100% on your local machine**, with **zero patient data ever leaving the device**."

---

## SLIDE 2: THE PROBLEM (90 seconds)

**[Gesture toward the stat cards as you mention each number.]**

> "Let me give you three numbers that define why we built this."
>
> "**250,000** — that's the number of preventable in-hospital deaths every year in the US alone. 80% of those patients showed warning signs 6 to 24 hours before deterioration. But those signs were missed — because of alert fatigue, understaffing, and human limitations."
>
> "**60 million** — that's the number of diabetic patients in India at risk of foot complications. 85% of lower limb amputations were preceded by a treatable ulcer. A simple photograph, analyzed by AI, could have changed the outcome."
>
> "**1.9 million** — that's the number of neurons a stroke patient loses *every single minute*. In rural clinics where there's no radiologist to read a CT scan, those minutes become hours."

**[Pause for effect.]**

> "The problem isn't that we lack the medical knowledge. The problem is that we lack the *tools* to apply it at scale — especially in resource-constrained settings."

---

## SLIDE 3: OUR SOLUTION (90 seconds)

**[Transition with energy — this is the "reveal" moment.]**

> "So we built SentinAl — three AI modules, one platform, zero cloud dependency."
>
> "**Module 1: Vital Signs Monitor.** We built a Temporal Transformer — the same architecture behind GPT — adapted for clinical time-series. It takes 12 hours of vital sign data, processes 34 engineered clinical features, and predicts deterioration with an **AUROC of 0.996**. That's near-perfect discrimination."
>
> "**Module 2: Diabetic Foot Wound Grader.** Upload a photograph. Our EfficientNet-B0 model, validated by CLIP to reject non-medical images, classifies the wound into one of four Wagner grades — with **97% accuracy and an F1 of 0.97**."
>
> "**Module 3: CT Stroke Detector.** Upload a brain CT scan. Our model detects hemorrhagic stroke with an **AUROC of 0.982** and immediately triggers an emergency alert."

**[Lean in slightly.]**

> "But here's what ties it all together: after every prediction, SentinAl automatically recommends the right specialist, assigns an urgency level, and finds the nearest hospital — complete with phone number, distance, and a Google Maps link."

---

## SLIDE 4: HOW IT WORKS / ARCHITECTURE (60 seconds)

**[Keep this crisp — judges care about the "how" but don't want a lecture.]**

> "The workflow is simple: Upload, Validate, Predict, Alert, Refer. End-to-end in under 2 seconds."
>
> "What makes SentinAl fundamentally different from existing clinical AI?"
>
> "First — **it's 100% on-device**. Every model runs locally using PyTorch. No patient data touches a server. No API keys. No cloud bills. This makes us HIPAA and DISHA compliant by design."
>
> "Second — **it works offline**. After the initial setup, the app needs no internet for inference. Only the hospital finder uses a lightweight OpenStreetMap call."
>
> "Third — **it runs on consumer hardware**. GPU acceleration is supported but optional. This means a clinic with a basic laptop can run clinical AI."
>
> "And fourth — we integrated a **local LLM via Ollama** that explains every prediction in plain clinical language. No black boxes."

---

## SLIDE 5: RESULTS & METRICS (60 seconds)

**[This is your credibility slide. Slow down here.]**

> "Let me walk you through the numbers."
>
> "For vital signs prediction: **AUROC 0.996** on a dataset of 293,000 time-series rows from 7,000 patients. We tackled severe class imbalance — only 5.4% of windows showed deterioration — using Focal Loss and weighted sampling. Our optimized threshold is 0.841."
>
> "For wound grading: **97.05% accuracy, F1 macro 0.97** across 9,934 images in 4 Wagner grades. We solved a critical Grade 3 imbalance — originally only 40 training images — through targeted augmentation."
>
> "For stroke detection: **AUROC 0.982, accuracy 92.2%** on 2,501 CT scans. F1 for stroke class is 0.89, with cosine annealing and careful regularization."

**[Pause.]**

> "These aren't research paper numbers. These are *deployed* metrics from models running live in our Streamlit application."

---

## SLIDE 6: CLOSING (30 seconds)

**[Slow down. Be genuine.]**

> "SentinAl isn't just a hackathon project. It's a proof of concept for a future where every ICU, every rural clinic, and every community health worker has access to specialist-level AI — running on the hardware they already own, protecting the data their patients trust them with."
>
> "Three modules. Near-perfect metrics. Zero cloud. Complete privacy."
>
> "Thank you. I'm happy to take questions — or if you'd like, I can give you a live demo right now."

**[Smile. Wait for applause or questions.]**

---

## Q&A PREPARATION — Anticipated Questions

### "How does the Temporal Transformer compare to LSTM?"
> "We implemented both. The Transformer achieves comparable AUROC but with two key advantages: first, self-attention allows any hour to directly attend to any other hour — no information dilution over long sequences. Second, attention weights are interpretable — we can show *which hours* the model focused on, which matters for clinical trust."

### "What about class imbalance?"
> "Only 5.4% of windows showed deterioration — a 17:1 imbalance. We used three strategies: Focal Loss that downweights easy negatives, a WeightedRandomSampler for balanced mini-batches, and threshold optimization — our F1-maximizing threshold is 0.841, not the default 0.5."

### "Why not use a cloud-based model like GPT-4 for explanations?"
> "Two reasons: privacy and cost. Patient data should never leave the device. And cloud API calls add latency, cost, and an internet dependency. Ollama's qwen2.5:3b runs locally and is more than sufficient for structured clinical summaries."

### "How do you validate that uploaded images are actually medical?"
> "We use OpenAI's CLIP model in zero-shot mode. Before any inference, we classify the image against 8 categories — foot wound, healthy foot, animal, food, landscape, portrait, screenshot, random. Only images with >25% foot wound confidence proceed. This prevents garbage-in-garbage-out."

### "What's the training data source?"
> "For vitals: 7,000 de-identified patient records with 22 clinical measurements over 48-72 hours. For wounds: 9,934 annotated diabetic foot images across 4 Wagner grades. For stroke: 2,501 labeled brain CT scans. All training was done offline; no data is bundled with the app."

### "Can this actually be deployed in a hospital?"
> "The current form is a Streamlit prototype — production deployment would need EHR integration via HL7 FHIR, clinical validation trials, and regulatory clearance. But the core AI is production-grade and the architecture is designed for edge deployment from day one."

### "What's on the roadmap?"
> "Next: ECG arrhythmia detection and X-ray pneumonia classification. Future: federated learning across hospitals so models improve without sharing data, a mobile app for point-of-care, and multi-language clinical summaries for India's diverse linguistic landscape."

---

## LIVE DEMO SCRIPT (if time allows, ~3 minutes)

1. **Open the app:** `streamlit run app.py`
2. **Overview page:** "Here's our dashboard — three modules, performance metrics at a glance, system status showing GPU and Ollama availability."
3. **PS2 Demo:** "Let me load the demo patient — 48 hours of vital signs. Watch the 6-panel dashboard populate. See the sliding window risk scores. Here's a HIGH risk alert — the model detected early signs of sepsis."
4. **PS1 Demo:** "I'll upload a foot wound image. CLIP validates it's a real wound... EfficientNet classifies it as Wagner Grade 2 with 94% confidence. The system recommends a podiatrist and shows me the nearest clinic."
5. **Specialist Recommender:** "It auto-detected my location, found 3 hospitals within 15km, with phone numbers and Google Maps links."
6. **Ollama Chat:** "I can ask the LLM to explain — 'What does Wagner Grade 2 mean for this patient?' — and get a plain-language clinical summary, all running locally."

---

## GENERAL TIPS

- **Open strong.** The ICU nurse story hooks judges emotionally.
- **Lead with impact, not tech.** Judges remember "250K preventable deaths" more than "8-head self-attention."
- **Slow down on metrics.** AUROC 0.996 is exceptional — let it land.
- **Privacy is your moat.** Emphasize "zero cloud" at every opportunity.
- **End with a live demo offer.** Confidence signals that the product works.
- **Time yourself.** Cut the architecture slide if you're over time — metrics and story matter more.
