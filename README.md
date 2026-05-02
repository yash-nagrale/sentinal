# 🛡️ SentinAL - Clinical AI Monitoring System

> **Privacy-First, Edge-Deployed Healthcare Intelligence**  
> Early detection of patient deterioration before it becomes critical

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

**SentinAL** is a comprehensive clinical AI platform designed to detect health deterioration in real-time. Rather than analyzing isolated symptoms, SentinAL operates as a **connected multi-modal healthcare monitoring system** that watches for multiple physiological warning signs simultaneously.

### Why SentinAL?

Clinical deterioration often presents warning signs **hours before a crisis**, but these are frequently missed due to:
- ⏱️ Monitoring gaps during high-volume patient care
- 📊 Difficulty correlating multiple vital signs in real-time
- 🏥 Resource constraints in healthcare facilities

**SentinAL** stands guard by combining cutting-edge machine learning with privacy-preserving edge deployment.

---

## ✨ Key Features

- 🎯 **Multi-Modal Monitoring**
  - Time-series vital sign analysis for patient deterioration detection
  - Computer vision for diabetic foot ulcer identification
  - Neurological risk assessment

- 🔒 **Privacy-First Architecture**
  - Edge-deployed models (runs locally)
  - No cloud dependency
  - HIPAA-friendly design

- ⚡ **Real-Time Detection**
  - Transformer-based deep learning models
  - Multiple detection models for robustness
  - Instant risk stratification

- 🖥️ **User-Friendly Interface**
  - Interactive Streamlit web application
  - Intuitive patient monitoring dashboard
  - Detailed risk analysis and recommendations

- 🤖 **AI-Powered Insights**
  - LLM-based clinical recommendations (via Ollama)
  - Explainable AI outputs
  - Healthcare provider decision support

---

## 🏗️ Architecture

SentinAL combines three complementary AI systems:

| Component | Purpose | Model |
|-----------|---------|-------|
| **PS1: Wound Detection** | Diabetic foot ulcer classification | Transformer-based image classifier |
| **PS2: Deterioration Monitoring** | Time-series vital sign analysis | Multi-task LSTM/Transformer |
| **PS5: Severity Classification** | Risk stratification & triage | Ensemble classifier |

---

## 🛠️ Tech Stack

- **ML/AI Framework**: PyTorch, Transformers, Scikit-learn
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **LLM Integration**: Ollama (local LLM support)
- **Visualization**: Plotly, Matplotlib
- **Image Processing**: Pillow, OpenCV

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB+ recommended for model inference)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd sentinal
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   ```

3. **Activate the environment**:

   **Windows (PowerShell)**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   **Windows (CMD)**:
   ```cmd
   .venv\Scripts\activate.bat
   ```

   **macOS / Linux**:
   ```bash
   source .venv/bin/activate
   ```

4. **Upgrade pip and install dependencies**:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Launch the application**:
   ```bash
   streamlit run app.py
   ```

   The app will open in your browser at `http://localhost:8501`

---

## 📊 Data & Models

### Pre-trained Models

The repository includes pre-trained models in `/models/`:
- `best_transformer.pt` - Main deterioration detection model
- `best_ps1.pt` - Foot wound classification model
- `best_ps5_classifier.pt` - Severity classification model
- `threshold_transformer.txt` - Detection threshold

### Processed Data

Pre-processed training/validation data is stored in `/data/processed/`:
- `X_seq_train.npy`, `X_seq_val.npy` - Time-series features
- `X_static_train.npy`, `X_static_val.npy` - Static patient features
- `y_train.npy` - Training labels

---

## 🔨 Building from Scratch

If you need to rebuild models and datasets locally:

```bash
# 1. Preprocess raw data
python src/preprocess.py

# 2. Train all models
python src/train.py              # General model training
python src/ps1_train.py          # Foot wound detection
python src/ps5_train.py          # Severity classification

# 3. Run predictions
python src/predict.py
```

Outputs will be automatically placed in:
- `data/processed/` - Processed features
- `models/` - Trained model weights
- `outputs/` - Predictions and results

---

## 📁 Project Structure

```
sentinal/
├── app.py                          # Main Streamlit application entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── data/
│   ├── train.csv                   # Raw training data
│   ├── val_no_labels.csv           # Validation set
│   └── processed/                  # Pre-processed features
│       ├── X_seq_*.npy             # Time-series features
│       ├── X_static_*.npy          # Static features
│       └── y_train.npy             # Labels
│
├── models/                         # Pre-trained model weights
│   ├── best_transformer.pt
│   ├── best_ps1.pt
│   ├── best_ps5_classifier.pt
│   └── threshold_transformer.txt
│
├── outputs/                        # Prediction results
│   ├── predictions_all_windows.csv
│   └── predictions_per_patient.csv
│
├── src/                            # Source code
│   ├── config.py                   # Configuration management
│   ├── preprocess.py               # Data preprocessing pipeline
│   ├── train.py                    # Model training
│   ├── model.py                    # Neural network architectures
│   ├── evaluate.py                 # Model evaluation metrics
│   ├── predict.py                  # Inference pipeline
│   ├── recommender.py              # Clinical recommendations engine
│   ├── ps1_*.py                    # Foot wound detection module
│   ├── ps5_*.py                    # Severity classification module
│   └── ui/                         # Streamlit UI components
│       ├── home_ui.py              # Home page
│       ├── ps1_ui.py               # Foot wound detection UI
│       ├── ps5_ui.py               # Severity classification UI
│       └── components_ui.py        # Reusable UI components
│
└── pitch deck/                     # Project documentation
    ├── Script.md                   # Presentation script
    └── SentinAL_Technical_Report.ipynb  # Technical details
```

---

## 🎯 Usage Guide

### Running the Web Application

```bash
streamlit run app.py
```

The interactive dashboard provides:
- **Patient Data Upload**: Import patient vital signs or medical images
- **Real-Time Risk Assessment**: Get instant deterioration risk scores
- **Clinical Insights**: AI-powered recommendations based on detected risks
- **Trend Analysis**: Visualize patient health trajectories
- **Exportable Reports**: Generate patient monitoring reports

### API/Script Usage

```python
from src.model import load_model
from src.predict import predict_deterioration

# Load pre-trained model
model = load_model('models/best_transformer.pt')

# Make predictions on new data
risk_score, confidence = predict_deterioration(patient_data, model)
```

---

## 🔬 Model Performance

| Task | Model | Accuracy | AUC-ROC |
|------|-------|----------|---------|
| Patient Deterioration | Transformer | ~94% | ~0.98 |
| Foot Wound Detection | CNN Transformer | ~92% | ~0.96 |
| Severity Classification | Ensemble | ~89% | ~0.94 |

---

## 📝 Configuration

Edit `src/config.py` to customize:
- Model thresholds and hyperparameters
- Ollama LLM settings
- Data directory paths
- Device selection (CPU/GPU)

Key settings:
```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OLLAMA_MODELS = ["qwen2.5:3b", "codellama:latest"]
FOOT_WOUND_THRESHOLD = 0.25  # Adjust detection sensitivity
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## ⚠️ Disclaimer

**SentinAL is a research and demonstration project.** It should not be used for actual clinical decision-making without proper validation, regulatory approval, and clinical oversight. Always consult with qualified healthcare professionals.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Team

**Team 2Infinity**
- Yash Nagrale
- Suraj
- Shubham
- Akshay

---

## 📞 Support & Questions

For questions, issues, or suggestions:
- Open an [Issue](https://github.com/yourusername/sentinal/issues)
- Check the [Technical Report](pitch%20deck/SentinAL_Technical_Report.ipynb)
- Review the [Presentation Script](pitch%20deck/Script.md)

---

## 🎓 Citation

If you use SentinAL in your research, please cite:
```
@project{sentinal2024,
  title={SentinAL: Privacy-First Clinical AI Monitoring System},
  author={Nagrale, Y. and Team 2Infinity},
  year={2024}
}
```

---

**Last Updated**: May 2024  
**Status**: Active Development
