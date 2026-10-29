# Quantum-Inspired Sequence Labelling for Code-Mixed Indian Languages

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![Transformer](https://img.shields.io/badge/Transformers-XLM--RoBERTa-yellow.svg)](https://huggingface.co/xlm-roberta-base)

An interactive, high-performance Natural Language Processing (NLP) application for **Token-Level Language Identification (LID)** in code-mixed Indian social media text. The system integrates a **Streamlit frontend** with a **Quantum-Inspired Sequence Labelling (QISL)** deep learning backend built atop **XLM-RoBERTa**.

---

## 🌟 Features & Highlights

### 🖥️ Frontend (Interactive Streamlit Interface)
* **Real-Time Word-by-Word Tagging**: Instantly identifies language switches as users type.
* **12 Multi-Family Presets**: One-click sample sentences covering Hindi, Bengali, Gujarati, Marathi, Tamil, Telugu, Kannada, Malayalam, and Spanish.
* **Color-Coded Visual Badges**: Visual indicators for detected languages with confidence scores.
* **Side-by-Side Model Comparison**: Real-time consensus and divergence analysis comparing Classical Linear vs. Quantum-Inspired predictions.
* **Sentence Composition Breakdown**: Visual metric badges summarizing the percentage distribution of languages in the input.

### 🧠 Backend (Quantum-Inspired NLP Architecture)
* **Complex Hilbert Space ($\mathcal{H}^{64}$)**: Maps contextual representations into complex vector space, separating lexical energy ($r$) from language phase orientation ($\theta$).
* **Quantum Phase Interference**: Generates constructive reinforcement within monolingual phrases ($\Delta\theta \approx 0$) and destructive cancellation across language switch boundaries ($\Delta\theta \approx \pi$).
* **Residual Self-State Evolution**: Eliminates attention oversmoothing and preserves individual token identities.
* **Born-Rule POVM Projections**: Rank-one POVM measurement operators $\Pi_c = |m_c\rangle\langle m_c|$ guaranteeing mathematical probability conservation.
* **Classical Execution**: Fully implemented in PyTorch using standard complex tensor algebra—runs seamlessly on standard CPU or GPU hardware without quantum processors.

---

## 📁 Repository Structure

```
project/
├── app.py                                 # Interactive Streamlit Web UI (Frontend)
├── run_app.bat                            # 1-Click Desktop Launcher for Windows
├── train.py                               # Standalone Model Training & Benchmark Script
├── Quantum_Sequence_Labelling_Colab.ipynb # Ready-to-run Google Colab Notebook (GPU: T4)
├── requirements.txt                       # Production Python Dependencies
├── push_to_github.bat                     # 1-Click GitHub Synchronization Tool
├── README.md                              # Application Documentation
└── src/                                   # Backend Engine
    ├── model_quantum.py                   # XLM-RoBERTa + QISL Architecture
    ├── model_baseline.py                  # XLM-RoBERTa + Linear Head Baseline
    ├── dataset_preprocessor.py            # SentencePiece Alignment & Subword Masking (-100)
    └── evaluator.py                       # Sequence Labelling Metrics (Accuracy, Macro-F1, Latency)
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/SominiH/Quantum-Inspired-Sequence-Labelling-for-Code-Mixed-Low-Resource-Indian-Languages-.git
cd Quantum-Inspired-Sequence-Labelling-for-Code-Mixed-Low-Resource-Indian-Languages-

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Web Application

**Option A (1-Click Windows Shortcut):**
* Double-click `run_app.bat` inside the project folder.

**Option B (Terminal Command):**
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser to interact with the model.

---

### 3. Model Training on Google Colab (GPU: T4)

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `Quantum_Sequence_Labelling_Colab.ipynb`.
3. Set hardware accelerator: `Runtime > Change runtime type > T4 GPU`.
4. Run all cells sequentially to train both the Classical Baseline and Quantum-Inspired models, evaluate comparative metrics, and export trained PyTorch checkpoints (`baseline_xlmr_lid.pt` and `quantum_xlmr_lid.pt`).

---

## 📊 Supported Language Classes

| Code | Language | Script Formats |
| :---: | :--- | :--- |
| `hi` | **Hindi** | Romanized / Hinglish (*mujhe*, *khushi*) & Devanagari (*हिन्दी*) |
| `en` | **English** | Standard vocabulary (*today*, *complete*, *project*) |
| `bn` | **Bengali** | Romanized / Benglish (*ami*, *bolchi*) & Eastern Nagari (*বাংলা*) |
| `gu` | **Gujarati** | Romanized (*mane*, *aaje*) & Gujarati script (*ગુજરાતી*) |
| `mr` | **Marathi** | Romanized (*mala*, *jhala*) & Devanagari (*मराठी*) |
| `ot` | **Other / Symbols** | Punctuation, numbers, emojis, and foreign tokens |

---

## 🛠️ Tech Stack

* **Frontend**: Streamlit 1.40+ (Python)
* **Deep Learning Framework**: PyTorch 2.x
* **Pretrained Backbone**: Hugging Face Transformers (`xlm-roberta-base`)
* **Mathematical Operations**: Complex Hilbert Tensor Operations, Born-Rule POVM Projections
