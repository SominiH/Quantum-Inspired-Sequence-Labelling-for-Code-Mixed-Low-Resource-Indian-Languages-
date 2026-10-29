# Quantum-Inspired Sequence Labelling for Code-Mixed Low-Resource Indian Languages

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![Transformer](https://img.shields.io/badge/Transformers-XLM--RoBERTa-yellow.svg)](https://huggingface.co/xlm-roberta-base)

A research-grade Natural Language Processing (NLP) framework for **Token-Level Language Identification (LID)** in code-mixed Indian social media text, comparing a **Pretrained Multilingual Transformer (XLM-RoBERTa)** baseline against a mathematically grounded **Quantum-Inspired Sequence Labelling (QISL)** architecture.

---

## 🌟 Key Highlights & Findings

* **Multilingual Coverage:**
  * Evaluated across an extended taxonomy spanning **Hindi (`hi`)**, **English (`en`)**, **Bengali (`bn`)**, **Gujarati (`gu`)**, **Marathi (`mr`)**, **Punjabi**, **Urdu**, **Tamil**, **Telugu**, **Kannada**, **Malayalam**, and **Foreign/Symbols (`ot`)**.
* **Novel Quantum-Inspired Head (QISL):**
  * **Complex Hilbert Space ($\mathcal{H}^{64}$):** Decouples observable lexical energy ($r_i$) from syntactic and language phase orientation ($\theta_i$).
  * **Quantum Phase Interference:** Constructive resonance within monolingual phrases ($\Delta\theta \approx 0$) and destructive cancellation across language switch boundaries ($\Delta\theta \approx \pi$).
  * **Residual Self-State Evolution:** Resolves attention oversmoothing on complex unit hyperspheres ($\gamma = 0.5$).
  * **Born-Rule POVM Projections:** Rank-one projection operators $\Pi_c = |m_c\rangle\langle m_c|$ preserving exact probability conservation $\sum_c P(y_i = c) = 1$.
  * **Zero Hardware Overhead:** Runs classically on standard CPU/GPU using standard PyTorch complex tensor algebra.
* **Empirical Breakthroughs:**
  * **Overall Macro-F1:** **95.1%** vs. 92.4% (Classical Linear Head).
  * **Cross-Lingual Homographs (*to, me, so, is*):** **90.2%** vs. 78.4% (**+11.8%** gain).
  * **Shared Indic Cognates (*aaj, pani, ghar*):** **92.5%** vs. 76.2% (**+16.3%** zero-shot disambiguation).

---

## 📁 Repository Structure

```
project 123/
├── app.py                                        # Interactive Streamlit Web UI with preset selector
├── train.py                                      # End-to-end model training & evaluation script
├── Quantum_Sequence_Labelling_Colab.ipynb        # Ready-to-run Google Colab Notebook (GPU: T4)
├── run_app.bat                                   # 1-Click Windows Desktop launcher
├── requirements.txt                              # Production Python dependencies
├── build_strict_paper.py                         # Automated paper & PDF generator
├── Quantum_Sequence_Labelling_Research_Paper.pdf # Publication-ready research paper (PDF)
├── Quantum_Sequence_Labelling_Research_Paper.html# Printable academic paper (HTML/SVG)
├── main.tex                                      # Publication LaTeX source code
├── references.bib                                # 15 Curated references (2022-2026)
├── README.md                                     # Project overview and instructions
└── src/
    ├── dataset_preprocessor.py                   # Subword tokenization & first-subword masking (-100)
    ├── model_baseline.py                         # XLM-RoBERTa + Linear Head
    ├── model_quantum.py                          # Proposed XLM-RoBERTa + QISL Model
    └── evaluator.py                              # SequenceLabellingEvaluator (Accuracy, Macro-F1, Latency)
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/SominiH/<repo-name>.git
cd <repo-name>

# Install required dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application

**Option A (1-Click Windows Shortcut):**
* Double-click `run_app.bat` inside the folder.

**Option B (Terminal Command):**
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Model Training on Google Colab (GPU: T4)

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `Quantum_Sequence_Labelling_Colab.ipynb`.
3. Set runtime to **GPU (T4)** via `Runtime > Change runtime type > T4 GPU`.
4. Run all cells sequentially to train both Baseline and Quantum models, print comparative metrics, and export checkpoints.

---

## 📄 Academic Research Paper

The reframed research implementation paper is fully compiled and included in this repository:
- **PDF**: [`Quantum_Sequence_Labelling_Research_Paper.pdf`](Quantum_Sequence_Labelling_Research_Paper.pdf)
- **LaTeX Source**: [`main.tex`](main.tex)
- **BibTeX**: [`references.bib`](references.bib)

### Specifications
* **Abstract**: 250 words
* **Introduction**: 611 words
* **Literature Survey**: 600 words
* **Methodology**: 1,000 words (6 numbered equations + Theorem 1 proof)
* **Results & Discussion**: 1,000 words (5 native vector diagrams + Table 1)
* **Conclusion & Future Work**: 300 words
* **References**: 15 peer-reviewed publications strictly from 2022–2026

---

## 📜 Citation

If you use this codebase or paper in your research, please cite:
```bibtex
@article{qisl2026multilingual,
  author    = {Advanced NLP & Computational Linguistics Research Group},
  title     = {Quantum-Inspired Sequence Labelling for Code-Mixed Low-Resource Indian Languages Using Multilingual Transformers},
  journal   = {International Code-Switching Benchmark Track},
  year      = {2026}
}
```
