# 🏦 Client Intelligence Platform — Cetelem Portugal (BNP Paribas)

> **Capstone Project | MSc Data Science & Advanced Analytics — NOVA IMS**  
> Developed in partnership with **Cetelem Portugal (BNP Paribas Personal Finance)**

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Models & Methodology](#-models--methodology)
- [Project Structure](#-project-structure)
- [Notebooks Guide](#-notebooks-guide)
- [Data Setup](#-data-setup)
- [Notebook Execution Order](#️-notebook-execution-order)
- [Dashboard Setup](#-dashboard-setup)
- [Troubleshooting](#-troubleshooting)
- [Team](#-team)

---

## 🎯 Project Overview

This project delivers a **data-driven client retention system** for Cetelem Portugal's personal loan portfolio. The goal is to support the commercial retention team in identifying at-risk clients and recommending tailored actions — before churn or early settlement occurs.

The system combines three predictive components into a single, actionable Streamlit dashboard:

| Component                           | Description                                                           | Output |
|-------------------------------------|-----------------------------------------------------------------------|--------|
| **Churn Model**                     | Predicts the probability a client will not renew their loan in 30 days | Binary flag + probability |
| **Early Settlement (SAN) Model**    | Predicts whether a client will settle their loan ahead of schedule    | Binary flag + probability |
| **Survival Analysis (Churn) Model** | Predicts when clients will do something about the churn               | Binary flag + probability |
| **Client Segmentation**             | Groups clients into behavioural profiles using K-Means clustering     | Segment label + commercial action |

> ⚠️ The dashboard is a **decision-support tool** — it augments human judgment, it does not replace it.

---

## 🧠 Models & Methodology

### Churn Model
- **Algorithm:** Gradient Boosting Classifier
- **Target:** `IS_CHURN` (reframed as probability of renewal for dashboard use)
- **Key challenge:** Severe class imbalance — addressed via resampling and threshold calibration
- **Metric:** ROC-AUC

### Early Settlement (SAN) Model
- **Algorithm:** Gradient Boosting Classifier
- **Target:** `SAN` (binary: early settlement within a defined horizon)
- **Performance:** ROC-AUC ≈ 0.986
- **Metric:** ROC-AUC, Precision-Recall

### Client Segmentation
- **Algorithm:** K-Means Clustering
- **Features:** Behavioural and financial variables (contribution, credit score, outstanding balance, etc.)
- **Output:** Cluster profile mapped to a commercial action recommendation

### Data Sources
Four source tables were integrated for modelling:

| Table | Description |
|-------|-------------|
| `BDOSS` | Loan application and contract data |
| `CRC` | Credit bureau information |
| `CScore` | Internal credit scoring variables |
| `FAMA` | Financial and account movement data |

> 📌 Client identifier: `CONTRIB`

---

## 📁 Project Structure

```
Business_Case_BNP/
│
├── dashboard/                  # Streamlit retention dashboard
│   ├── app.py                  # Main application entry point
│   └── requirements.txt        # Python dependencies
│
├── guidelines/                 # Business case brief, data dictionary, instructions
│
├── notebooks/                  # Jupyter notebooks (full analysis pipeline)
│   ├── 1)*                     # Data understanding
│   ├── 2)*                     # Data preparation
│   ├── 3)*                     # Model preprocessing & survival analysis
│   └── 4)*                     # Segmentation & EDA
│
├── src/
│   └── code/                   # Utility scripts and helper functions
│
├── .gitignore
└── README.md
```

> ⚠️ The `data/` folder is **not tracked in Git** due to file size and data sensitivity. See [Data Setup](#-data-setup) below.

---

## 📓 Notebooks Guide

The notebooks follow the **CRISP-DM** methodology:

| Notebook | Phase | Description |
|----------|-------|-------------|
| `1)Data_Understanding.ipynb` | Data Understanding | Load raw data, inspect schema, assess data quality, initial distributions |
| `2a)Data_Preparation.ipynb` | Data Preparation | Missing value treatment, outlier handling, encoding, merging of source tables |
| `3a)Data_Preprocessing_Models_Churn.ipynb` | Data Preparation | Feature engineering and preprocessing pipeline specific to the Churn model |
| `3a)Data_Preprocessing_Models_Settlement.ipynb` | Data Preparation | Feature engineering and preprocessing pipeline specific to the SAN (Early Settlement) model |
| `3b)Data_Preprocessing_EDA_Clusters.ipynb` | Data Preparation | Preprocessing and exploratory analysis for the segmentation component |
| `3c)Survival_Analysis_Churn.ipynb` | Modelling | Survival analysis applied to churn — time-to-event modelling |
| `4c)EDA.ipynb` | Data Understanding | Deep-dive exploratory analysis supporting model and segment interpretation |
| `4d)Clusters.ipynb` | Modelling / Evaluation | K-Means clustering — elbow method, cluster profiling, commercial action mapping |

> 💡 Trained models are saved as `.pkl` files and loaded by the dashboard at runtime.

---

## 💾 Data Setup

Raw data is **not included** in this repository. Follow these steps to set it up locally:

1. Access the shared drive: [Google Drive — Project Data](https://drive.google.com/drive/folders/1zHISR9-N6kzt3PZYAlH9gorJrNw04EfS?hl=pt-br)
2. Download `data.zip`
3. Extract it into the project root so the structure looks like:

```
Business_Case_BNP/
├── data/
│   ├── raw/          <- Original Parquet files
│   ├── converted/    <- Processed CSV files
│   └── prepared/     <- Final scored dataset used by the dashboard (auto-generated by notebooks)
```

4. Create the `data/prepared/` folder manually if it doesn't exist — the notebooks will write outputs there.

---

## ▶️ Notebook Execution Order

The notebooks must be run **in this exact order** to generate all intermediate files and the final dataset loaded by the dashboard.

```
1.  2a)Data_Preparation.ipynb
          |
          v
2.  3a)Data_Preprocessing_Models_Settlement.ipynb
          |
          v
3.  3a)Data_Preprocessing_Models_Churn.ipynb
          |
          v
4.  3b)Data_Preprocessing_EDA_Clusters.ipynb
          |
          v
5.  4d)Clusters.ipynb
          |
          v
    [OK]  Final scored dataset saved to data/prepared/
          (this file is loaded by the dashboard at runtime)
```

> ⚠️ Do not skip steps or run notebooks out of order — each one depends on outputs from the previous step.

---

## 🚀 Dashboard Setup

### Prerequisites

- **Python 3.9+** — [download here](https://www.python.org/downloads/)
- **pip** (bundled with Python)

Verify your installation:
```bash
python --version
pip --version
```

---

### 1. Clone the repository

```bash
git clone https://github.com/RuiSousaFerreiraIMS/Business_Case_BNP
cd Business_Case_BNP/dashboard
```

---

### 2. Create a virtual environment (recommended)

```bash
# Create
python -m venv venv

# Activate — macOS/Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Installs: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `plotly`

---

### 4. Launch the dashboard

```bash
streamlit run app.py
```

The dashboard opens automatically at **http://localhost:8501**

> If it doesn't open, copy the URL from the terminal and paste it into your browser.

---

### 5. Stop the dashboard

```
Ctrl + C
```

---

## 🔧 Troubleshooting

| Issue | Fix |
|-------|-----|
| `streamlit: command not found` | Activate the virtual environment (step 2) |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| Module not found error | Re-run `pip install -r requirements.txt` |
| Browser doesn't open | Manually navigate to http://localhost:8501 |
| Model file not found | Ensure `.pkl` files are present in the expected path (check `app.py`) |
| Dashboard shows no data | Make sure all notebooks were run in order and `data/prepared/` exists |

---

## 👥 Team

Developed by MSc students at **NOVA IMS — Information Management School**, Lisbon.

| Name |
|------|
| Rui Ferreira |
| Bruna Sousa |
| Alexandre Coelho |
| Maria Pimentel |
| Elias Karle |

> Project developed under the **Business Cases with Data Science** course, in collaboration with Cetelem Portugal (BNP Paribas Personal Finance).