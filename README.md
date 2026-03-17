# Business_Case_BNP

## Project Setup

To set up the project locally, first clone the repository:

```bash
git clone https://github.com/RuiSousaFerreiraIMS/Business_Case_BNP
cd Business_Case_BNP
```

## Project Structure

- `data/` – Contains all project data  
  - `raw/` – Original Parquet files  
  - `converted/` – Processed CSV files ready for analysis  

- `guidelines/` – Project instructions, business case PDF, and data dictionary  

- `notebooks/` – Jupyter notebooks for exploration and analysis  

- `src/` – Source code  
  - `code/` – Main scripts  
  - `features/` – Python functions and utility scripts  
  - `models/` – (Optional) saved model files  

## Data Setup

The project does **not include raw data in the Git repository** due to file size.  

To get started:

1. Access the shared Google Drive/OneDrive link for the case: https://drive.google.com/drive/folders/1zHISR9-N6kzt3PZYAlH9gorJrNw04EfS?hl=pt-br  
2. Download the file `data.zip`.  
3. Extract the contents into the `data/` folder in the root of the project, replacing the empty `data/` folder in the repository.
---
# 🏦 Client Intelligence Dashboard — Setup Guide

## Prerequisites

- **Python 3.9 or higher** installed ([python.org](https://www.python.org/downloads/))
- **pip** available (bundled with Python by default)

Check your versions in the terminal:
```bash
python --version
pip --version
```

---

## 1. Get the files

Place both files in the same folder on your machine:

```
📁 dashboard/
├── app.py
└── requirements.txt
```

---

## 2. Create a virtual environment (recommended)

```bash
# Create environment
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

---

## 3. Install dependencies

Inside the `dashboard/` folder, run:

```bash
pip install -r requirements.txt
```

This will automatically install: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `joblib`, and `plotly`.

---

## 4. Launch the dashboard

```bash
streamlit run app.py
```

Your browser will open automatically at **http://localhost:8501**

> If it doesn't open on its own, copy the link from the terminal and paste it into your browser.

---

## 5. Stop the dashboard

In the terminal where Streamlit is running, press:

```
Ctrl + C
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `streamlit: command not found` | Activate the virtual environment (step 2) |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| Module not found error | Re-run `pip install -r requirements.txt` |
| Browser doesn't open | Manually go to http://localhost:8501 |