# AI-Powered Fraud Detection System

> A real-time, explainable fraud detection application built with XGBoost and Streamlit.

---

## 🚀 Project Overview

This project demonstrates an end-to-end machine learning pipeline for detecting fraudulent financial transactions. It includes:

* **Data preprocessing** of transactions.
* **Modeling** using **XGBoost** for handling imbalanced classification.
* **Explainability** with **SHAP** waterfall plots.
* **Streamlit UI** for analysts to upload data, review flagged transactions, and visualize model explanations.

Ideal for hackathons and proof-of-concept deployments in financial domains.

---

## 📂 Repository Structure

```
├── data/
│   └── AIML dataset.csv         # Sample dataset
├── model/
│   └── fraud_detection_pipeline.pkl  # Serialized pipeline
├── app/
│   └── enhanced_app.py        # Streamlit application code
├── sample_fraud_test.csv      # Example test cases
└── README.md                  # Project documentation
```

---

## 🛠️ Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Monish395/Codzilla-25.git
   cd hackathon
   ```

2. **Create and activate a Python environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate       # Linux / macOS
   venv\Scripts\activate        # Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   > **requirements.txt** should include:
   >
   > ```text
   > pandas
   > numpy
   > scikit-learn
   > xgboost
   > shap
   > streamlit
   > joblib
   > ```

---

## 🧠 Model Details

* **Algorithm**: `XGBClassifier` with `eval_metric='logloss'` and `use_label_encoder=False`.
* **Preprocessing**:

  * **Numeric features** scaled via `StandardScaler`.
  * **Categorical features** one-hot encoded with `drop='first'`.
* **Handling class imbalance**: threshold tuning (default `0.5`) and stratified train-test split.
* **Evaluation**: Classification report, confusion matrix, precision/recall/F1.
* **Explainability**: SHAP waterfall plots for individual predictions.

---

## 📱 Streamlit Application

Launch the interactive UI for analysis:

```bash
cd hackathon
streamlit run enhanced_app.py
```

**Features**:

* Upload transaction CSV/XLSX.
* View flagged transactions in a table.
* Generate SHAP explanations for selected transactions.

---

## ⚙️ Usage Example

1. Start the Streamlit app:

   ```bash
   streamlit run hackathon/enhanced_app.py
   ```
2. In the browser, upload **`AIML dataset.csv`**.
3. Review the highlighted fraud cases and explore explanations.

---

## 🔮 Future Work

* **Continuous Learning**: Retrain pipeline on analyst feedback.
* **API deployment**: Wrap model in a RESTful service (FastAPI/Flask).
* **Advanced UI**: Dash or React dashboard with live streaming of transactions.
* **Feature Engineering**: Add velocity-based and geographic anomaly features.

---

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for:

* Bug fixes
* New features or UI improvements
* Performance tuning and hyperparameter search
