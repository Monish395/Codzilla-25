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

---

![WhatsApp Image 2025-08-08 at 10 12 41_1d4778d1](https://github.com/user-attachments/assets/b613a6b3-c29d-4ad5-a88f-59f80f9ea28d)

---

![WhatsApp Image 2025-08-08 at 10 12 58_a7731852](https://github.com/user-attachments/assets/9e1187fa-e1ca-4dfc-8533-ea0bf51cbc42)

---

![WhatsApp Image 2025-08-08 at 10 13 19_08f66d04](https://github.com/user-attachments/assets/f8b55a86-19b1-4f7b-ad58-f9c95e590cae)

---

![WhatsApp Image 2025-08-08 at 10 14 16_8cced9f7](https://github.com/user-attachments/assets/b850c733-bb3c-4e0b-89f3-d6c97a06f3f0)

---

![WhatsApp Image 2025-08-08 at 10 14 35_677fc157](https://github.com/user-attachments/assets/d3f3dc73-12df-4e2c-b22d-bd24f8045859)

---

![WhatsApp Image 2025-08-08 at 10 15 03_e007a3b7](https://github.com/user-attachments/assets/ca2603c5-75cb-42b4-81dd-9734667003ad)

---

![WhatsApp Image 2025-08-08 at 10 15 19_ef74367a](https://github.com/user-attachments/assets/209c4fc0-296b-4caf-a9d9-c71e3ade2f9d)

---

![WhatsApp Image 2025-08-08 at 10 15 36_3cfaa835](https://github.com/user-attachments/assets/efcfde9f-aabd-418e-b398-6722047989d9)

---

![WhatsApp Image 2025-08-08 at 10 16 00_89e1656e](https://github.com/user-attachments/assets/b5e73ce7-be0a-4ece-9159-a43ddaddfbce)

---

![WhatsApp Image 2025-08-08 at 10 16 36_ea024dc4](https://github.com/user-attachments/assets/b3ad77fc-8d93-4ddd-9a72-da950361c6c0)



