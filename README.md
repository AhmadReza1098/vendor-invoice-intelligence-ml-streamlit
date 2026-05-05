# 📦 Vendor Invoice Intelligence System
**Freight Cost Prediction & Invoice Risk Flagging**

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Business Objectives](#business-objectives)
- [Data Sources](#data-sources)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Application](#application)
- [Project Structure](#project-structure)
- [How to Run This Project](#how-to-run-this-project)
- [Author & Contact](#author--contact)

---

<h2 id="project-overview">🚀 Project Overview</h2>

This project implements an **end-to-end machine learning system** designed to support finance teams by:
1. **Predicting expected freight cost** for vendor invoices to assist with budgeting.
3. **Flagging high-risk invoices** that require manual review due to abnormal cost, freight, or operational delivery patterns.

<h2 id="business-objectives">🎯 Business Objectives</h2>

- 📉 **Improved Cost Forecasting:** Accurately predict shipping costs based on invoice dollar amounts and quantities.
![Predicting expected freight cost](images/freight_cost_prediction.png)

- 🛡️ **Reduced Financial Leakage:** Automatically detect discrepancies between billed invoices and actually received items.
![Flagging high-risk invoices](images/invoice_manual_approval.png)

- ⚡ **Faster Finance Operations:** Streamline the approval process by auto-approving safe invoices and isolating risky ones for manual review.
  
<h2 id="data-sources">🗄️ Data Sources</h2>

Data is extracted from an internal SQLite database (`inventory.db`) consisting of:
- **`vendor_invoice` table:** Contains billed amounts, freight charges, and invoice dates.
- **`purchases` table:** Contains item-level receiving data, quantities, and PO dates.
Feature engineering was performed using SQL `WITH` clauses (CTEs) to aggregate item-level data and match it to final invoices.

<h2 id="exploratory-data-analysis">🔍 Exploratory Data Analysis</h2>

Key findings during EDA:
- Freight costs exhibit a non-linear relationship with total invoice dollars.
- High-risk invoices frequently show a mismatch between `invoice_dollars` and `total_item_dollars` (> $5 variance).
- Abnormal receiving delays (PO Date to Receiving Date > 10 days) correlate strongly with problematic invoices.

<h2 id="models-used">⚙️ Models Used</h2>

The pipeline leverages **Scikit-Learn** for modeling and preprocessing:
1. **Data Preprocessing:** `StandardScaler` to normalize feature distributions (saved as `scaler.pkl`).
2. **Regression (Freight Prediction):** 
- Linear Regressor (baseline)
- Decision Tree Regressor
- Random Forest Regressor (Final)
Random Forest Regressor optimized for continuous cost forecasting.
4. **Classification (Risk Flagging):**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier (fianl model with GridSearchCV)
Random Forest Classifier optimized via `GridSearchCV` (1,080 fits across 5-fold cross-validation) targeting `f1_score`.

<h2 id="evaluation-metrics">📊 Evaluation Metrics</h2>

- **Risk Flagging Model:** Evaluated using Accuracy, Precision, Recall, and F1-Score to minimize False Negatives (missing a risky invoice).
- **Freight Cost Model:** Evaluated using Mean Absolute Error (MAE) and R² Score.

<h2 id="application">🌐 Application</h2>

The project includes a fully interactive **Streamlit Web Application** for the finance team. 
Users can seamlessly toggle between two modules:
- **Freight Cost Prediction:** Input quantity and dollars to get instant freight estimates.
- **Invoice Manual Approval Flag:** Input invoice and item metrics to run live inference through the scaled Random Forest pipeline, returning an instant `SAFE` or `RISKY` verdict.

<h2 id="project-structure">📁 Project Structure</h2>

```text
├── data/
│   └── inventory.db                # Raw SQLite Database
├── models/
│   ├── predict_freight_model.pkl   # Trained Freight Model
│   ├── predict_flag_invoice.pkl    # Trained Risk Classifier
│   └── scaler.pkl                  # Feature Scaler
├── src/
│   ├── data_preprocessing.py       # SQL extraction and scaling logic
│   ├── modeling_evaluation.py      # Random Forest & GridSearch setup
│   └── train.py                    # Main pipeline orchestrator
├── inference/
│   ├── predict_freight.py          # Freight inference script
│   ├── predict_invoice.py          # Risk flagging inference script
│   └── app.py                      # Streamlit User Interface
└── README.md
