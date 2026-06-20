# Customer Churn Prediction

A complete, end-to-end machine learning project that predicts customer churn for a telecom provider using the IBM Telco Customer Churn dataset. The project covers the full data science lifecycle — from raw data exploration to a business-ready set of retention recommendations.

## Project Overview

Customer churn is one of the most expensive problems a subscription-based business can face: it is far cheaper to retain an existing customer than to acquire a new one. This project builds a classification model that flags customers who are likely to churn, and translates the model's findings into concrete actions the business can take to reduce churn.

The analysis is organized into six phases, all contained in a single Jupyter notebook:

1. **Data Exploration** — initial inspection of the dataset (shape, types, missing values, duplicates, target balance).
2. **Exploratory Data Analysis (EDA)** — visual analysis of how churn relates to demographics, contract type, billing, and service usage.
3. **Data Preprocessing** — cleaning, encoding, scaling, and splitting the data for modeling.
4. **Model Training** — training four classification algorithms.
5. **Model Evaluation** — comparing models on accuracy, precision, recall, F1, and ROC-AUC.
6. **Feature Importance & Business Insights** — identifying the key churn drivers and turning them into retention recommendations.

## Dataset

The project uses the **IBM Telco Customer Churn** dataset — 7,043 customer records across 21 features, including demographics (gender, senior citizen status, dependents), account information (tenure, contract type, payment method), billing (monthly/total charges), and subscribed services (internet, phone, streaming, security, support).

- **Target variable:** `Churn` (Yes / No)
- **Class balance:** ~73% retained, ~27% churned
- **Data quality:** no missing values or duplicate rows in the raw data; 11 rows had blank `TotalCharges` values (new customers with zero tenure) that were dropped during preprocessing.

> Note: the raw CSV (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) is not included in this repository. To reproduce the notebook, download the dataset and place it in a `data/` folder at the project root.

## Methods

### Preprocessing
- Dropped the non-predictive `customerID` column.
- Converted `TotalCharges` from text to numeric and dropped the resulting null rows.
- Encoded the target (`Yes`/`No` → `1`/`0`).
- One-hot encoded all categorical features (`drop_first=True` to avoid multicollinearity).
- Split the data 80/20 into train and test sets, stratified on the target to preserve class balance.
- Standardized features for the Logistic Regression model using `StandardScaler` (tree-based models were trained on the unscaled data, since they don't require it).

### Models Trained
| Model | Type |
|---|---|
| Logistic Regression | Linear |
| Decision Tree | Tree-based |
| Random Forest | Ensemble (bagging) |
| Gradient Boosting | Ensemble (boosting) |

All models were trained with a fixed `random_state=42` for reproducibility.

## Results

Models were evaluated on the held-out test set (1,407 customers) using five metrics, with **ROC-AUC** treated as the primary metric since it best captures how well a model separates churners from non-churners under class imbalance.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| **Gradient Boosting** | 0.795 | 0.638 | 0.532 | 0.580 | **0.841** |
| Logistic Regression | 0.804 | 0.648 | 0.575 | 0.609 | 0.836 |
| Random Forest | 0.790 | 0.626 | 0.519 | 0.567 | 0.816 |
| Decision Tree | 0.719 | 0.470 | 0.463 | 0.466 | 0.637 |

**Gradient Boosting** was selected as the best-performing model, narrowly outperforming Logistic Regression on ROC-AUC. The single Decision Tree underperformed the ensemble methods across every metric, as expected.

## Key Business Insights

Feature importance analysis on the best model (Gradient Boosting) surfaced a clear churn profile. The customers most likely to churn:

- Have **low tenure** (newer customers) — the single strongest churn signal, by a wide margin
- Subscribe to **Fiber optic internet**, despite it being the premium, higher-priced product
- Pay via **electronic check** rather than automatic billing — a stronger signal than contract type itself
- Are on a **month-to-month contract** rather than a one- or two-year term
- Have **not** opted into **Online Security**

### Recommendations
1. Build a structured onboarding/retention program for customers in their first months of tenure — tenure is the single biggest lever available.
2. Review pricing, reliability, and support for the Fiber optic segment specifically, since it's the second strongest churn driver and a large, high-value segment to lose.
3. Encourage migration away from manual electronic check payments toward automatic billing, since this is a stronger predictor than contract type.
4. Incentivize customers to move from month-to-month to longer-term contracts.
5. Promote security and support add-ons as part of new customer bundles.
6. Use the trained model to score the active customer base monthly and prioritize retention outreach by predicted churn risk and revenue at stake.

Full reasoning and supporting analysis are documented in [`Project_notes.md`](Project_notes.md) and in Phase 6 of the notebook.

## Repository Structure

```
.
├── app/                            # Application code
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw Telco Customer Churn dataset
├── notebooks/
│   ├── models/                     # Saved trained models (generated when notebook is run)
│   │   ├── decision_tree.pkl
│   │   ├── gradient_boosting.pkl
│   │   ├── logistic_regression.pkl
│   │   └── random_forest.pkl
│   ├── model_comparison.csv        # Exported model evaluation metrics
│   └── Notebook.ipynb              # Full analysis: EDA → preprocessing → modeling → insights
├── src/                            # Source code
├── .gitignore
├── Project_notes.md                # Phase-by-phase write-up of findings and results
└── README.md
```

## Tech Stack

- **Python 3**
- **pandas** / **numpy** — data manipulation
- **matplotlib** — visualization
- **scikit-learn** — preprocessing, modeling, and evaluation
- **joblib** — model persistence

## Getting Started

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd customer-churn-prediction
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn joblib
   ```
3. Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
4. Open and run `Notebook.ipynb` from top to bottom.

## Future Improvements

- Hyperparameter tuning (e.g. `GridSearchCV` / `RandomizedSearchCV`) to push ROC-AUC further.
- Address class imbalance directly with techniques like SMOTE or class weighting.
- Add SHAP values for more granular, per-customer explainability.
- Package the best model behind a simple API or dashboard for live scoring.

## Author

Built as a portfolio data science project demonstrating the full workflow from raw data to actionable business recommendations.
