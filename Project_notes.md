# Project Notes — Customer Churn Prediction

Running notes on what happened, and what I found, after running each phase of the notebook.

## Phase 1: Data Exploration

Running the data_exploration phase, found out that the data had no missing values.

- The dataset has **7,043 rows and 21 columns**.
- No null values anywhere, and `(df == "")` confirms there are also no hidden empty strings.
- No duplicate rows.
- The target (`Churn`) is imbalanced: **73.5% "No"** vs **26.5% "Yes"** — about 1 in 4 customers churned. This imbalance is why I prioritized ROC-AUC over plain accuracy later on when comparing models.

## Phase 2: Exploratory Data Analysis (EDA)

Ran the EDA phase to understand *which* customers are more likely to churn before building any model.

- **Contract type** stood out immediately — month-to-month customers churn far more often than one-year or two-year customers. This was the clearest visual signal in the whole EDA.
- **Tenure** matters a lot: the tenure histogram and the tenure-by-churn boxplot both show that customers who churn tend to have much shorter tenure than those who stay.
- **Monthly Charges** also separates the two groups — churners tend to pay more per month on average than non-churners.
- **Internet Service**: customers on Fiber optic churn more than DSL customers or those with no internet service, despite Fiber optic being the premium offering.
- **Payment Method**: customers paying via electronic check show higher churn than those on automatic payment methods (credit card / bank transfer).
- Demographic factors (gender, partner, dependents, senior citizen) showed weaker relationships with churn compared to contract, tenure, and billing-related features.
- The numeric correlation matrix (`SeniorCitizen`, `tenure`, `MonthlyCharges`) showed only weak-to-moderate correlations between these three on their own, confirming that churn is driven by a *combination* of factors rather than any single numeric variable — which is part of why a model is useful here rather than just reading off one chart.

> Note: the EDA charts make contract type look like the single most dramatic split visually, but Phase 6's feature importance analysis tells a more precise story — see below.

## Phase 3: Data Preprocessing

Prepared the dataset for machine learning.

- Dropped `customerID` — it's a unique identifier with no predictive value.
- `TotalCharges` was stored as text. Converting it to numeric with `pd.to_numeric(..., errors="coerce")` revealed **11 rows** with missing values — these turned out to be brand-new customers with `tenure = 0`, so `TotalCharges` was blank. Dropped these 11 rows, leaving **7,032 rows**.
- Encoded the target: `Yes → 1`, `No → 0`. Class counts after cleaning: **5,163 "No"** vs **1,869 "Yes"**.
- One-hot encoded all categorical columns with `drop_first=True`, which expanded the dataset to **31 columns** (30 features + target).
- Split the data 80/20 into train/test sets using `stratify=y` so both sets keep the same ~73/27 churn ratio. Final shapes:
  - `X_train`: (5625, 30), `X_test`: (1407, 30)
  - `y_train`: (5625,), `y_test`: (1407,)
- Scaled the features with `StandardScaler` for use with Logistic Regression (tree-based models were trained on the unscaled versions, since they don't need scaling).

## Phase 4: Model Training

Trained four classification models: **Logistic Regression**, **Decision Tree**, **Random Forest**, and **Gradient Boosting** — all with `random_state=42` for reproducibility. Logistic Regression was trained on the scaled features; the three tree-based models were trained on the original (unscaled) features. All four trained successfully, and each trained model was saved to disk with `joblib` under a `models/` folder for later reuse without retraining.

## Phase 5: Model Evaluation

Evaluated all four models on the test set using Accuracy, Precision, Recall, F1 Score, and ROC-AUC (results saved to `model_comparison.csv`):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8038 | 0.6476 | 0.5749 | 0.6091 | 0.8357 |
| Decision Tree | 0.7186 | 0.4701 | 0.4626 | 0.4663 | 0.6366 |
| Random Forest | 0.7896 | 0.6258 | 0.5187 | 0.5673 | 0.8165 |
| Gradient Boosting | 0.7953 | 0.6378 | 0.5321 | 0.5802 | 0.8407 |

Sorting by ROC-AUC (the metric I trust most here, given the class imbalance) puts **Gradient Boosting first (0.841)**, just ahead of Logistic Regression (0.836). Random Forest follows (0.816), and the single Decision Tree trails well behind everything else (0.637) — unsurprising, since it's the only model here without any ensembling to reduce overfitting.

Looking at the per-class classification reports, all models are noticeably better at identifying customers who **stay** (class 0) than customers who **churn** (class 1) — recall on the churn class tops out around 0.57 (Logistic Regression). This is the expected effect of the 73/27 class imbalance, and it's worth keeping in mind: the model will miss a meaningful fraction of actual churners, so it should be used as a *risk-ranking* tool (e.g. for prioritizing retention outreach) rather than a hard yes/no decision.

## Phase 6: Feature Importance and Business Insights

**Goal:** figure out which features matter most, why customers are actually leaving, and what the business should do about it.

**Picking the model:** Using the ROC-AUC ranking from Phase 5, **Gradient Boosting** was selected as the best model going forward.

**Which features matter most?**
Pulling `feature_importances_` from the Gradient Boosting model and sorting the top 10 gives a more precise ranking than the EDA charts alone suggested:

1. **`tenure`** (≈0.32) — by far the strongest signal in the model. How long someone has been a customer matters more than any other single factor.
2. **`InternetService_Fiber optic`** (≈0.19) — the second strongest driver, despite being the premium, higher-priced product.
3. **`PaymentMethod_Electronic check`** (≈0.10) — the third strongest driver, and notably stronger than contract type itself.
4. **`Contract_Two year`**, **`TotalCharges`**, **`MonthlyCharges`**, **`Contract_One year`** — a secondary tier, each contributing roughly 0.06–0.07.
5. **`OnlineSecurity_Yes`**, **`PaperlessBilling_Yes`**, **`OnlineSecurity_No internet service`** — smaller but still meaningful contributors, rounding out the top 10.

So while the EDA in Phase 2 made contract type look like the most dramatic visual split, the model itself weighs **tenure, internet type, and payment method** more heavily once all features are considered together — contract length still matters, but it's a secondary factor rather than the headline driver.

**Why are customers leaving?**
The highest-risk profile that emerges from combining the model's feature importances with the Phase 2 EDA is: a **new customer** (low tenure), subscribed to **Fiber optic internet**, paying via **electronic check**, on a **month-to-month contract**, without **Online Security**. This makes sense from a business standpoint:
- **Early-tenure risk dominates** — tenure is the single biggest signal, since new customers haven't yet built up a relationship, habits, or switching friction with the company.
- **Fiber optic dissatisfaction** — a premium, higher-priced product that still shows the second-highest importance for churn, pointing to a possible gap between price/expectations and the actual experience (reliability, support, or perceived value).
- **Electronic check as a behavioral marker** — this ranks above contract type, and may be a proxy for a broader pattern of customers who are less embedded/committed to the service overall (e.g. haven't set up automatic billing).
- **Low commitment, low switching cost** — month-to-month customers still have no penalty for leaving, which keeps contract type relevant even if it's not the top driver.
- **Weaker safety net** — customers without Online Security have one less reason to stay engaged with the broader service ecosystem.

**What should the business do?**
In order of expected impact, based on the model's actual feature importances:
1. Build a dedicated onboarding/retention push for the first few months of a customer's lifecycle — tenure is the single biggest lever available.
2. Investigate the Fiber optic experience specifically — price, reliability, or support could all be contributing, and it's a large, high-value segment to lose.
3. Encourage a shift away from electronic check toward automatic payment methods, since this is a stronger churn signal than contract type itself.
4. Incentivize customers onto longer (one- or two-year) contracts — still an effective, if secondary, lever.
5. Bundle Online Security into new/month-to-month plans to increase "stickiness."
6. Operationalize the model: score the live customer base every month, rank customers by predicted churn probability, and route the highest-risk, highest-revenue customers to the retention team first.

**Bottom line:** the project shows that churn at this telecom is driven primarily by **how new the customer is and what service/payment setup they have** — tenure, Fiber optic, and electronic check outweigh contract type itself in the model's eyes. That's still good news for the business, since onboarding programs, service-quality reviews, and billing-method nudges are all levers that can be acted on directly.
