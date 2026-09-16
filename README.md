# Customer Churn Prediction

Predicting which telecom customers are likely to cancel their subscription, using a full pipeline from raw multi-table SQL data to a deployed interactive app.

## Overview

This project simulates a realistic company task: a telecom provider's customer, usage, and support data lives across several related tables, and the goal is to identify customers at risk of churning so the retention team can act proactively.

The project covers the complete workflow used in real data science work — not just model training:

1. **Data extraction (SQL)** — joining data spread across 4 related tables in a SQLite database
2. **EDA & feature engineering** — cleaning, visual analysis, and encoding categorical features
3. **Model training** — training and comparing 3 classification models
4. **Model interpretation** — explaining predictions and addressing class imbalance
5. **Deployment** — an interactive Streamlit app for live predictions

## Project structure

```
customer-churn-prediction/
├── data/
│   └── telecom_churn.db          # Raw SQLite database (4 related tables)
├── notebooks/
│   ├── 01_sql_extraction.ipynb   # SQL: SELECT, WHERE, GROUP BY, JOIN
│   ├── 02_eda_feature_engineering.ipynb
│   ├── 03_model_training.ipynb   # Logistic Regression, Random Forest, Gradient Boosting
│   └── 04_model_interpretation.ipynb
├── app/
│   ├── app.py                    # Streamlit prediction app
│   ├── churn_model_balanced.pkl  # Trained model (class_weight="balanced")
│   └── X_train.csv               # Feature reference for encoding new input
├── docs/
│   └── Customer_Churn_Full_Documentation.docx  # Full write-up with all code and results
├── requirements.txt
└── README.md
```

## Tech stack

- **Data**: SQLite, SQL (JOIN, GROUP BY, aggregations)
- **Analysis**: Python, Pandas, NumPy, Matplotlib, Seaborn
- **Modeling**: Scikit-learn (Logistic Regression, Random Forest, Gradient Boosting)
- **App**: Streamlit

## Results

### Model comparison (default settings)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** | 0.917 | 0.571 | 0.364 | **0.444** |
| Gradient Boosting | 0.908 | 0.500 | 0.364 | 0.421 |
| Random Forest | 0.908 | 0.500 | 0.273 | 0.353 |

Logistic Regression was selected as the best model based on F1-score, since the dataset is imbalanced (~9.5% churn rate) and accuracy alone is misleading.

### Addressing low recall with `class_weight="balanced"`

The default model only caught 36% of customers who actually churned. Retraining with balanced class weights trades some precision for substantially better recall:

| Metric (churn class) | Default | `class_weight="balanced"` |
|---|---|---|
| Precision | 0.57 | 0.23 |
| Recall | 0.36 | **0.64** |
| F1 | 0.44 | 0.33 |

This is a business trade-off, not a purely technical one — which model is "better" depends on the relative cost of a missed at-risk customer versus a false alarm.

### Key drivers of churn

Feature importance analysis (see `notebooks/04_model_interpretation.ipynb`) showed that **number of support tickets** and being on the **Basic plan** were the strongest predictors of churn — both actionable insights for a retention team.

## Running the project

### 1. Set up the environment

```bash
pip install -r requirements.txt
```

### 2. Run the notebooks in order

Open `notebooks/01_sql_extraction.ipynb` through `04_model_interpretation.ipynb` in Jupyter or Google Colab, in numeric order.

> **Note:** each notebook expects its input file(s) in the *same folder* it runs from (this matches how the project was originally built and tested in Google Colab). Before running `01_sql_extraction.ipynb`, copy `data/telecom_churn.db` into the `notebooks/` folder (or upload it alongside the notebook if using Colab). Each notebook then saves the file(s) the next one needs (e.g. `churn_data_extracted.csv`, `X_train.csv`, `churn_model.pkl`) into that same folder.

> The `app/` folder already contains a pre-trained model and reference file, so you can skip straight to step 3 if you just want to try the app.

### 3. Run the app

```bash
cd app
streamlit run app.py
```

Enter a customer's plan type, usage, and support history to get a live churn probability.

## License

MIT
