# Delay Intelligence: Shipment Delivery Analysis and Prediction

## About this project

This project looks at a practical logistics question: **which shipments are likely to arrive late, how long might delivery take, and what can operations teams do about it?**

The workflow starts with a deliberately messy shipment export. It tidies the data, adds useful information about each delivery hub, explores the patterns in the shipments, and trains two machine-learning models. One model estimates delivery time in hours; the other flags shipments that may miss their promised delivery window. A final analysis groups hubs with similar operating patterns and turns the findings into straightforward recommendations.

The supplied data generator creates synthetic shipment records for 24 hubs across India. These records are for coursework and experimentation; they are not real customer or carrier records. The generator is kept unchanged from the supplied project file so that its seeded data can be reproduced.

## Repository contents

```text
.
├── README.md
├── requirements.txt
├── generate_data.py
├── data/
│   ├── hubs.csv
│   ├── raw_shipments.csv
│   ├── new_shipments.csv
│   └── modelling_table.csv
├── src/
│   ├── __init__.py
│   └── shipment_prep.py
├── part_a_clean_eda.py
├── part_b_regression.py
├── part_c_classification.py
├── part_d_segments.py
├── package_model.py
├── predict_delay.py
├── models/
│   ├── delay_model.pkl
│   ├── classification_model.pkl
│   ├── delay_threshold.csv
│   ├── feature_order.csv
│   ├── impute_values.csv
│   └── model_manifest.json
└── outputs/
    ├── eda_summary.csv
    ├── missing_values.csv
    ├── service_summary.csv
    ├── region_summary.csv
    ├── regression_metrics.csv
    ├── regression_predictions.csv
    ├── classification_metrics.csv
    ├── classification_predictions.csv
    ├── confusion_matrix.csv
    ├── hub_segments.csv
    ├── operations_recommendations.csv
    └── new_shipment_predictions.csv
```

## How to run

Use Python 3.11 or newer. The commands below are run from the repository's top-level folder.

1. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   # macOS / Linux
   source .venv/bin/activate
   # Windows PowerShell: .venv\Scripts\Activate.ps1
   ```

2. Install the pinned packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Generate the source CSV files. This recreates `hubs.csv`, `raw_shipments.csv` and `new_shipments.csv` using the fixed seeds in the provided generator:

   ```bash
   python generate_data.py
   ```

4. Run the analysis and modelling scripts in order:

   ```bash
   python part_a_clean_eda.py
   python part_b_regression.py
   python part_c_classification.py
   python part_d_segments.py
   python package_model.py
   python predict_delay.py
   ```

   Part A creates `data/modelling_table.csv` and the initial EDA outputs. Parts B and C train and evaluate the regression and classification models. Part D creates hub segments and operational recommendations. `package_model.py` writes the feature order, imputation values and model manifest. The final script scores the unseen rows in `data/new_shipments.csv`.

All paths are resolved relative to the project folder, so the scripts do not depend on a particular computer, notebook, or `/content` directory.

## Data findings

The raw export is intentionally imperfect. It includes duplicate rows, missing values, inconsistent capitalisation and spacing in service names, currency-formatted declared values, and a sentinel parcel-weight value (`999`) that should not be treated as a genuine measurement. The cleaning process removes exact duplicate rows, standardises text, parses dates and currency values, and treats weights above 100 kg as missing before imputation.

The generated export contains **1,224 rows before cleaning**. The cleaning step removes **24 exact duplicates**, leaving **1,200 shipments** for modelling, and identifies **5 sentinel weights** for correction. In this run, the overall delay rate is **23.17%**, and average delivery time is **56.96 hours**. The service-level summaries show delay rates of approximately **49.0% for Express**, **20.6% for Standard** and **8.6% for Economy**. By region, the generated data has the highest delay rate in the **East (43.25%)**; North, South and West are all around 12–14%. These are observations about this synthetic dataset, not claims about real-world carrier performance. The exact counts and summaries are exported to `outputs/eda_summary.csv`, `outputs/missing_values.csv`, `outputs/service_summary.csv` and `outputs/region_summary.csv`.

Feature engineering joins hub tier and daily capacity to each shipment. It also derives promised delivery hours from service type, the log of declared value, COD status, weekend status and dispatch hour. The fixed feature list is shared by the training and prediction scripts to avoid training/inference mismatches.

## Model results

### Regression

A Random Forest regression model estimates `delivery_hours`. Performance is evaluated on a held-out test set using:

- **MAE:** average absolute prediction error, measured in hours.
- **RMSE:** root mean squared error, measured in hours; larger errors receive more weight.
- **R²:** the proportion of target variation explained by the model.

On the fixed 20% hold-out test set (240 shipments), the run produced **MAE = 3.66 hours**, **RMSE = 4.78 hours** and **R² = 0.9385**. These are baseline results on synthetic data. The measured results are saved in `outputs/regression_metrics.csv`. Individual actual and predicted delivery times are saved in `outputs/regression_predictions.csv`.

### Classification

A Random Forest classifier estimates whether a shipment will be delayed. A shipment is labelled delayed when its actual delivery time is greater than the promised hours associated with its service type. The initial decision threshold is 0.50.

On the same 240-row test set, the classifier achieved **86.67% accuracy**, **0.75 precision**, **0.643 recall**, **0.692 F1** and **0.918 ROC-AUC** at the 0.50 threshold. These metrics answer different questions: for example, recall measures how many truly delayed shipments the model catches, while precision measures how often a delay alert is correct. The confusion matrix is also exported. Results are available in `outputs/classification_metrics.csv` and `outputs/confusion_matrix.csv`; row-level test predictions are in `outputs/classification_predictions.csv`.

The classification threshold is stored in `models/delay_threshold.csv`, making the decision rule visible and adjustable rather than hiding it in the prediction code.

## Operations recommendation

The hub analysis compares delay rate, average distance, average number of stops and average delivery time. K-means groups hubs with broadly similar patterns; the segment IDs are data-driven labels, not a ranking. The output also assigns a plain-language profile based on each segment's average delay rate.

The recommendation file highlights hubs with comparatively high delay rates for a closer review of route capacity, dispatch cut-offs and weather contingency planning. Long-distance hubs are flagged for lane and hand-off review. Hubs without either signal are recommended for routine monitoring. These are decision-support suggestions, not automatic operational instructions.

See `outputs/hub_segments.csv` and `outputs/operations_recommendations.csv`.

## Production plan

Before using a similar system in a live delivery operation, the following steps would be sensible:

1. Replace synthetic records with governed, quality-checked shipment and hub data.
2. Validate data definitions with operations teams, especially what counts as a delay and how promised hours are calculated.
3. Train and validate on time-based splits so that evaluation better reflects future shipments.
4. Track model quality by hub, service type, region and other operationally relevant groups.
5. Monitor data drift, delay rates, false alerts and missed delays; retrain only through a reviewed release process.
6. Expose predictions through an authenticated API or operations dashboard, with timestamps and model-version logging.
7. Keep a human dispatcher in control of consequential actions and provide a way to correct inaccurate source data or predictions.

## Limitations

- The dataset is synthetic and generated from programmed assumptions. Strong results on it do not prove that the models will perform equally well on real shipments.
- The data covers a limited simulated period and a fixed set of 24 hubs. Seasonality and future network changes may not be represented.
- Some generated columns are random identifiers or operational noise. They are retained in the supplied feature design for this exercise, but should be tested for usefulness and removed if they do not generalise.
- The random train/test split is suitable for a reproducible classroom baseline, but a chronological split is preferable for a production forecasting use case.
- The 0.50 classification threshold is a starting point. A real organisation should choose a threshold based on the relative cost of missed delays and unnecessary alerts.
- The segment labels and recommendations are descriptive. They do not establish the cause of a delay or prove that a proposed intervention will improve performance.
- Predictions should support, not replace, operational judgement. The model should not be used as the sole basis for customer compensation, employee evaluation or other high-impact decisions.

## Reproducibility

The generator uses fixed random seeds and the model scripts use fixed split/model seeds. Re-running the workflow with the pinned dependencies should reproduce the datasets and results closely. The fitted model files are included for convenience; running the scripts again will replace them with freshly trained versions.
