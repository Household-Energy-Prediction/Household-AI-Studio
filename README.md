# Household Energy Prediction (AI Studio – Team 15)

This project predicts **Global Active Power** (household electricity usage) using the UCI “Individual household electric power consumption” dataset.  
We compare classical ML models and time-series approaches across **minute-level** and **hourly-resampled** data, and test **time-based feature engineering** (cyclical hour/day/month encodings).

---

## Project Goals
- Build a reproducible ML pipeline (preprocessing → feature engineering → train → evaluate)
- Compare models:
  - Linear Regression (LR)
  - Random Forest (RF)
  - Time-series deep models (Darts / N-BEATS experiments)
- Evaluate performance with:
  - **R² (primary)**
  - **RMSE / MAE (secondary)**
- Examine interpretability using coefficients (LR) and feature importance (RF)

---

## Repository Structure

### Data
- `household_power_consumption.csv.zip`  
  Zipped raw dataset. Unzip into the project root to run the pipeline.

### Preprocessing / Feature Engineering
- `ConvertDatetime.py` - creates a cleaned dataset with a datetime index
- `RemoveOutliers.py` - optional outlier handling
- `Normalize.py` - optional normalization utilities
- `AdditionalVar.py` - additional variable/feature helpers
- `Resample.py` - resampling utilities
- `ResampleHourly.py` - generates **hourly averaged** dataset
- `AddTimeFeatures.py` - adds:
  - raw time parts: hour, dayofweek, month, year, is_weekend
  - cyclical encodings: sin/cos for hour/day/month

### Modeling
**Minute-level**
- `linearRegression.py` - LR baseline (minute-level)
- `TrainRandomForest.py` - RF baseline (minute-level)

**Hourly**
- `linearRegression_Hourly.py` - LR on hourly-resampled
- `TrainRandomForest_Hourly.py` - RF on hourly-resampled

**Hourly + Time Features**
- `linearRegression_Hourly_TimeFeatures.py` - LR + time features
- `TrainRandomForest_Hourly_TimeFeatures.py` - RF + time features (best overall)

### Time-Series (Experimental)
- `Darts.py` - Darts experiments (e.g., N-BEATS)
- `main.ipynb` - notebook version of parts of the pipeline

### Other
- `LICENSE` - MIT
- `.gitignore`

---

## Requirements

Install core dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
