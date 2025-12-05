import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from darts import TimeSeries
from darts.metrics import mape, r2_score
from darts.models.forecasting.nbeats import NBEATSModel
from darts.metrics import mae
from darts.metrics import rmse
from darts.utils.model_selection import train_test_split
from pytorch_lightning.callbacks import EarlyStopping

torch.set_float32_matmul_precision('high')

# Change hourly to any resampled
df = pd.read_csv(os.path.join(os.getcwd(), "daily.csv"), header=0, low_memory=False)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index("datetime", inplace=True)

# Fill in missing values by just copying last observations to missing
# to get regular time series
full_index = pd.date_range(df.index.min(), df.index.max(), freq='d')
df = df.reindex(full_index)
df = df.ffill().bfill()

# Get features and labels
X = df[[x for x in df.columns if x != 'Global_active_power']]
y = df['Global_active_power']
X = X.astype(np.float32)
y = y.astype(np.float32)

# Get timeseries
covariates = TimeSeries.from_dataframe(X)
target_series = TimeSeries.from_series(y)

# Train Val Split
train, val = train_test_split(target_series, test_size=0.2)
train_cov, val_cov = train_test_split(covariates, test_size=0.2)

# Use grid search to find the best_params
# param_grid = {
#     "input_chunk_length": [6, 12, 18],
#     "output_chunk_length": [1, 3, 6],
#     "n_epochs": [100],
#     "activation": ["ReLU", "LeakyReLU"],
#     "optimizer_kwargs": [{"lr": 1e-4}, {"lr": 1e-3}],
# }
#
# best_model, best_params, best_score = NBEATSModel.gridsearch(
#     parameters=param_grid,
#     series=train,
#     val_series=val,
#     past_covariates=covariates,
#     metric=rmse,
# )
# Best Params
# {'input_chunk_length': 6, 'output_chunk_length': 1, 'n_epochs': 100, 'activation': 'LeakyReLU', 'optimizer_kwargs': {'lr': 0.0001}}

train, rest = train_test_split(target_series, test_size=0.3)
val, test = train_test_split(rest, test_size=0.03)
train_cov, rest_cov = train_test_split(covariates, test_size=0.3)
val_cov, test_cov = train_test_split(rest_cov, test_size=0.03)

best_params = {'input_chunk_length': 6, 'output_chunk_length': 1, 'n_epochs': 100,
               'activation': 'LeakyReLU', 'optimizer_kwargs': {'lr': 0.0001}}

# Refit best model on train+val
final_train = train.append(val)
final_train_cov = train_cov.append(val_cov)
best_model = NBEATSModel(**best_params)
best_model.fit(series=final_train, past_covariates=covariates)  # Can use covariates = full

# Forecast and evaluate on test set only!
n_forecast = len(test)
history_for_prediction = final_train
test_forecast = best_model.predict(n=n_forecast, series=history_for_prediction, past_covariates=covariates)

# Plot
test.plot(label="Actual - Test")
test_forecast.plot(label="Forecast - Test")
plt.legend()
plt.show()

print("On TRUE holdout test set:")
print("MAPE: ", mape(test, test_forecast))
print("RMSE: ", rmse(test, test_forecast))
print("MAE: ", mae(test, test_forecast))
print("R2: ", r2_score(test, test_forecast))

