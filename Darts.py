import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from darts import TimeSeries
from darts.metrics import mape
from darts.metrics import mae
from darts.metrics import rmse
from darts.models import NBEATSModel
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

# Define NBEATSModel
model = NBEATSModel(
    input_chunk_length=6,
    output_chunk_length=6,
    n_epochs=100,
    activation='LeakyReLU',
    optimizer_kwargs={'lr': 1e-6},
    pl_trainer_kwargs={
        "accelerator": "gpu",
        "devices": [0],
        "callbacks": [EarlyStopping(monitor='val_loss', patience=5)],
    },
)

# Fit using past_covariates and plot result
model.fit(series=train, past_covariates=covariates, val_series=val, val_past_covariates=val_cov)
forecast = model.predict(n=len(val), past_covariates=covariates)
val[-len(val):].plot(label="Actual")
forecast.plot(label="Forecast")
plt.legend()
plt.show()

print(mape(val[-len(val):], forecast[-len(val):]))
print(rmse(val[-len(val):], forecast[-len(val):]))
print(mae(val[-len(val):], forecast[-len(val):]))
# res = model.residuals(
#     series=val,
#     past_covariates=val_cov if 'val_cov' in locals() else None,
#     forecast_horizon=1,
#     retrain=False,
#     last_points_only=True,  # gives you a Darts TimeSeries for the forecast time points
#     verbose=False,
# )
#
# print("\n=== Residual diagnostics ===")
# print("res type:", type(res))
# print("res values shape:", res.values().shape)
# print("res index:", res.time_index[:5], "...", res.time_index[-5:])
# print("First 10 residuals:", res.values()[:10].flatten())
#
# # Optionally, measure score
# mape_val = mape(val, res + val)  # since res = target - forecast  => forecast = target - res
# print(f"MAPE on val set (from residuals): {mape_val:.2f}%")
