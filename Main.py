import sklearn;
import sklearn.pipeline;
import sklearn.preprocessing;
import sklearn.linear_model;
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns

from ConvertDatetime import convertDatetime;
from Resample import resample;
from Normalize import normalize;
from RemoveOutliers import removeOutliers;

household_power_df = os.path.join(os.getcwd(), "household_power_consumption.csv")
df = pd.read_csv(household_power_df, header=0, low_memory=False);

df = (
        df.pipe(lambda x: x.dropna())   # drop rows with missing values
          .pipe(convertDatetime)
          .pipe(normalize)
          .pipe(removeOutliers)
          .pipe(resample)[0]
)

print(df.head());
print(df.shape);