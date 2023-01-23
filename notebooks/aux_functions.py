import pandas as pd
import numpy as np
from sktime.performance_metrics.forecasting import MeanSquaredError
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

# Unique values

def unique_values(df, column):

    unique_values = df[column].unique().tolist()
    return unique_values

# Outliers

def detect_outliers(df, column):
    """
    Función que detecta los outliers en una columna de un dataframe utilizando la técnica Z-Score
    """
    # Z-Score
    df["Z-Score"] = (df[column] - df[column].mean()) / df[column].std()

    # filter outliers
    outliers = df[(np.abs(df["Z-Score"]) > 3)]

    return outliers

# Performance

RMSE = MeanSquaredError(square_root=True)
MAPE = MeanAbsolutePercentageError(symmetric=False)

def ForecastPerformance(original,forecast):
    print(f'RMSE: {round(RMSE(original, forecast),2)}')
    print(f'MAPE: {round(MAPE(original, forecast)*100,2)}%')
