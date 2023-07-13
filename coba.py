import statsmodels.api as sm
from pandas import datetime
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


gempa = pd.read_csv("gempa_indonesia.csv")
gempa.rename(columns={'date': 'time'}, inplace=True)
gempa['time'] = pd.to_datetime(gempa['time'])
df = gempa[['time', 'mag', 'depth', 'province']]
df.columns = ['ds', 'y', 'depth', 'provinces']
df = pd.get_dummies(df, columns=['provinces'])
models = {}
province_cols = [col for col in df.columns if col.startswith('provinces_')]

for province_col in province_cols:
    province = province_col.replace('provinces_', '')
    province_df = df[df[province_col] == 1]
    if len(province_df) < 2:
        print(f"{province} has less than 2 rows, skipping.")
        continue
    m = Prophet()
    m.fit(province_df)
    models[province] = m

future = pd.DataFrame({'ds': pd.date_range(start='2023-06-07', periods=30)})

province_forecasts = {}

for province, model in models.items():
    forecast = model.predict(future)
    province_forecasts[province] = forecast

for province, forecast in province_forecasts.items():
    print(f"Forecast for {province}:")
    print(forecast[['ds', 'yhat']].tail())
    print()

province_name = input("Enter the name of the province: ")
date = input("Enter the date in the format YYYY-MM-DD : ")

if province_name in province_forecasts:
    forecast = province_forecasts[province_name]
    forecast = forecast[forecast['ds'] == date]
    if forecast.empty:
        print("No forecast data available for that date.")
    else:
        forecast['yhat'] = forecast['yhat'].apply(lambda x: round(x, 1))
        print(forecast[['ds', 'yhat']].tail())
else:
    print(f"Province {province_name} not found.")
