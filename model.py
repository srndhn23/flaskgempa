import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from pandas import datetime
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from sklearn import preprocessing
from scipy.stats import boxcoxA
from scipy.special import inv_boxcox
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


gempa = pd.read_csv("gempa_indonesia.csv")
gempa.rename(columns = {'date':'time'}, inplace = True)
gempa['time']= pd.to_datetime(gempa['time'])
df = gempa[['time', 'mag', 'depth', 'province']]
df.columns = ['ds', 'y', 'depth', 'provinces']
df = pd.get_dummies(df, columns=['provinces'])
models = []

province_cols = [col for col in df.columns if col.startswith('provinces_')]
for province_col in province_cols:
    province = province_col.replace('provinces_', '')
    province_df = df[df[province_col] == 1]
    if len(province_df)<2:
        print(f"{province} has less than 2 rows, skipping.")
        continue
    # Fit the model for each province
    m = Prophet()
    m.fit(province_df)
    # Add the model to the list
    models.append(m)
    
# Create a future dataframe
future = models[0].make_future_dataframe(periods=30)

# Loop through the models and make predictions
province_forecasts = {}
for i, model in enumerate(models):
    forecast = model.predict(future)
    province_forecasts[province_cols[i].replace('provinces_', '')] = forecast
    
# Loop through the province forecasts and display the results
for province, forecast in province_forecasts.items():
    print(f"Forecast for {province}:")
    print(forecast[['ds', 'yhat']].tail())
    print()



# ---------------------------------------------------------------------------

'''import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from pandas import datetime
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
import sklearn
from sklearn import preprocessing
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from pydantic import BaseModel
import joblib

class MyModel:
    def __init__(self):
        self.models = {}
        self.df = None
        self.province_cols = None
        
    def load_data(self):
        gempa = pd.read_csv("Data_Gempa_Bumi.csv")
        df = gempa[['date', 'mag', 'depth', 'province']]
        df.columns = ['ds', 'y', 'depth', 'provinces']
        df = pd.get_dummies(df, columns=['provinces'])
        self.df = df
        self.province_cols = [col for col in df.columns if col.startswith('provinces_')]
        
    def fit_models(self):
        for province_col in self.province_cols:
            province = province_col.replace('provinces_', '')
            province_df = self.df[self.df[province_col] == 1]
            if len(province_df)<2:
                print(f"{province} has less than 2 rows, skipping.")
                continue
            m = Prophet()
            m.fit(province_df)
            self.models[province] = m
            
    def predict(self, province_name, date):
        if not self.models:
            # Load data and fit models if not already loaded
            self.load_data()
            self.fit_models()
        
        if province_name.lower() in self.models:
            m = self.models[province_name.lower()]
            future = m.make_future_dataframe(periods=30)
            forecast = joblib.Memory(cachedir='cachedir', verbose=0).cache(m.predict)(future)
            forecast = forecast[forecast['ds'] == date]
            if forecast.empty:
                return None
            else:
                forecast['yhat'] = forecast['yhat'].apply(lambda x: round(x,1))
                return forecast['yhat'].values[0]
        else:
            return None'''

'''
gempa = pd.read_csv("Data_Gempa_Bumi.csv")

df = gempa[['time', 'mag', 'depth', 'province']]

df.columns = ['ds', 'y', 'depth', 'provinces']

df = pd.get_dummies(df, columns=['provinces'])

# Create an empty list to store the models
models = []

# Get the names of the encoded provinces columns
province_cols = [col for col in df.columns if col.startswith('provinces_')]

# Loop through the encoded provinces columns
for province_col in province_cols:
    # Get the province name
    province = province_col.replace('provinces_', '')
    # Create a dataframe for each province
    province_df = df[df[province_col] == 1]
    if len(province_df)<2:
        print(f"{province} has less than 2 rows, skipping.")
        continue
    # Fit the model for each province
    m = Prophet()
    m.fit(province_df)
    # Add the model to the list
    models.append(m)
    
# Create a future dataframe
future = models[0].make_future_dataframe(periods=60)

# Loop through the models and make predictions
province_forecasts = {}
for i, model in enumerate(models):
    forecast = model.predict(future)
    province_forecasts[province_cols[i].replace('provinces_', '')] = forecast
    
# prompt the user to enter the name of the province
# province_name = input("Enter the name of the province: ")
# date = input("Enter the date in the format YYYY-MM-DD : ")

# check if the province name is in the dictionary
if province_name in province_forecasts:
    # get the forecast dataframe for the province
    forecast = province_forecasts[province_name]
    forecast = forecast[forecast['ds'] == date]
    if forecast.empty:
        print("No forecast data available for that date.")
    else:
        # do something with the forecast dataframe
        forecast['yhat'] = forecast['yhat'].apply(lambda x: round(x,1))
        print(forecast[['ds', 'yhat']].tail())
else:
    print(f"Province {province_name} not found.")'''