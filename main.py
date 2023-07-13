from flask import Flask, request, render_template, jsonify, Response
import numpy as np
import pandas as pd
from prophet import Prophet
import mysql.connector
from datetime import datetime, timedelta
import math
import statsmodels.api as sm
# from pandas import datetime
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

app = Flask(__name__, static_folder='static')


# Mysql Connection
cnx = mysql.connector.connect(
    database = 'railway',
    host = 'containers-us-west-157.railway.app',
    password = 'NCodJAUN2nBxE7ItrgQm',
    port = '7499',
    user = 'root'
)

@app.route("/", methods=['GET'])
def index():
    return render_template("dashboard.html")

@app.route("/about", methods=['GET'])
def about():
    return render_template("about.html")

@app.route("/penanggulangan", methods=['GET'])
def penanggulangan():
    return render_template("penanggulangan.html")

def get_province_name(province):
    province_names = {
        "ID-MU": "Maluku Utara",
        "ID-PA": "Papua",
        "ID-NT": "Nusa Tenggara Timur",
        "ID-LA": "Lampung",
        "ID-AC": "Aceh",
        "ID-MA": "Maluku",
        "ID-SU": "Sumatra Utara",
        "ID-JT": "Jawa Tengah",
        "ID-GO": "Gorontalo",
        "ID-JB": "Jawa Barat",
        "ID-BE": "Bengkulu",
        "ID-SA": "Sulawesi Utara",
        "ID-SB": "Sumatra Barat",
        "ID-NB": "Nusa Tenggara Barat",
        "ID-RI": "Riau",
        "ID-KS": "Kalimantan Selatan",
        "ID-ST": "Sulawesi Tengah",
        "ID-BT": "Banten",
        "ID-SS": "Sumatra Selatan",
        "ID-JI": "Jawa Timur",
        "ID-SN": "Sulawesi Selatan",
        "ID-BA": "Bali",
        "ID-PB": "Papua Barat",
        "ID-YO": "Daerah Istimewa Yogyakarta",
        "ID-JA": "Jambi",
        "ID-JK": "DKI Jakarta",
        "ID-KI": "Kalimantan Timur",
        "ID-SG": "Sulawesi Tenggara",
        "ID-KU": "Kalimantan Utara",
        "ID-SR": "Sulawesi Barat",
        "ID-KB": "Kalimantan Barat"
    }
    
    return province_names.get(province, "Unknown")

@app.route("/histori/<province>/log_magnitudes", methods=['GET'])
def histori(province):
    cur = cnx.cursor()
    cur.execute("SELECT date, latitude, longitude, depth, mag, place FROM histori WHERE province = %s ORDER BY date DESC", (province,))
    results = cur.fetchall()
    cur.close()

    data = [[row[0], row[1], row[2], row[3], row[4], row[5]] for row in results]

    log_magnitudes = [math.log10(row[4]) for row in results]

    province_name = get_province_name(province)
    context = {'data': data, 'province': province, 'log_magnitudes': log_magnitudes, 'province_name': province_name}
    return render_template("histori.html", **context)

@app.route("/predict", methods=['POST'])
def predict():
    province = request.form['province']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    # Load earthquake data
    gempa = pd.read_csv("gempa_indonesia.csv")
    gempa.rename(columns={'date': 'time'}, inplace=True)
    gempa['time'] = pd.to_datetime(gempa['time'])
    df = gempa[['time', 'mag', 'depth', 'province']]
    df.columns = ['ds', 'y', 'depth', 'provinces']
    df = pd.get_dummies(df, columns=['provinces'])

    # Train Prophet models for each province
    models = {}
    province_cols = [col for col in df.columns if col.startswith('provinces_')]
    for province_col in province_cols:
        province_name = province_col.replace('provinces_', '')
        province_df = df[df[province_col] == 1]
        if len(province_df) < 2:
            print(f"{province_name} has less than 2 rows, skipping.")
            continue
        m = Prophet()
        m.fit(province_df)
        models[province_name] = m

    # Generate forecast for the specified province and date range
    forecast_data = None
    if province in models:
        model = models[province]
        future = pd.date_range(start=start_date, end=end_date)
        forecast = model.predict(pd.DataFrame({'ds': future}))[['ds', 'yhat']]
        forecast = forecast.round({'yhat': 1})
        forecast_data = forecast.rename(columns={'yhat': 'Forecast'})

    if forecast_data is not None and not forecast_data.empty:
        return render_template("dashboard.html", forecast_data=forecast_data, province=province)
    else:
        return render_template("dashboard.html", province=province)

@app.route("/coba", methods=['GET'])
def coba():
    try:
        cnx = mysql.connector.connect(user='root', password='', host='localhost', database='gempa')
        cursor = cnx.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        print("MySQL connection is working.")
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
    finally:
        cursor.close()
        cnx.close()

    return render_template("coba.html")

if __name__ == "__main__":
    app.run()
