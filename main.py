import os
from flask import Flask, request, render_template, Response
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
import matplotlib.pyplot as plt
from io import BytesIO
import base64
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
    cur.execute("SELECT DISTINCT date, waktu, latitude, longitude, depth, mag, place FROM histori WHERE province = %s ORDER BY date DESC", (province,))
    results = cur.fetchall()
    cur.close()

    data = [[row[0], row[1], row[2], row[3], row[4], row[5], row[6]] for row in results]

    magnitudes = [row[5] for row in results]

    num_data_points = len(magnitudes)
    num_bins = max(1, math.ceil(math.sqrt(num_data_points)))

    plt.figure(figsize=(8, 6))
    plt.hist(magnitudes, bins=num_bins, edgecolor='white', color='blue', alpha=0.6)
    plt.xlabel('Magnitudo')
    plt.ylabel('Frequency')
    plt.title('Histogram Magnitudo')
    plt.grid(True)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    plot_data = base64.b64encode(buffer.read()).decode()

    # Get filter parameters from the URL query string
    filter_start_date = request.args.get('filter_start_date')
    filter_end_date = request.args.get('filter_end_date')
    filter_magnitude = request.args.get('filter_magnitude')
    filter_location = request.args.get('filter_location')

    # Apply filters to the data
    filtered_data = data
    if filter_start_date and filter_end_date:
        start_date = datetime.strptime(filter_start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(filter_end_date, '%Y-%m-%d').date()
        filtered_data = [row for row in filtered_data if start_date <= row[0] <= end_date]
    if filter_magnitude:
        filtered_data = [row for row in filtered_data if float(row[5]) >= float(filter_magnitude)]
    if filter_location:
        filter_location_lower = filter_location.lower()
        filtered_data = [row for row in filtered_data if filter_location_lower in row[6].lower()]

    # Pagination
    page = request.args.get('page', default=1, type=int)
    items_per_page = 25
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    paginated_data = filtered_data[start_idx:end_idx]

    total_pages = math.ceil(len(filtered_data) / items_per_page)

    province_name = get_province_name(province)
    context = {'data': paginated_data, 'province': province, 'province_name': province_name, 'plot_data': plot_data, 'total_pages': total_pages, 'page': page}
    return render_template("histori.html", **context)

@app.route("/predict", methods=['POST'])
def predict():
    province = request.form['province']
    start_date = datetime.strptime(request.form['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(request.form['end_date'], '%Y-%m-%d') + timedelta(days=1, seconds=-1)

    today = datetime.today()

    # Load earthquake data
    gempa = pd.read_csv("gempa_indonesia.csv")
    gempa.rename(columns={'date': 'time'}, inplace=True)
    gempa['time'] = pd.to_datetime(gempa['time'])
    df = gempa[['time', 'mag', 'depth', 'province']]
    df.columns = ['ds', 'y', 'depth', 'provinces']
    df = pd.get_dummies(df, columns=['provinces'])

    if start_date.date() <= today.date():
        return render_template("dashboard.html", message="Maaf, Anda hanya dapat memprediksi magnitudo gempa di masa depan.")

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

    forecast_data = None

    if province in models:
        model = models[province]
        future = pd.date_range(start=start_date, end=end_date)
        forecast = model.predict(pd.DataFrame({'ds': future}))[['ds', 'yhat']]
        forecast = forecast.round({'yhat': 1})
        forecast_data = forecast.rename(columns={'yhat': 'Forecast'})

        # Simpan hasil prediksi ke database
        if not forecast_data.empty:
            for index, row in forecast_data.iterrows():
                tanggal_prediksi = row['ds']
                prediksi_magnitudo = row['Forecast']
                # Simpan hasil prediksi ke tabel hasil_prediksi
                query = "INSERT INTO hasil_prediksi (date, mag, province) VALUES (%s, %s, %s)"
                values = (tanggal_prediksi, prediksi_magnitudo, province)
                cur = cnx.cursor()
                cur.execute(query, values)
                cnx.commit()
    else:
        forecast_data = pd.DataFrame()

    if not forecast_data.empty:
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
