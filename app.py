from flask import Flask, request, jsonify, render_template
from autogluon.tabular import TabularPredictor
import pandas as pd
import joblib
import os
from pathlib import Path

# Bezpieczne ścieżki względem aktualnego pliku
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "webapp" / "templates"
STATIC_DIR = BASE_DIR / "webapp" / "static"
MODEL_DIR = BASE_DIR / "agModels_sm"

app = Flask(__name__, template_folder=str(
    TEMPLATE_DIR), static_folder=str(STATIC_DIR))

# Wczytanie modelu
predictor = TabularPredictor.load(str(MODEL_DIR))


def get_time_of_day(hour):
    if 0 <= hour < 6:
        return 'night'
    elif 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    else:
        return 'evening'


def clean_standardize_data(data):
    df = pd.DataFrame([{
        "class": data["class"],
        "date": data["date"],
        "airline": data["airline"],
        "dep_time": data["dep_time"],
        "departure_city": data["departure_city"],
        "stops": int(data["stop"]),
        "arr_time": data["arr_time"],
        "arrival_city": data["arrival_city"],
    }])
    df['dep_time'] = pd.to_datetime(
        df['dep_time'], format='%H:%M', errors='coerce')

    df['arr_time'] = pd.to_datetime(
        df['arr_time'], format='%H:%M', errors='coerce')

    df['dep_hour'] = df['dep_time'].dt.hour
    df['dep_mins'] = df['dep_time'].dt.minute

    df['arr_hour'] = df['arr_time'].dt.hour
    df['arr_mins'] = df['arr_time'].dt.minute

    # Jeśli arr_time < dep_time, dodaj 1 dzień (przylot następnego dnia)
    df.loc[df['arr_time'] < df['dep_time'], 'arr_time'] += pd.Timedelta(days=1)

    # Obliczenie czasu trwania lotu
    df["flight_duration"] = (
        df["arr_time"] - df["dep_time"]).dt.total_seconds() / 60  # w minutach

    # Opcjonalnie zaokrąglenie do int
    df["flight_duration"] = df["flight_duration"].astype(int)

    df = df.drop(["dep_time", "arr_time"], axis=1)

    df["date"] = pd.to_datetime(df['date'], format='%Y-%m-%d')

    # Stworzenie nowych kolumn dla miesiąca i dnia wylotu
    df["month"] = df['date'].dt.month
    df["day"] = df['date'].dt.day
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)

    df['time_of_day'] = df['dep_hour'].apply(get_time_of_day)

    start_date = pd.Timestamp.today().normalize()

    df['days_left'] = (df['date']-start_date).dt.days

    df = df.drop(["date"], axis=1)

    numeric_cols = ["flight_duration", "stops", "dep_hour",
                    "dep_mins", "arr_hour", "arr_mins", "month", "day", "days_left"]

    scaler = joblib.load(str(BASE_DIR / "scaler.pkl"))
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form
        df = clean_standardize_data(data)

        prediction = predictor.predict(df)
        pred = float(prediction.values[0])

        return jsonify({"prediction": pred})

    except Exception as e:
        import traceback
        import sys

        # loguj do konsoli (Render logs)
        print("Błąd w predykcji:", e, file=sys.stderr)
        traceback.print_exc()

        # zwróć szczegóły błędu w formacie JSON do frontendu
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
