from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

# Wczytanie danych
cleaned = pd.read_csv("./data/cleaned.csv")

# Lista kolumn do standaryzacji
numeric_cols = ["flight_duration", "stops", "dep_hour",
                "dep_mins", "arr_hour", "arr_mins", "month", "day", "days_left"]

# Standaryzacja wybranych kolumn
scaler = StandardScaler()
cleaned_scaled_numeric = pd.DataFrame(
    scaler.fit_transform(cleaned[numeric_cols]),
    columns=numeric_cols,
    index=cleaned.index
)

# Zapis skalera do pliku (do późniejszego użycia przy predykcji)
joblib.dump(scaler, './scaler.pkl')

# Pozostałe kolumny (nie podlegające standaryzacji)
other_cols = cleaned.drop(columns=numeric_cols)

# Połączenie ustandaryzowanych kolumn z resztą danych
final_df = pd.concat([cleaned_scaled_numeric, other_cols], axis=1)

# Zapis do pliku
final_df.to_csv("./data/final.csv", index=False)
