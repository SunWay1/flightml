# Importowanie biblioteki do pracy ze zbiorami
import pandas as pd

# Wczytywanie połączonego zbioru
combined = pd.read_csv("./data/combined.csv")

# Usuwanie kolumn opisujących numer lotu i identyfikator linii lotniczej
combined = combined.drop(["ch_code", "num_code"], axis=1)

# Wyczyszczenie danych w kolumnie "stop"
# Sprowadzenie wszystkich wartości do małych liter i usunięcie przerw/znaków nowej linii
combined["stop"] = combined["stop"].str.lower().str.strip()

# Wymiana słownych opisów na numeryczne odpowiedniki lub None jeśli żaden warunek nie jest spełniony
combined["stop"] = combined["stop"].apply(
    lambda x: 0 if "non-stop" in x else 1 if "1-stop" in x else 2 if "2+-stop" in x else None)

# Usunięcie rzędów w których kolumna "stop" przyjmuje wartość None
combined.dropna(subset=["stop"])

# Zamiana typu string na datetime w formacie HH:MM z opcją do wymuszenia zmiany w wypadku niekompletnych danych
combined['dep_time'] = pd.to_datetime(
    combined['dep_time'], format='%H:%M', errors='coerce')
combined['arr_time'] = pd.to_datetime(
    combined['arr_time'], format='%H:%M', errors='coerce')

# Utworzenie nowych kolumn
combined['dep_hour'] = combined['dep_time'].dt.hour
combined['dep_mins'] = combined['dep_time'].dt.minute

combined['arr_hour'] = combined['arr_time'].dt.hour
combined['arr_mins'] = combined['arr_time'].dt.minute

# Usunięcie niepotrzebnych kolumn opisujących kod lotu
combined = combined.drop(["dep_time", "arr_time"], axis=1)


# Funkcja konwersji kolumny "time_taken" na minuty
def parse_duration(duration):
    try:
        # Rozbicie stringa na godziny i minuty (np. '3h 10m' -> 3 i 10)
        parts = duration.lower().replace('h', '').replace('m', '').split()

        # Przypisanie liczby godzin i minut do zmiennych
        hours = int(parts[0])
        minutes = int(parts[1])

        # Zwrócenie sumy w minutach
        return hours * 60 + minutes

    # Łapanie błędu w konwersji
    except Exception:
        return None


# Zastosowanie funkcji konwersji do kolumny 'time_taken'
combined['time_taken'] = combined['time_taken'].apply(parse_duration)

# Usunięcie rzędów z wartościami None
combined = combined.dropna(subset=["time_taken"])

# Zmiana typu kolumny na int
combined["time_taken"] = combined["time_taken"].astype(int)

# Zmiana kolumny "date" na typ datetime w formacie dd-mm-yyyy
combined["date"] = pd.to_datetime(combined['date'], format='%d-%m-%Y')

# Stworzenie nowych kolumn dla miesiąca i dnia wylotu
combined["month"] = combined['date'].dt.month
combined["day"] = combined['date'].dt.day

# Zmiany w kolumnie 'price'
# Średni kurs wymiany rupii indyjskich na złotówki
exchange_rate = 0.04

try:
    # Usunięcie "," z ceny i zmiana typu danych na float
    combined['price'] = combined['price'].str.replace(',', '').astype(float)

    # Zmiana waluty na złotówki
    combined['price'] = combined['price'] * exchange_rate

    # Łapanie błędu jeśli istnieje wartość której nie można zamienić na float
except ValueError as e:
    print(f"Błąd w konwersji kolumny 'price' na numeryczną - {e}")
    raise


# Stworzenie nowej kolumny 'is_weekend' opisującej czy dany dzień wypadał w sobote lub niedzielę
# Zamiast bool wykorzystanie int gdzie 1 odpowiada True a 0 False
combined["is_weekend"] = (combined["date"].dt.dayofweek >= 5).astype(int)


# Funkcja przypisująca porę dnia
def get_time_of_day(hour):
    if 0 <= hour < 6:
        return 'night'
    elif 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    else:
        return 'evening'


# Stworzenie nowej kolumny 'time_of_day' na podstawie godziny odlotu przechowywanej w kolumnie 'dep_hour'
combined['time_of_day'] = combined['dep_hour'].apply(get_time_of_day)

# Przypisanie pierwszego dnia ze zbioru jako daty początkowej
start_date = pd.to_datetime('11-02-2022', format='%d-%m-%Y')

# Obliczenie ile dni do odlotu zostało i przypisanie tej wartości do nowej kolumny 'days_left'
combined['days_left'] = (combined['date']-start_date).dt.days

# Usunięcie kolumny z datą
combined = combined.drop(["date"], axis=1)


# Tworzenie słownika do zmiany nazw kolumn na lepiej opisujące dane
column_mapping = {
    "from": "departure_city",
    "time_taken": "flight_duration",
    "stop": "stops",
    "to": "arrival_city",
}

# Zmiana nazw kolumn w miejscu
combined.rename(columns=column_mapping, inplace=True)

# Zapisanie wyczyszczonego zbioru do pliku 'cleaned.csv'
combined.to_csv("./data/cleaned.csv", index=False)
