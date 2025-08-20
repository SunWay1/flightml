# Importowanie biblioteki do pracy ze zbiorami
import pandas as pd

# Wczytywanie danych
economy = pd.read_csv("./data/economy.csv")
business = pd.read_csv("./data/business.csv")

# Dodawanie kolumny 'class' wskazującej typ biletu
economy['class'] = 'economy'
business['class'] = 'business'

# Połączenie danych w jeden zbiór
flights = pd.concat([economy, business], ignore_index=True)

# Zapis do nowego pliku CSV
flights.to_csv("./data/combined.csv", index=False)
