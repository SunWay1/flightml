# Importowanie bibliotek do wizualizacji, metryk ewaluacyjnych oraz AutoGluon
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np

# Wczytanie danych
df = pd.read_csv("./data/final.csv")
label = 'price'

# Podział zbioru na treningowy i testowy (80/20)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Trenowanie mniejszych modeli z użyciem AutoGluon
# predictor = TabularPredictor(label="price", path="agModels_sm/").fit(
#     train_data=train_df,

#     # Preset zbalansowany między jakością a szybkością uczenia
#     presets="medium_quality_faster_train",

#     # Wykluczenie cięższych lub niepotrzebnych typów modeli
#     excluded_model_types=[
#         "KNeighborsClassifier",         # KNN – mało efektywny przy dużych zbiorach
#         "NeuralNetTorchClassifier",     # Sieci neuronowe – wolniejsze i cięższe
#         "CatBoostClassifier",           # Podobny do LightGBM, ale cięższy
#         "ExtraTreesClassifier",
#         "FastText",                     # Modele tekstowe – zbędne w tym przypadku
#         "ImagePredictor",               # Nie używamy obrazów
#         "TabTransformerClassifier",     # Sieci transformerowe dla danych tabularycznych
#         "TextPredictor",                # Predykcja tekstu – niepotrzebna
#         "NeuralNetFastAI"               # Inna wersja sieci neuronowych
#     ],

#     # Użyte algorytmy – tylko lekkie modele boostujące + możliwość ensemblingu wagowego
#     hyperparameters={
#         "GBM": {},                      # LightGBM – lekki, wydajny model do danych tabularycznych
#         "XGB": {},                      # XGBoost – szybki i skuteczny, ale mniejszy niż CatBoost
#         # Umożliwia wybranie najlepszego modelu spośród tych bez ensemblingu
#         "ENS_WEIGHTED": {}
#     },

#     # Wyłączenie baggingu – brak wielu foldów, zmniejszenie liczby plików
#     num_bag_folds=0,

#     # Wyłączenie stacking – brak wielu poziomów modeli ensemble
#     num_stack_levels=0,
# )

predictor = TabularPredictor.load("./agModels_sm/")

# Wyświetlenie najlepszego modelu
print("Najlepszy model:", predictor.model_best)

# Lista nazw modeli
model_names = predictor.model_names()

# Przygotowanie danych testowych
X_test = test_df.drop(columns=[label])
y_test = test_df[label]

# Obliczanie metryk dla zbioru testowego
metrics = {'model': [], 'MAE': [], 'RMSE': [], 'R2': []}

# Obliczenia metryk predykcji na zbiorze testowym
for model in model_names:
    preds = predictor.predict(X_test, model=model)
    metrics['model'].append(model)
    metrics['MAE'].append(mean_absolute_error(y_test, preds))
    metrics['RMSE'].append(np.sqrt(mean_squared_error(y_test, preds)))
    metrics['R2'].append(r2_score(y_test, preds))

# Konwersja słownika metryk na DataFrame
metrics_df = pd.DataFrame(metrics).set_index('model')


# Wykres słupkowy dla jednej wybranej metryki
def plot_metric(metric_name, df):
    plt.figure(figsize=(4, 2))
    colors = plt.cm.Pastel1.colors  # pastelowe kolory

    bars = []
    for i, (model, value) in enumerate(df[metric_name].items()):
        bar = plt.bar(
            i, value,
            width=0.4,
            edgecolor='black',
            color=colors[i % len(colors)],
            linewidth=1.2
        )
        bars.append(bar)

        # wartość słupka wyżej dla lepszej widoczności
        plt.text(i, value + df[metric_name].max() * 0.03,
                 f'{value:.2f}', ha='center', va='bottom', fontsize=10)

    plt.title(f'{metric_name}', fontsize=12)
    plt.xlabel('')
    plt.xticks([])  # usunięcie etykiet z osi X

    # dodanie legendy z nazwami modeli
    plt.legend([bar[0] for bar in bars], df.index, loc='lower right')

    # zwiększenie limitu Y
    plt.ylim(0, df[metric_name].max() * 1.25)

    plt.grid(True, axis='y', linestyle='--', alpha=0.5, color='gray')
    plt.tight_layout()
    plt.savefig(f"./charts_sm/{metric_name.lower()}.png", dpi=300)
    plt.close()


# Wykresy ważności cech (feature importance)
def plot_feature_importance(predictor, model_names, test_data):
    for model in model_names:
        importance_df = predictor.feature_importance(
            data=test_data, model=model)

        plt.figure(figsize=(10, 6))

        bars = plt.barh(
            importance_df.index[::-1],
            importance_df['importance'][::-1],
            edgecolor='black',
            color='lightblue',  # pastelowy kolor poziomych słupków
            linewidth=1.2
        )

        max_val = importance_df['importance'].max()

        # wartości słupków przesunięte na prawo
        for bar in bars:
            width = bar.get_width()
            plt.text(width + max_val * 0.02,
                     bar.get_y() + bar.get_height()/2,
                     f'{width:.2f}', va='center', fontsize=9)

        plt.title(f'Ważność cech (feature importance)', fontsize=12)
        plt.xlim(0, max_val * 1.15)  # zwiększony zakres X
        plt.grid(True, axis='x', linestyle='--', alpha=0.5, color='gray')
        plt.tight_layout()
        plt.savefig(f"./charts_sm/{model}_fi.png", dpi=300)
        plt.close()


# Generowanie wykresów metryk
plot_metric('MAE', metrics_df)
plot_metric('RMSE', metrics_df)
plot_metric('R2', metrics_df)

# Generowanie wykresów ważności cech
# plot_feature_importance(predictor, model_names, test_df)
