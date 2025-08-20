# ✈️ Flight Ticket Price Prediction

This project demonstrates a complete machine learning pipeline for predicting airline ticket prices based on various flight parameters. It includes data preprocessing, model training using AutoGluon, and deployment of a lightweight web application built with Flask.
👉 Try it here: [flightml.xyz](https://www.flightml.xyz/)

## 📊 Dataset & Preprocessing

The original dataset contained several categorical and temporal features related to flights, such as:

- Departure and arrival cities
- Airline name
- Class (economy/business)
- Flight times and dates
- Number of stops

To prepare the dataset for modeling, the following preprocessing steps were applied:

- **Datetime conversion**: Departure and arrival times were parsed into hours and minutes, and used to compute total flight duration (in minutes).
- **Weekend feature**: A binary feature (`is_weekend`) was added to indicate whether the flight is scheduled on a weekend.
- **Days left**: The number of days between the date of prediction and the flight date.
- **Missing values**: Rows with incorrect time formats or missing required fields were removed.
- **Categorical encoding**: Categorical values such as airline, class, and cities were left as strings, as AutoGluon automatically handles encoding.
- **Feature scaling**: A `StandardScaler` was applied to numeric features like duration, hours, minutes, etc.

## 🤖 Model Training

Models were trained using **AutoGluon TabularPredictor**, which automatically trains multiple models and performs ensembling.

Two main configurations were used:

### Full model (local, no deployment, not included in this repository):
- Preset: `best_quality`
- Includes bagging and stacking (3 levels)
- Top models: `WeightedEnsemble_L3`, `LightGBM_BAG_L2`
- Best MAE: 34 
- Best R²: ~0.99

### Lightweight model (for deployment):
- Preset: `medium_quality_faster_train`
- Bagging and stacking disabled (for smaller size)
- Selected models: LightGBM, XGBoost, WeightedEnsemble
- MAE: ~40  
- RMSE: ~75  
- Still provides solid predictions with much smaller resource requirements

## 🌐 Web Application

A web application was developed using:

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python Flask
- Hosted for free on [Render.com](https://render.com)

The app accepts user input such as departure and arrival cities, date, time, number of stops, and predicts the price of the ticket. Thanks to AutoGluon's category encoder, it supports **custom user input**, including new cities or airlines not present in the training dataset.


## 📈 Conclusion

This project shows that flight ticket prices can be predicted with reasonable accuracy using tree-based models and well-engineered features. Even with strict deployment constraints, a simplified version of the model performs well enough for real-world use.
