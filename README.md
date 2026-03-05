✈️ Airline Delay Prediction using Machine Learning
📌 Project Overview

Airline delays are a major challenge in the aviation industry, impacting airline operations, airport efficiency, and passenger satisfaction. Predicting delays can help airlines optimize schedules and reduce operational disruptions.

This project focuses on analyzing historical airline delay data and building machine learning models to predict flight delays based on different contributing factors.

The project uses machine learning techniques to:

1.Analyze delay causes

2.Identify key delay factors

3.Build predictive models

4.Evaluate model performance

5.Two machine learning algorithms were implemented:

6.Random Forest Regressor

7.XGBoost Regressor

These models help predict airline delays based on multiple operational and environmental factors.

🎯 Project Objectives

The main objectives of this project are:

Perform Exploratory Data Analysis (EDA) on airline delay data

Identify key factors responsible for delays

Apply machine learning models to predict delay patterns

Compare the performance of different algorithms

Extract insights from the dataset

📊 Dataset Description

The dataset contains historical airline delay records.

Each row represents flight delay statistics for a particular airport and carrier.

Key Features
Feature	Description
year	Year of flight record
month	Month of flight
carrier	Airline carrier code
airport	Airport code
arr_flights	Number of arriving flights
carrier_ct	Carrier-related delay count
weather_ct	Weather delay count
nas_ct	National Airspace System delay
security_ct	Security delay count
late_aircraft_ct	Late aircraft delay count
arr_delay	Total arrival delay
arr_del15	Flights delayed more than 15 minutes

Target variable used for prediction:

arr_del15
🧠 Machine Learning Workflow
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Feature Engineering
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Train-Test Split
     │
     ▼
Model Training
(Random Forest & XGBoost)
     │
     ▼
Model Evaluation
     │
     ▼
Delay Prediction
📂 Project Structure
Airline-Delay-Prediction
│
├── ML_Model_class.ipynb
├── Airline_Delay_Cause.csv
├── README.md
│
├── images
│   ├── heatmap.png
│   ├── delay_trend.png
│   ├── feature_importance.png
│   └── model_performance.png
🔎 Data Preprocessing

Before training the models, the dataset underwent multiple preprocessing steps.

Handling Missing Values

Missing numerical values were replaced with the median of the respective columns.

Example:

dataset[numerical_columns] = dataset[numerical_columns].fillna(dataset[numerical_columns].median())
Encoding Categorical Variables

Categorical variables such as carrier and airport were encoded using One-Hot Encoding.

pd.get_dummies(dataset, columns=categorical_columns)
📈 Exploratory Data Analysis (EDA)

EDA was performed to understand patterns and relationships within the dataset.

Key Visualizations

Correlation heatmap

Monthly delay trends

Delay cause analysis

Feature importance

These visualizations help understand which factors contribute the most to airline delays.

🤖 Machine Learning Models

Two regression models were used in this project.

🌲 Random Forest Regression

Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines their predictions.

Advantages

Handles large datasets

Reduces overfitting

High prediction accuracy

Example implementation:

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
⚡ XGBoost Regressor

XGBoost is an advanced gradient boosting algorithm known for its high performance and efficiency.

Advantages

Faster training

High predictive accuracy

Regularization support

Handles missing data effectively

Example implementation:

from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=1000, random_state=42)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
📉 Model Evaluation

To evaluate model performance, several metrics were used.

Evaluation Metrics
Metric	Description
R² Score	Measures prediction accuracy
RMSE	Root Mean Squared Error
MAE	Mean Absolute Error

Example:

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print("R2 Score:", r2_score(y_test, y_pred))

print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

print("MAE:", mean_absolute_error(y_test, y_pred))
⭐ Feature Importance

Feature importance analysis revealed the most influential factors affecting airline delays.

Important features include:

Late aircraft delays

Weather delays

Carrier delays

NAS delays

These factors significantly influence the overall delay prediction.

📊 Model Performance Comparison
Model	Performance
Random Forest	Good accuracy
XGBoost	Higher accuracy

XGBoost slightly outperformed Random Forest due to its gradient boosting optimization.

## 4. Evaluation Results
The final XGBoost model demonstrates high predictive power:
- **R2 Score:** 0.990616.
- **RMSE:** 294.39.
