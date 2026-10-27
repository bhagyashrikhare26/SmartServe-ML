
# SmartServe-ML
## Smart College Canteen Food Demand Predictor

SmartServe-ML is a Machine Learning mini-project
that predicts food demand using historical order
data, meal information, pricing and promotions.

## Problem Statement

Predicting food demand can help canteen operators
plan food preparation and inventory more effectively,
potentially reducing food wastage.

## Dataset

Source: Kaggle Food Demand Forecasting

https://www.kaggle.com/competitions/food-demand-forecasting/data

- Records: 456,548
- Features: 7
- Target: num_orders
- Missing values: 0

## Machine Learning Algorithms

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. K-Nearest Neighbors Regressor

## Model Performance

| Algorithm | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Linear Regression | 149.31 | 260.16 | 0.4925 |
| Decision Tree | 136.92 | 269.71 | 0.4546 |
| Random Forest | 131.73 | 249.17 | 0.5345 |
| KNN | 88.17 | 197.37 | 0.7080 |

KNN achieved the best results among the four
tested models.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib / Plotly (if installed)

## Features

- Food demand prediction
- Dataset visualization
- Model performance comparison
- Interactive Streamlit dashboard

## Installation

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

## Run Application

python -m streamlit run app.py

## Limitations

This is a prototype based on a public
food-demand dataset, not actual college canteen
records. Predictions are estimates.

## Future Scope

- Collect real college canteen data
- Add holiday and attendance features
- Improve model performance
- Deploy the dashboard online
