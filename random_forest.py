
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load dataset
df = pd.read_csv("data/train.csv")

# Input and output
X = df.drop(columns=["id", "num_orders"])
y = df["num_orders"]

# Split data by week
split_week = int(df["week"].max() * 0.8)

train_mask = df["week"] <= split_week
test_mask = df["week"] > split_week

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]
y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

# Encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("category",
         OneHotEncoder(handle_unknown="ignore"),
         ["center_id", "meal_id"])
    ],
    remainder="passthrough"
)

# Create Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regression", RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    ))
])

# Train
print("Training Random Forest...")
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nRandom Forest Results")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

print("\nFirst 5 Predictions:")
for actual, predicted in zip(
    y_test.iloc[:5], predictions[:5]
):
    print(
        "Actual:", actual,
        "| Predicted:", round(max(0, predicted))
    )
