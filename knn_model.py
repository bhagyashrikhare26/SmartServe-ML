
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Step 1: Load dataset
df = pd.read_csv("data/train.csv")

# Step 2: Input and target
X = df.drop(columns=["id", "num_orders"])
y = df["num_orders"]

# Step 3: Split by week
split_week = int(df["week"].max() * 0.8)

train_mask = df["week"] <= split_week
test_mask = df["week"] > split_week

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]
y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

# Step 4: Encode categories and scale numbers
preprocessor = ColumnTransformer(
    transformers=[
        (
            "category",
            OneHotEncoder(handle_unknown="ignore"),
            ["center_id", "meal_id"]
        ),
        (
            "numeric",
            StandardScaler(),
            [
                "week",
                "checkout_price",
                "base_price",
                "emailer_for_promotion",
                "homepage_featured"
            ]
        )
    ]
)

# Step 5: Create KNN model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regression", KNeighborsRegressor(
        n_neighbors=5,
        weights="distance",
        n_jobs=-1
    ))
])

# Step 6: Train model
print("Training KNN...")
model.fit(X_train, y_train)

# Step 7: Predict
print("Making predictions...")
predictions = model.predict(X_test)

# Step 8: Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nKNN Results")
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

# Step 9: Save trained model
import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/knn_model.pkl")

print("\nKNN Model Saved Successfully!")
