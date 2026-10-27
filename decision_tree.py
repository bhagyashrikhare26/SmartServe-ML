
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Step 1: Load dataset
df = pd.read_csv("data/train.csv")

# Step 2: Separate input and output
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

# Step 4: Encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("category",
         OneHotEncoder(handle_unknown="ignore"),
         ["center_id", "meal_id"])
    ],
    remainder="passthrough"
)

# Step 5: Create Decision Tree model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regression", DecisionTreeRegressor(
        max_depth=12,
        min_samples_leaf=10,
        random_state=42
    ))
])

# Step 6: Train model
print("Training Decision Tree...")
model.fit(X_train, y_train)

# Step 7: Make predictions
predictions = model.predict(X_test)

# Step 8: Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nDecision Tree Results")
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
