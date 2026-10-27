
import pandas as pd

# Step 1: Load dataset
df = pd.read_csv("data/train.csv")

# Step 2: Select input features
X = df.drop(columns=["id", "num_orders"])

# Step 3: Select target column
y = df["num_orders"]

# Step 4: Display results
print("Preprocessing Successful!")

print("\nInput Features:")
print(X.columns.tolist())

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)

# tainning and testing

# Step 5: Split data based on week

# Use the last 20% of weeks for testing
split_week = int(df["week"].max() * 0.8)

train_mask = df["week"] <= split_week
test_mask = df["week"] > split_week

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining weeks:", df.loc[train_mask, "week"].min(),
      "to", df.loc[train_mask, "week"].max())

print("Testing weeks:", df.loc[test_mask, "week"].min(),
      "to", df.loc[test_mask, "week"].max())

print("\nData Split Successful!")
