import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("cleaned_zomato.csv", encoding="latin1")

print("=" * 60)
print("TASK 4 - RESTAURANT RATING PREDICTION")
print("=" * 60)

# -----------------------------
# Select Features and Target
# -----------------------------
features = [
    "Votes",
    "Price range",
    "Has Online delivery",
    "Cuisines"
]

target = "Aggregate rating"

# Keep required columns
df = df[features + [target]].dropna()

# -----------------------------
# Encode Categorical Columns
# -----------------------------
encoder = LabelEncoder()

df["Has Online delivery"] = encoder.fit_transform(df["Has Online delivery"])
df["Cuisines"] = encoder.fit_transform(df["Cuisines"])

# -----------------------------
# Split Data
# -----------------------------
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -----------------------------
# Build Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-" * 30)

print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

# -----------------------------
# Feature Importance
# -----------------------------
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance")
print("-" * 30)
print(importance)

print("\nTask 4 Completed Successfully!")