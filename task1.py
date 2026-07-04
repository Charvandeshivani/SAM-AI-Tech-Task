import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Load Dataset
# -----------------------------
file_path = r"zomato/zomato.csv"

encodings = ["utf-8", "latin1", "ISO-8859-1", "cp1252"]

df = None

for enc in encodings:
    try:
        df = pd.read_csv(file_path, encoding=enc)
        print(f"\nDataset loaded successfully using '{enc}' encoding.")
        break
    except Exception:
        pass

if df is None:
    print("Error: Unable to read the dataset.")
    exit()

# -----------------------------
# Dataset Information
# -----------------------------
print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\nDuplicate Records Before:", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicate Records After:", df.duplicated().sum())

# -----------------------------
# Check Required Columns
# -----------------------------
required_columns = ["City", "Aggregate rating"]

for col in required_columns:
    if col not in df.columns:
        print(f"\nError: Column '{col}' not found.")
        print("\nAvailable Columns:")
        print(df.columns.tolist())
        exit()

# Remove missing values
df = df.dropna(subset=["City", "Aggregate rating"])

# -----------------------------
# Analysis
# -----------------------------
print("\nTotal Restaurants:", len(df))
print("Total Cities:", df["City"].nunique())

city_count = df["City"].value_counts()

print("\nTop 10 Cities:")
print(city_count.head(10))

city_rating = (
    df.groupby("City")["Aggregate rating"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Rating by City:")
print(city_rating)

# -----------------------------
# Create Images Folder
# -----------------------------
os.makedirs("images", exist_ok=True)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10,6))

city_count.head(10).plot(
    kind="bar",
    color="steelblue",
    edgecolor="black"
)

plt.title("Top 10 Cities by Number of Restaurants")
plt.xlabel("City")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig("images/top10_cities.png", dpi=300)

plt.show()

# -----------------------------
# Save Cleaned Dataset
# -----------------------------
df.to_csv("cleaned_zomato.csv", index=False)

print("\nCleaned dataset saved as cleaned_zomato.csv")
print("Chart saved as images/top10_cities.png")
print("\nTask Completed Successfully!")