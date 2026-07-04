import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("cleaned_zomato.csv", encoding="latin1")

print("=" * 50)
print("TASK 3 - PRICE RANGE ANALYSIS")
print("=" * 50)

# -----------------------------
# Distribution of Restaurants
# -----------------------------
price_count = df["Price range"].value_counts().sort_index()

print("\nDistribution of Restaurants Across Price Ranges:")
print(price_count)

# -----------------------------
# Average Rating by Price Range
# -----------------------------
price_rating = (
    df.groupby("Price range")["Aggregate rating"]
      .mean()
      .sort_index()
)

print("\nAverage Rating by Price Range:")
print(price_rating)

# Highest Rated Price Range
highest = price_rating.idxmax()
highest_rating = price_rating.max()

print(f"\nPrice Range with Highest Average Rating: {highest}")
print(f"Average Rating: {highest_rating:.2f}")

# -----------------------------
# Create Images Folder
# -----------------------------
os.makedirs("images", exist_ok=True)

# -----------------------------
# Bar Chart
# -----------------------------
plt.figure(figsize=(8,5))

price_count.plot(
    kind="bar",
    color="steelblue",
    edgecolor="black"
)

plt.title("Distribution of Restaurants by Price Range")
plt.xlabel("Price Range")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig("images/price_range_bar.png", dpi=300)

plt.show()

# -----------------------------
# Pie Chart
# -----------------------------
plt.figure(figsize=(6,6))

plt.pie(
    price_count,
    labels=price_count.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Restaurant Distribution by Price Range")

plt.savefig("images/price_range_pie.png", dpi=300)

plt.show()

print("\nCharts Saved Successfully!")
print("1. images/price_range_bar.png")
print("2. images/price_range_pie.png")

print("\nTask 3 Completed Successfully!")