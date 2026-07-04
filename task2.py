import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("cleaned_zomato.csv", encoding="latin1")

print("=" * 50)
print("TASK 2 - ONLINE DELIVERY ANALYSIS")
print("=" * 50)

# -----------------------------
# Restaurants offering online delivery
# -----------------------------
online_delivery = df["Has Online delivery"].value_counts()

print("\nRestaurants Offering Online Delivery:")
print(online_delivery)

total_restaurants = len(df)
online_yes = online_delivery.get("Yes", 0)
online_no = online_delivery.get("No", 0)

print("\nTotal Restaurants:", total_restaurants)
print("Restaurants with Online Delivery:", online_yes)
print("Restaurants without Online Delivery:", online_no)

# -----------------------------
# Percentage
# -----------------------------
online_percentage = (online_yes / total_restaurants) * 100
offline_percentage = (online_no / total_restaurants) * 100

print(f"\nPercentage with Online Delivery : {online_percentage:.2f}%")
print(f"Percentage without Online Delivery : {offline_percentage:.2f}%")

# -----------------------------
# Average Ratings
# -----------------------------
avg_rating = (
    df.groupby("Has Online delivery")["Aggregate rating"]
    .mean()
)

print("\nAverage Ratings:")
print(avg_rating)

# -----------------------------
# Create Images Folder
# -----------------------------
os.makedirs("images", exist_ok=True)

# -----------------------------
# Pie Chart
# -----------------------------
plt.figure(figsize=(6,6))

plt.pie(
    [online_yes, online_no],
    labels=["Online Delivery", "No Online Delivery"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Restaurants Providing Online Delivery")

plt.savefig("images/online_delivery_pie.png", dpi=300)

plt.show()

# -----------------------------
# Bar Chart
# -----------------------------
plt.figure(figsize=(6,5))

avg_rating.plot(
    kind="bar",
    color=["green", "orange"],
    edgecolor="black"
)

plt.title("Average Rating by Online Delivery")
plt.xlabel("Online Delivery")
plt.ylabel("Average Rating")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig("images/online_delivery_rating.png", dpi=300)

plt.show()

print("\nCharts saved successfully!")
print("1. images/online_delivery_pie.png")
print("2. images/online_delivery_rating.png")

print("\nTask 2 Completed Successfully!")