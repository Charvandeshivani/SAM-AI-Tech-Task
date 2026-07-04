import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------
# Load Dataset
# -----------------------------------
df = pd.read_csv("cleaned_zomato.csv", encoding="latin1")

print("=" * 65)
print("        RESTAURANT RECOMMENDATION SYSTEM")
print("=" * 65)

# Remove rows with missing cuisines
df = df.dropna(subset=["Cuisines"])

# -----------------------------------
# User Input
# -----------------------------------
city = input("Enter City: ")
cuisine = input("Enter Preferred Cuisine: ")
max_price = int(input("Enter Maximum Price Range (1-4): "))
min_rating = float(input("Enter Minimum Rating (0-5): "))

# -----------------------------------
# Filter Restaurants
# -----------------------------------
filtered = df[
    (df["City"].str.contains(city, case=False, na=False)) &
    (df["Cuisines"].str.contains(cuisine, case=False, na=False)) &
    (df["Price range"] <= max_price) &
    (df["Aggregate rating"] >= min_rating)
].copy()

if filtered.empty:
    print("\n❌ No restaurants found matching your preferences.")
    exit()

# -----------------------------------
# Cosine Similarity
# -----------------------------------
vectorizer = TfidfVectorizer(stop_words="english")

tfidf_matrix = vectorizer.fit_transform(filtered["Cuisines"])

cosine_sim = cosine_similarity(tfidf_matrix)

filtered["Similarity Score"] = cosine_sim[0]

recommendations = filtered.sort_values(
    by=["Similarity Score", "Aggregate rating"],
    ascending=False
)

# -----------------------------------
# Display Recommendations
# -----------------------------------
print("\n✅ Top Restaurant Recommendations\n")

print(
    recommendations[
        [
            "Restaurant Name",
            "City",
            "Cuisines",
            "Price range",
            "Aggregate rating",
            "Votes",
            "Similarity Score"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nTask 5 Completed Successfully!")