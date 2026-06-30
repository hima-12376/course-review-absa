import pandas as pd
from aspect_extractor import extract_aspects

print("Script started...")

def rating_to_label(rating):
    if rating >= 4:
        return 2  # Positive
    elif rating == 3:
        return 1  # Neutral
    else:
        return 0  # Negative

# 🔥 Load dataset
df = pd.read_csv("Coursera_reviews.csv")

print("Dataset loaded successfully!")
print("Total reviews in CSV:", len(df))

# 🔥 Start small for testing
MAX_REVIEWS = 2000   # Change to 10000 or 20000 later

absa_data = []

for i, row in df.iterrows():

    if i >= MAX_REVIEWS:
        break

    if i % 500 == 0:
        print(f"Processing review {i}...")

    review = str(row["reviews"])
    rating = row["rating"]

    # Skip empty reviews
    if review.strip() == "":
        continue

    aspects = extract_aspects(review)

    for aspect in aspects:
        absa_data.append({
            "text": review,
            "aspect": aspect,
            "label": rating_to_label(rating)
        })

print("Finished processing reviews.")

# 🔥 Convert to DataFrame
absa_df = pd.DataFrame(absa_data)

# Remove duplicates (optional but good practice)
absa_df.drop_duplicates(inplace=True)

# Save dataset
absa_df.to_csv("absa_coursera_small.csv", index=False)

print("ABSA dataset created successfully!")
print("Total ABSA samples:", len(absa_df))