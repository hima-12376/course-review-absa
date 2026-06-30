from sentiment_model import SentimentAnalyzer
from aspect_extractor import extract_aspects

print("Program started...")
print("Loading models...")

if __name__ == "__main__":

    print("Before model initialization")
    sentiment_model = SentimentAnalyzer()
    print("After model initialization")

    print("\nCourse Review Aspect-Based Sentiment Analyzer")
    print("Type 'exit' to quit\n")

    while True:

        review = input("Enter a course review: ")

        if review.lower() == "exit":
            print("Exiting...")
            break

        aspects = extract_aspects(review)

        if not aspects:
            print("No important aspects found.\n")
            continue

        print("\nExtracted Aspects & Sentiments:\n")

        for aspect in aspects:
            sentiment, confidence = sentiment_model.predict_aspect_sentiment(review, aspect)
            print(f"Aspect: {aspect} → Sentiment: {sentiment} ({confidence}% confidence)")

        print("\n--------------------------------\n")