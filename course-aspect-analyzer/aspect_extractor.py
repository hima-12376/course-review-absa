import spacy

nlp = spacy.load("en_core_web_sm")

def extract_aspects(review):
    doc = nlp(review)

    aspects = []

    for chunk in doc.noun_chunks:
        # Keep meaningful noun phrases
        if len(chunk.text) > 2:
            aspects.append(chunk.text.strip())

    # Remove duplicates
    aspects = list(set(aspects))

    return aspects