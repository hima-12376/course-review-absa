import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentAnalyzer:

    def __init__(self):
        print("Loading Dual Fine-Tuned Hybrid Model...")

        # 🔵 Fine-tuned DeBERTa
        self.deberta_model_name = "absa_model_small"
        self.deberta_tokenizer = AutoTokenizer.from_pretrained(self.deberta_model_name)
        self.deberta_model = AutoModelForSequenceClassification.from_pretrained(self.deberta_model_name)

        # 🔴 Fine-tuned RoBERTa
        self.roberta_model_name = "absa_roberta_model"
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(self.roberta_model_name)
        self.roberta_model = AutoModelForSequenceClassification.from_pretrained(self.roberta_model_name)

        self.deberta_model.eval()
        self.roberta_model.eval()

        self.labels = ["Negative", "Neutral", "Positive"]

        print("Hybrid ensemble loaded successfully!\n")

    # -------------------------------------------------
    # Clause Isolation
    # -------------------------------------------------
    def isolate_clause(self, review, aspect):

        sentences = re.split(r'[.!?]', review)
        target_sentence = review

        for sentence in sentences:
            if aspect.lower() in sentence.lower():
                target_sentence = sentence.strip()
                break

        clauses = re.split(
            r',|\bbut\b|\bhowever\b|\balthough\b|\bthough\b',
            target_sentence,
            flags=re.IGNORECASE
        )

        for clause in clauses:
            if aspect.lower() in clause.lower():
                return clause.strip()

        return target_sentence


    # -------------------------------------------------
    # Hybrid Prediction
    # -------------------------------------------------
    def predict_aspect_sentiment(self, review, aspect):

        clause = self.isolate_clause(review, aspect)

        # ---------------------------
        # 🔵 DeBERTa Prediction
        # ---------------------------
        deberta_inputs = self.deberta_tokenizer(
            clause,
            aspect,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            deberta_outputs = self.deberta_model(**deberta_inputs)

        deberta_probs = F.softmax(deberta_outputs.logits, dim=1)[0]

        deberta_pred = torch.argmax(deberta_probs).item()

        # ---------------------------
        # 🔴 RoBERTa Prediction
        # ---------------------------
        roberta_inputs = self.roberta_tokenizer(
            clause,
            aspect,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            roberta_outputs = self.roberta_model(**roberta_inputs)

        roberta_probs = F.softmax(roberta_outputs.logits, dim=1)[0]

        roberta_pred = torch.argmax(roberta_probs).item()

        # ---------------------------
        # 🔥 Hybrid Fusion
        # ---------------------------
        final_probs = (0.5 * deberta_probs) + (0.5 * roberta_probs)
        final_probs = final_probs / final_probs.sum()

        prediction = torch.argmax(final_probs).item()
        confidence = torch.max(final_probs).item()

        sentiment = self.labels[prediction]

        # Hybrid disagreement rule
        if sentiment != "Negative" and 0.45 < confidence < 0.60:
            sentiment = "Neutral"

        # -------------------------------------------------
        # ⭐ TERMINAL COMPARISON OUTPUT
        # -------------------------------------------------

        print("\n-------------------------------------")
        print("Aspect:", aspect)
        print("Clause:", clause)

        print("\nDeBERTa probabilities:", deberta_probs.tolist())
        print("DeBERTa Prediction:", self.labels[deberta_pred])

        print("\nRoBERTa probabilities:", roberta_probs.tolist())
        print("RoBERTa Prediction:", self.labels[roberta_pred])

        print("\nHybrid probabilities:", final_probs.tolist())
        print("Hybrid Prediction:", sentiment)
        print("Hybrid Confidence:", round(confidence * 100, 2), "%")
        print("-------------------------------------")

        return sentiment, round(confidence * 100, 2)