import pandas as pd
import torch
import numpy as np
import evaluate

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from torch.nn import CrossEntropyLoss

print("Loading dataset...")

# ------------------------------------------------
# 1️⃣ Load Dataset
# ------------------------------------------------
df = pd.read_csv("absa_coursera_small.csv")

print("Label distribution before training:")
print(df["label"].value_counts())

dataset = Dataset.from_pandas(df)

# ------------------------------------------------
# 2️⃣ Load Base Model + Tokenizer
# ------------------------------------------------
model_name = "microsoft/deberta-v3-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# ------------------------------------------------
# 3️⃣ Preprocess (MUST MATCH INFERENCE)
# ------------------------------------------------
def preprocess(example):
    return tokenizer(
        example["text"],     # full review
        example["aspect"],   # aspect as second sequence
        truncation=True,
        padding="max_length",
        max_length=128
    )

print("Tokenizing dataset...")
dataset = dataset.map(preprocess)

dataset = dataset.train_test_split(test_size=0.1)

dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch")

# ------------------------------------------------
# 4️⃣ Load Model
# ------------------------------------------------
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)

# ------------------------------------------------
# 5️⃣ Handle Class Imbalance Using Weights
# Label mapping:
# 0 → Negative
# 1 → Neutral
# 2 → Positive
# ------------------------------------------------

class_weights = torch.tensor([4.5, 4.0, 1.0])

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ------------------------------------------------
# 6️⃣ Metrics
# ------------------------------------------------
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return metric.compute(predictions=preds, references=labels)

# ------------------------------------------------
# 7️⃣ Training Arguments
# ------------------------------------------------
training_args = TrainingArguments(
    output_dir="./absa_weighted_model",
    num_train_epochs=2,               # You said you cannot increase epochs
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_strategy="no",
    evaluation_strategy="no",
    logging_steps=100,
    report_to="none"
)

# ------------------------------------------------
# 8️⃣ Train
# ------------------------------------------------
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("Starting training...")
trainer.train()

# ------------------------------------------------
# 9️⃣ Save Model Properly
# ------------------------------------------------
trainer.save_model("./absa_model_small")
tokenizer.save_pretrained("./absa_model_small")

print("🔥 Training complete!")
print("Model saved to: ./absa_model_small")