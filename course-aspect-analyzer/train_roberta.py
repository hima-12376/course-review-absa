import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import numpy as np
import evaluate

print("Loading dataset...")

df = pd.read_csv("absa_coursera_small.csv")
dataset = Dataset.from_pandas(df)

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(example):
    return tokenizer(
        example["text"],
        example["aspect"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(preprocess, batched=False)
dataset = dataset.train_test_split(test_size=0.1)
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return metric.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir="./absa_roberta_model",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="no",
    logging_steps=100,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("Training RoBERTa...")
trainer.train()

trainer.save_model("./absa_roberta_model")
tokenizer.save_pretrained("./absa_roberta_model")

print("RoBERTa fine-tuning complete!")