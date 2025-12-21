import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict, Dataset, load_from_disk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

import wandb
wandb.init(mode="offline")

def load_arrow(filepath):
    try:
        dataset = load_from_disk(filepath)
        print("Successfully loaded dataset from disk")
        return dataset
    except Exception as e:
        print(e)
        return None

dataset_path = "/content/drive/MyDrive/crypto_arrow_dataset"

import os
if os.path.exists(dataset_path):
    print(f"Path exists: {os.path.exists(dataset_path)}")
    print(f"Files in directory: {os.listdir(dataset_path)}")
else:
    print(f"Path does not exist. Current directory: {os.getcwd()}")
    print(f"Available files: {os.listdir('.')}")

dataset = load_arrow(dataset_path)

if dataset is None:
    try:
        dataset = DatasetDict({
            'train': load_from_disk(f"{dataset_path}/train"),
            'test': load_from_disk(f"{dataset_path}/test"),
            'validation': load_from_disk(f"{dataset_path}/validation")
        })
        print("dataset download throw DatasetDict")
    except Exception as e:
        print(e)
        exit()

print(f"\nDataset type: {type(dataset)}")
print(f"Dataset keys: {list(dataset.keys())}")

available_columns = dataset['train'].column_names
text_column = "BODY"
label_column = "label"
if text_column not in available_columns:
    text_column = available_columns[1]
    print(f" 'text' not found in available columns, use: {text_column}")

if label_column not in available_columns:
    label_cand = [col for col in available_columns if any(keyword in col.lower() for keyword in ['label', 'sentiment', 'class', 'category'])]
    if label_cand:
        label_column = label_cand[0]
        print(f"Label column:'{label_column}")
    else:
        print("Label column not found")

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples[text_column],
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

tokenized_datasets = DatasetDict()
for name, data in dataset.items():
    tokenized_datasets[name] = data.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
    )

columns_to_keep = ["input_ids", "attention_mask", "label"]
for name in tokenized_datasets.keys():
    tokenized_datasets[name].set_format(
        type="torch",
        columns=columns_to_keep
    )

print(tokenized_datasets["train"]["label"][:5])
print(type(tokenized_datasets["train"]["label"][0]))

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)

training_args = TrainingArguments(
    output_dir="./bert_encoder_for_volatility",
    overwrite_output_dir=False,
    learning_rate=3e-5,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=100,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to=None,
    save_total_limit=5
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted')
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

train_result = trainer.train()
trainer.save_model("./fine_tuned_bert_encoder")
encoder = model.distilbert
encoder.save_pretrained("./fine_tuned_bert_encoder_only")
tokenizer.save_pretrained("./fine_tuned_bert_encoder_only")

test_result = trainer.evaluate(tokenized_datasets["test"])
print(f"Test Accuracy: {test_result.get('eval_accuracy', 0):.4f}")
print(f"Test F1: {test_result.get('eval_f1', 0):.4f}")
print(f"Test Loss: {test_result.get('eval_loss', 0):.4f}")
# Test Accuracy: 0.8112
# Test F1: 0.8110
# Test Loss: 0.4444