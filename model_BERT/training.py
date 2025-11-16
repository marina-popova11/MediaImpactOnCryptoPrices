import torch
import numpy as np
import pandas as pd
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_arrow(filepath):
    try:
        dataset = load_from_disk(filepath)
        print("Successfully loaded dataset from disk")
        return dataset
    except Exception as e:
        print(e)
        return None

dataset_path = "./crypto_sentiment_dataset"
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

for name, data in dataset.items():
    print(f"{name.upper()} chapter:")
    print(f"{data.column_names}")
    print(f"Count of notes: {len(data)}")
    if len(data) > 0:
        print("Example:")
        for col in data.column_names:
            print(f"{col}: {data[0][col]}")

available_columns = dataset['train'].column_names
text_column = "text"
label_column = "sentiment"
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

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(text):
    return tokenizer(
        text[text_column],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

tokenized_datasets = DatasetDict()
for name, data in dataset.items():
    tokenized_datasets[name] = data.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
    )

if label_column:
    for name in tokenized_datasets.keys():
        tokenized_datasets[name] = tokenized_datasets[name].rename_column(label_column, "labels")

columns_to_keep = ["input_ids", "attention_mask"]
if label_column:
    columns_to_keep.append("labels")

for name in tokenized_datasets.keys():
    tokenized_datasets[name].set_format("torch", columns=columns_to_keep)

print("Tokenization finished")

if label_column:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )

    training_args = TrainingArguments(
        output_dir="./bert_sentiment_crypto",
        overwrite_output_dir=True,
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,
        save_total_limit=2,
    )

    # Calculates the accuracy of the model during validation
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    train_result = trainer.train()
    trainer.save_model("./fine_tuned_bert_crypto_sentiment")
    trainer.save_state()

    print("Further education finished")

    print("Tests")
    test_result = trainer.evaluate(tokenized_datasets["test"])
    print(f"Accuracy for test: {test_result.get("eval_accuracy", 0):.4f}")


label_column = "sentiment"
text_column = "text"
model_name = "bert-base-uncased"

class BertEmbeddingExtractor:
    def __init__(self, model_path=None):
        if model_path and label_column:
            try:
                self.model = AutoModel.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                print("Using a pre-trained model for embeddings")
            except:
                self.model = AutoModel.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                print("Using the basic model for embeddings")
        else:
            self.model = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("Using the basic model for embeddings")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"model downloaded to {self.device}")

    def get_embeddings(self, text, batch_size=32):
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(text), batch_size):
                batch_text = text[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(self.device)

                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(batch_embeddings.cpu().numpy())
        return np.vstack(embeddings)

embedding_extractor = BertEmbeddingExtractor("./fine_tuned_bert_crypto_sentiment" if label_column else None)

def get_text_from_dataset(dataset):
    return dataset[text_column]

# train_embeddings = embedding_extractor.get_embeddings(get_text_from_dataset(dataset['train']))
# print(f"Train emb: {train_embeddings.shape}")
#
# test_embeddings = embedding_extractor.get_embeddings(get_text_from_dataset(dataset['test']))
# print(f"Test emb: {test_embeddings.shape}")
#
# val_embeddings = embedding_extractor.get_embeddings(get_text_from_dataset(dataset['validation']))
# print(f"Validation emb: {val_embeddings.shape}")