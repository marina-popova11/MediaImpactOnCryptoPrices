import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

df = pd.read_csv("btc_hourly_news.csv")
print(df.shape)

df = df.dropna(subset=["text"])
dataset = Dataset.from_pandas(df)

model_path = "/content/drive/MyDrive/bert_final_81_percent"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForMaskedLM.from_pretrained(model_path)

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

import wandb
wandb.init(mode="offline")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.20,
)

training_args = TrainingArguments(
    output_dir="./distilbert_dapt",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=200,
    lr_scheduler_type="linear",
    save_steps=300,
    save_total_limit=2,
    eval_strategy="no",
    logging_steps=300,
    learning_rate=5e-5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
output_dir = "./distilbert_dapt_finetuned_retry"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("DAPT обучение завершено! Модель сохранена в:", output_dir)

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import DataCollatorForLanguageModeling

def compute_mlm_perplexity(model, dataset, tokenizer, batch_size=8, mlm_prob=0.20):
    model.eval()
    device = next(model.parameters()).device

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_prob
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

    total_loss = 0
    total_tokens = 0

    loss_fct = CrossEntropyLoss(ignore_index=-100)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        target_tokens = (labels != -100).sum().item()

        total_loss += loss.item() * target_tokens
        total_tokens += target_tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

df_test = pd.read_csv("test_data.csv")
df = df_test.iloc[:500]
val_dataset = Dataset.from_pandas(df)
val_tokenized = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
model_path = "./distilbert_dapt_finetuned_retry"
model = DistilBertForMaskedLM.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
ppl = compute_mlm_perplexity(model, val_tokenized, tokenizer)
print("Perplexity:", ppl) # Perplexity: 12.628557205200195