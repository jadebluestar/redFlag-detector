# train_redflag.py
import os
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --------- Config ----------
CSV_PATH = "redflag_dataset.csv"   # make sure this file is in the same folder
MODEL_NAME = "bert-base-uncased"   # if OOM, swap to "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 8
NUM_EPOCHS = 3
OUTPUT_DIR = "./redflag_model"
# --------------------------

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print("GPU:", torch.cuda.get_device_name(0))
    except Exception:
        pass

# 1) Load CSV
df = pd.read_csv(CSV_PATH)
label2id = {"red_flag": 0, "green_flag": 1}
df["label"] = df["label"].map(label2id)

# 2) Convert to HF Dataset + split
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# 3) Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

dataset = dataset.map(tokenize_batch, batched=True)

# safely remove 'text' and any accidental index columns
cols_to_remove = [c for c in ["text", "__index_level_0__", "index"] if c in dataset["train"].column_names]
if cols_to_remove:
    dataset = dataset.remove_columns(cols_to_remove)

# set torch format (only the columns needed)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 4) Data collator for dynamic padding (saves memory)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 5) Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, id2label={0: "red_flag", 1: "green_flag"}, label2id=label2id
)

# 6) Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", pos_label=1)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# 7) TrainingArguments - fp16 enables mixed precision (faster + less memory on supported GPUs)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,                # use mixed-precision if available - speeds up and reduces memory
    push_to_hub=False
)

# 8) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 9) Train
trainer.train()

# 10) Save
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved model to", OUTPUT_DIR)
