import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import DistilBertTokenizer, DataCollatorWithPadding
from src.models.BERT_transformer_model import BERTTransformerSentimentRNN


with open("/config.json", "r") as f:
    config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#---------------------------------------data_loader---------------------------------------

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class BERTTransformerIMDBDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length = 256):
        df = pd.read_csv(csv_path)
        self.reviews = df["review"].tolist()
        self.labels = [1 if s=="positive" else 0 for s in df["sentiment"]]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        encoding = self.tokenizer(review, truncation= True, max_length= self.max_length, return_tensors="pt")
        return {"input_ids": encoding["input_ids"].squeeze(0), "attention_mask": encoding["attention_mask"].squeeze(0), "labels": labels}

full_dataset = BERTTransformerIMDBDataset(csv_path=config["raw_path"], tokenizer= tokenizer, max_length=config["BERT_max_length"])
train_size=int(len(full_dataset)*0.8)
val_size=len(full_dataset)-train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size,val_size], generator=torch.Generator().manual_seed(42))

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_loader = DataLoader(dataset=train_dataset, batch_size= config["BERT_transformer_batch_size"], shuffle= True, collate_fn=data_collator)
val_loader = DataLoader(dataset=val_dataset, batch_size= config["BERT_transformer_batch_size"], collate_fn= data_collator)


#---------------------------------------training---------------------------------------

def binary_accuracy(preds, y):
    rounded_preds = preds > 0
    correct = (rounded_preds == y).float()
    return correct.mean()

def train(model, loader, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        predictions = model(input_ids, attention_mask).squeeze(1)

        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(parameters= model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss/len(loader), epoch_acc/len(loader)

def evaluate(model, loader, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            predictions = model(input_ids, attention_mask).squeeze(1)

            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss/len(loader), epoch_acc/len(loader)

model = BERTTransformerSentimentRNN(dropout= config["BERT_transformer_dropout"]).to(device)
optimizer = optim.AdamW(model.parameters(), lr = config["BERT_transformer_learning_rate"])
criterion = nn.BCEWithLogitsLoss()

print(f"Starting Transformer Fine-Tuning on {device}...")

best_valid_loss = float("inf")

for epoch in range(config["BERT_transformer_epochs"]):
    train_loss, train_acc = train(model= model, loader=train_loader, optimizer=optimizer, criterion=criterion)
    val_loss, val_acc = evaluate(model=model, loader=val_loader, criterion=criterion)

    print(f"Epoch: {epoch + 1:02}")
    print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%")
    print(f"\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc * 100:.2f}%")

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        torch.save(model.state_dict(), config["BERT_transformer_model_path"])
        print(f"--- Best Transformer Saved ---")