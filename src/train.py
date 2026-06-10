import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import get_loaders
from src.vocabulary_class import Vocabulary
from src.models.lstm_model import SentimentLSTM
from src.models.scratch_transformer_model import SentimentTransformer


with open("/home/belal/projects/Sentiment-Analysis/config.json", "r") as f:
    config = json.load(f)

MODEL_TYPE = "scratch_transformer"

data_cfg = config["data"]
model_cfg = config[MODEL_TYPE]

DATA_PATH = os.path.join(data_cfg["processed_dir"], "processed_data.pt")
VOCAB_PATH = os.path.join(data_cfg["processed_dir"], "vocab.json")
MODEL_PATH = model_cfg["model_path"]

vocab_instance = Vocabulary()
with open(VOCAB_PATH, "r") as f:
    vocab_data = json.load(f)
    vocab_instance.stoi = vocab_data["stoi"]
    vocab_instance.itos = {int(k):v for k,v in vocab_data["itos"].items()}

VOCAB_SIZE = len(vocab_instance)
PAD_IDX = vocab_instance.stoi["<PAD>"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, val_loader = get_loaders(DATA_PATH, model_cfg["batch_size"], PAD_IDX)

def binary_accuracy(preds, y):
    rounded_preds = preds > 0
    correct = (rounded_preds == y).float()
    return correct.mean()


def train(model, train_loader, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for texts, masks, labels in train_loader:
        texts = texts.to(device)
        masks = masks.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()
        if isinstance(model, SentimentLSTM):
            predictions = model(texts).squeeze(1)
        elif isinstance(model, SentimentTransformer):
            predictions = model(texts, mask=masks).squeeze(1)

        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(train_loader), epoch_acc / len(train_loader)


def evaluate(model, val_loader, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for texts, masks, labels in val_loader:
            texts = texts.to(device)
            masks = masks.to(device)
            labels = labels.to(device).float()

            if isinstance(model, SentimentLSTM):
                predictions = model(texts).squeeze(1)
            elif isinstance(model, SentimentTransformer):
                predictions = model(texts, mask=masks).squeeze(1)

            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(val_loader), epoch_acc / len(val_loader)


if MODEL_TYPE == "lstm":
    model = SentimentLSTM(
        vocab_size=VOCAB_SIZE,
        embed_dim=model_cfg["embed_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        output_dim=model_cfg["output_dim"],
        n_layers=model_cfg["n_layers"],
        bidirectional=model_cfg["bidirectional"],
        dropout=model_cfg["dropout"],
        pad_idx=PAD_IDX,
    ).to(device)
elif MODEL_TYPE == "scratch_transformer":
    model = SentimentTransformer(
        vocab_size = VOCAB_SIZE,
        embed_dim = model_cfg["embed_dim"],
        n_heads = model_cfg["n_head"],
        ff_dim = model_cfg["ff_dim"],
        n_layers = model_cfg["n_layers"],
        output_dim = model_cfg["output_dim"],
        dropout= model_cfg["dropout"],
        pad_idx= PAD_IDX,
    ).to(device)



optimizer = optim.Adam(model.parameters(), lr= model_cfg["learning_rate"])
criterion = nn.BCEWithLogitsLoss()

print(f"Starting {MODEL_TYPE} Training on {device}...")

best_valid_loss = float('inf')

for epoch in range(model_cfg["epochs"]):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    print(f"Epoch: {epoch + 1:02}")
    print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%")
    print(f"\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc * 100:.2f}%")

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }, MODEL_PATH)
        vocab_instance.save(VOCAB_PATH)
        print(f"--- Best Model Saved at Epoch {epoch + 1} ---")