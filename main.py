import json
import os
import torch
import gradio as gr
from vocabulary_class import Vocabulary
from src.models.lstm_model import SentimentLSTM
from src.models.scratch_transformer_model import SentimentTransformer



with open("/home/belal/projects/Sentiment-Analysis/config.json", "r") as f:
    config = json.load(f)

MODEL_TYPE = "scratch_transformer"

data_cfg = config["data"]
model_cfg = config[MODEL_TYPE]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_PATH = os.path.join(data_cfg["processed_dir"], "vocab.json")
MODEL_PATH = model_cfg["model_path"]

vocab = Vocabulary()
with open(VOCAB_PATH, "r") as f:
    vocab_data = json.load(f)
    vocab.stoi = vocab_data["stoi"]
    vocab.itos = {int(k):v for k,v in vocab_data["itos"].items()}

if MODEL_TYPE == "lstm":
    model = SentimentLSTM(
        vocab_size=len(vocab),
        pad_idx=vocab.stoi["<PAD>"],
        embed_dim=model_cfg["embed_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        output_dim=model_cfg["output_dim"],
        n_layers=model_cfg["n_layers"],
        bidirectional=model_cfg["bidirectional"],
        dropout=model_cfg["dropout"],
    ).to(device)
elif MODEL_TYPE == "scratch_transformer":
    model = SentimentTransformer(
        vocab_size = len(vocab),
        pad_idx=vocab.stoi["<PAD>"],
        embed_dim = model_cfg["embed_dim"],
        n_heads = model_cfg["n_head"],
        ff_dim = model_cfg["ff_dim"],
        n_layers = model_cfg["n_layers"],
        output_dim = model_cfg["output_dim"],
        dropout= model_cfg["dropout"],
    ).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def predict_sentiment(message, history):
    if not message.strip():
        return "Please type a valid review."

    tokens = Vocabulary.tokenize(message)
    token_ids = vocab.numericalize(tokens)
    tensor_input = (torch.tensor(token_ids,dtype=torch.long).unsqueeze(0).to(device))

    with torch.no_grad():
        logit = model(tensor_input)
        probability = torch.sigmoid(logit).item()

    if probability >= 0.5:
        sentiment = "Positive"
        confidence = probability*100
    else:
        sentiment = "negative"
        confidence = (1-probability)*100

    return f"Detected Sentiment: {sentiment} | Confidence: {confidence:.2f}%"

if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=predict_sentiment,
        title="Live Customer Sentiment Assistant",
        description="Type a customer review below to analyze its sentiment in real-time.",
        examples=[
            "I absolutely love this product! Highly recommended.",
            "The service was terrible and shipping took forever.",
        ],
    )

    demo.launch(share=True)