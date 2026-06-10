import json
import os
import torch
import gradio as gr
from vocabulary_class import Vocabulary
from src.models.lstm_model import SentimentLSTM
from src.models.scratch_transformer_model import SentimentTransformer



with open("/home/belal/projects/Sentiment-Analysis/config.json", "r") as f:
    config = json.load(f)

data_cfg = config["data"]
lstm_model_cfg = config["lstm"]
transformer_model_cfg = config["scratch_transformer"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_PATH = os.path.join(data_cfg["processed_dir"], "vocab.json")

vocab = Vocabulary()
with open(VOCAB_PATH, "r") as f:
    vocab_data = json.load(f)
    vocab.stoi = vocab_data["stoi"]
    vocab.itos = {int(k):v for k,v in vocab_data["itos"].items()}

lstm_model = SentimentLSTM(
    vocab_size=len(vocab),
    pad_idx=vocab.stoi["<PAD>"],
    embed_dim=lstm_model_cfg["embed_dim"],
    hidden_dim=lstm_model_cfg["hidden_dim"],
    output_dim=lstm_model_cfg["output_dim"],
    n_layers=lstm_model_cfg["n_layers"],
    bidirectional=lstm_model_cfg["bidirectional"],
    dropout=lstm_model_cfg["dropout"],
).to(device)

transformer_model = SentimentTransformer(
    vocab_size = len(vocab),
    pad_idx=vocab.stoi["<PAD>"],
    embed_dim = transformer_model_cfg["embed_dim"],
    n_heads = transformer_model_cfg["n_head"],
    ff_dim = transformer_model_cfg["ff_dim"],
    n_layers = transformer_model_cfg["n_layers"],
    output_dim = transformer_model_cfg["output_dim"],
    dropout= transformer_model_cfg["dropout"],
).to(device)

lstm_model.load_state_dict(torch.load(lstm_model_cfg["model_path"], map_location=device))

checkpoint = torch.load(transformer_model_cfg["model_path"], map_location=device)
transformer_model.load_state_dict(checkpoint["model_state_dict"])

lstm_model.eval()
transformer_model.eval()

def predict_sentiment(message, history):
    if not message.strip():
        return "Please type a valid review."

    tokens = Vocabulary.tokenize(message)
    token_ids = vocab.numericalize(tokens)
    tensor_input = (torch.tensor(token_ids,dtype=torch.long).unsqueeze(0).to(device))
    attention_mask = (tensor_input != vocab.stoi["<PAD>"]).bool().to(device)

    with torch.no_grad():
        lstm_logit = lstm_model(tensor_input)
        lstm_probability = torch.sigmoid(lstm_logit).item()
        transformer_logit = transformer_model(tensor_input, mask= attention_mask)
        transformer_probability = torch.sigmoid(transformer_logit).item()


    if lstm_probability >= 0.5:
        lstm_sentiment = "Positive"
        lstm_confidence = lstm_probability*100
    else:
        lstm_sentiment = "negative"
        lstm_confidence = (1-lstm_probability)*100

    if transformer_probability >= 0.5:
        transformer_sentiment = "Positive"
        transformer_confidence = transformer_probability*100
    else:
        transformer_sentiment = "negative"
        transformer_confidence = (1-transformer_probability)*100

    return (f"LSTM Model Detected Sentiment: {lstm_sentiment} | Confidence: {lstm_confidence:.2f}%\n"
            f"Transformer Model Detected Sentiment: {transformer_sentiment} | Confidence: {transformer_confidence:.2f}%")

if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=predict_sentiment,
        title="Comparing Architectures: LSTM vs. From-Scratch Transformer",
        description="Type a customer review below to analyze its sentiment.",
        examples=[
            "I absolutely love this product! Highly recommended.",
            "The service was terrible and shipping took forever.",
        ],
    )

    demo.launch(share=True)