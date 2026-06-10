import json
import torch
import gradio as gr
from transformers import DistilBertTokenizer
from models.BERT_transformer_model import BERTTransformerSentimentRNN

with open("/home/belal/projects/Sentiment-Analysis/config.json", "r") as f:
    config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = config["BERT_transformer_model_path"]

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

model = BERTTransformerSentimentRNN(dropout=config["BERT_transformer_dropout"]).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def predict_sentiment_bert_transformer(message, history):
    if not message.strip():
        return "Please type a valid review."

    encoding = tokenizer(
        message,
        truncation=True,
        max_length=config["BERT_max_length"],
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logit = model(input_ids, attention_mask)
        probability = torch.sigmoid(logit).item()

    if probability >= 0.5:
        sentiment = "Positive 😊"
        confidence = probability * 100
    else:
        sentiment = "Negative 😠"
        confidence = (1 - probability) * 100

    return f"Transformer Sentiment: {sentiment} | Confidence: {confidence:.2f}%"

if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=predict_sentiment_bert_transformer,
        title="DistilBERT Live Sentiment Assistant",
        description="Type a review to see how the Transformer understands context and sarcasm in real-time.",
        examples=[
            "I expected this movie to be a total waste of time, but to my absolute surprise, the brilliant acting kept me hooked till the very end.",
            "The movie wasn't bad, but it certainly wasn't good either. Just average.",
            "Visually stunning, but the story was incredibly hollow and boring."
        ],
    )

    demo.launch(share=True)