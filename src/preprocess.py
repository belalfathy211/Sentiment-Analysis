import pandas as pd
import os
import torch
from vocabulary_class import Vocabulary


def run_preprocess(raw_path, processed_dir, min_freq=5):
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)

    print("[1/5] Loading IMDB Dataset...")
    df = pd.read_csv(raw_path)

    print("[2/5] Tokenizing reviews (Cleaning HTML & Symbols)...")
    tokenized_data = [Vocabulary.tokenize(review) for review in df["review"]]

    print(f"[3/5] Building Vocabulary (min_freq={min_freq})...")
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build_vocabulary(tokenized_data)
    print(f"Success: Vocabulary size is {len(vocab)}")

    print("[4/5] Converting tokens to numerical IDs...")
    numerical_tokens = [vocab.numericalize(tokens) for tokens in tokenized_data]
    labels = [1 if sentiment == "positive" else 0 for sentiment in df["sentiment"]]

    print("[5/5] Saving processed data and vocabulary...")
    vocab.save(os.path.join(processed_dir, "vocab.json"))
    processed_data = {"tokens": numerical_tokens, "labels": labels}
    torch.save(processed_data, os.path.join(processed_dir, "processed_data.pt"))

    print("\n" + "=" * 30)
    print("Pipeline Complete!")
    print(f"1. Vocabulary: {processed_dir}/vocab.json")
    print(f"2. Processed Data: {processed_dir}/processed_data.pt")
    print("=" * 30)


if __name__ == "__main__":
    raw_path = "/home/belal/projects/Sentiment-Analysis/data/raw/IMDB Dataset.csv"
    processed_dir = "/home/belal/projects/Sentiment-Analysis/data/processed"
    run_preprocess(raw_path, processed_dir)
