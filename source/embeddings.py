import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os


INPUT_PATH = "data/processed/earnings_chunks.csv"
OUTPUT_PATH = "data/processed/earnings_embeddings.npy"


def generate_embeddings():

    print("Loading chunks...")

    df = pd.read_csv(INPUT_PATH)

    texts = df["text"].tolist()

    print(f"Total chunks: {len(texts)}")

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    np.save(
        OUTPUT_PATH,
        embeddings
    )

    print("\nEmbedding generation complete.")
    print("-----------------------------")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_embeddings()