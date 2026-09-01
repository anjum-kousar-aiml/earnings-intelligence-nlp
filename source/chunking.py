import pandas as pd
import os


INPUT_PATH = "data/processed/clean_earnings_calls.csv"
OUTPUT_PATH = "data/processed/earnings_chunks.csv"


def chunk_text(text, chunk_size=1000, overlap=150):
    """
    Split text into overlapping word-based chunks.
    """

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_chunks():

    print("Loading processed dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Transcripts: {len(df)}")

    chunk_records = []

    for _, row in df.iterrows():

        transcript = row["earnings_transcript"]

        chunks = chunk_text(
            transcript,
            chunk_size=1000,
            overlap=150
        )

        for chunk_id, chunk in enumerate(chunks):

            chunk_records.append({
                "symbol": row["symbol"],
                "company_name": row["company_name"],
                "year": row["year"],
                "quarter": row["quarter"],
                "date": row["date"],
                "sector": row["sector"],
                "chunk_id": chunk_id,
                "text": chunk
            })

    chunks_df = pd.DataFrame(chunk_records)

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    chunks_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nChunking complete.")
    print("-----------------")
    print(f"Total chunks: {len(chunks_df)}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nExample chunk:")
    print(chunks_df.iloc[0]["text"][:1000])


if __name__ == "__main__":
    create_chunks()