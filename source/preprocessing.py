import pandas as pd
import re
import os


INPUT_PATH = "data/raw/earnings_calls.csv"
OUTPUT_PATH = "data/processed/clean_earnings_calls.csv"


def clean_text(text):
    """Clean an earnings call transcript."""

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def preprocess_data():

    print("Loading raw dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Original records: {len(df)}")

    # --------------------------------------------------
    # 1. Remove duplicate company-quarter combinations
    # --------------------------------------------------

    df = df.drop_duplicates(
        subset=["symbol", "year", "quarter"]
    )

    print(f"After removing duplicates: {len(df)}")

    # --------------------------------------------------
    # 2. Remove records with missing transcripts
    # --------------------------------------------------

    df = df.dropna(
        subset=["earnings_transcript"]
    )

    # --------------------------------------------------
    # 3. Clean transcript text
    # --------------------------------------------------

    df["earnings_transcript"] = (
        df["earnings_transcript"]
        .apply(clean_text)
    )

    # --------------------------------------------------
    # 4. Remove very short transcripts
    # --------------------------------------------------

    df["transcript_length"] = (
        df["earnings_transcript"]
        .str.len()
    )

    df = df[
        df["transcript_length"] >= 1000
    ]

    # --------------------------------------------------
    # 5. Sort chronologically
    # --------------------------------------------------

    df = df.sort_values(
        by=["symbol", "year", "quarter"]
    )

    # --------------------------------------------------
    # 6. Reset index
    # --------------------------------------------------

    df = df.reset_index(drop=True)

    # --------------------------------------------------
    # 7. Save processed dataset
    # --------------------------------------------------

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nPreprocessing complete.")
    print("-----------------------")
    print(f"Final records: {len(df)}")
    print(f"Companies: {df['symbol'].nunique()}")
    print(
        f"Years: {df['year'].min()} - "
        f"{df['year'].max()}"
    )
    print(
        f"Average transcript length: "
        f"{df['transcript_length'].mean():.0f} characters"
    )
    print(f"Saved to: {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    preprocess_data()