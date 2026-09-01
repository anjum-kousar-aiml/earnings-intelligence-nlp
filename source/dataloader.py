import pandas as pd
import pyarrow.parquet as pq
import os


# ============================================================
# PATHS
# ============================================================

PARQUET_PATH = (
    r"D:\\nlp_sec_project\\source\\episodes.parquet"
)

OUTPUT_PATH = (
    r"D:\\nlp_sec_project\\data\\raw\\earnings_calls.csv"
)


# ============================================================
# COLUMNS NEEDED FOR OUR NLP PROJECT
# ============================================================

COLUMNS = [
    "symbol",
    "company_name",
    "year",
    "quarter",
    "date",
    "sector",
    "earnings_transcript"
]


# ============================================================
# SELECTED COMPANIES
# ============================================================

SELECTED_COMPANIES = [
    "AAPL",
    "AMZN",
    "MSFT",
    "GOOGL",
    "META",
    "NVDA",
    "AMD",
    "INTC",
    "IBM",
    "ORCL",
    "CSCO",
    "ADBE",
    "CRM",
    "JPM",
    "BAC",
    "WMT",
    "COST",
    "KO",
    "PEP",
    "JNJ"
]


def collect_earnings_calls():

    print("Starting local dataset processing...")
    print("-----------------------------------")

    # Check that the file exists
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(
            f"Parquet file not found:\n{PARQUET_PATH}"
        )

    print(f"Source file:")
    print(PARQUET_PATH)

    # Open Parquet without loading everything into RAM
    parquet_file = pq.ParquetFile(
        PARQUET_PATH
    )

    print(
        f"\nTotal rows in source: "
        f"{parquet_file.metadata.num_rows}"
    )

    print(
        f"Total row groups: "
        f"{parquet_file.num_row_groups}"
    )

    print(
        f"\nSelected companies: "
        f"{len(SELECTED_COMPANIES)}"
    )

    # Create output directory
    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    # Remove previous output if it exists
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    # Keep track of company-quarter combinations
    seen_episodes = set()

    total_collected = 0
    total_skipped = 0

    header_written = False

    # ========================================================
    # PROCESS ROW GROUPS ONE AT A TIME
    # ========================================================

    for group_number in range(
        parquet_file.num_row_groups
    ):

        print(
            f"\nProcessing row group "
            f"{group_number + 1}/"
            f"{parquet_file.num_row_groups}..."
        )

        table = parquet_file.read_row_group(
            group_number,
            columns=COLUMNS
        )

        df = table.to_pandas()

        # ----------------------------------------------------
        # Keep only selected companies
        # ----------------------------------------------------

        df = df[
            df["symbol"].isin(
                SELECTED_COMPANIES
            )
        ]

        if len(df) == 0:
            continue

        # ----------------------------------------------------
        # Remove missing transcripts
        # ----------------------------------------------------

        df = df.dropna(
            subset=["earnings_transcript"]
        )

        # ----------------------------------------------------
        # Remove extremely short transcripts
        # ----------------------------------------------------

        df["earnings_transcript"] = (
            df["earnings_transcript"]
            .astype(str)
        )

        df = df[
            df["earnings_transcript"]
            .str.len() >= 1000
        ]

        # ----------------------------------------------------
        # Remove duplicate company-quarter records
        # ----------------------------------------------------

        keep_rows = []

        for index, row in df.iterrows():

            key = (
                row["symbol"],
                int(row["year"]),
                int(row["quarter"])
            )

            if key not in seen_episodes:

                seen_episodes.add(key)
                keep_rows.append(index)

        df = df.loc[keep_rows]

        # ----------------------------------------------------
        # Save this batch immediately
        # ----------------------------------------------------

        if len(df) > 0:

            df.to_csv(
                OUTPUT_PATH,
                mode="a",
                header=not header_written,
                index=False
            )

            header_written = True

            total_collected += len(df)

            print(
                f"  Collected: "
                f"{len(df)}"
            )

            print(
                f"  Total so far: "
                f"{total_collected}"
            )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if total_collected == 0:

        raise ValueError(
            "No matching earnings calls were found."
        )

    final_df = pd.read_csv(
        OUTPUT_PATH
    )

    final_df = final_df.sort_values(
        by=[
            "symbol",
            "year",
            "quarter"
        ]
    ).reset_index(drop=True)

    final_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 55)
    print("DATASET COLLECTION COMPLETE")
    print("=" * 55)

    print(
        f"Records: "
        f"{len(final_df)}"
    )

    print(
        f"Companies: "
        f"{final_df['symbol'].nunique()}"
    )

    print(
        f"Years: "
        f"{final_df['year'].min()} - "
        f"{final_df['year'].max()}"
    )

    print("\nRecords per company:")
    print(
        final_df["symbol"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    collect_earnings_calls()