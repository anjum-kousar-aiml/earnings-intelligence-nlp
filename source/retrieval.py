import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


EMBEDDINGS_PATH = "data/processed/earnings_embeddings.npy"
CHUNKS_PATH = "data/processed/earnings_chunks.csv"
INDEX_PATH = "vector_store/earnings.index"

MODEL_NAME = "all-MiniLM-L6-v2"


def load_resources():

    index = faiss.read_index(INDEX_PATH)

    chunks_df = pd.read_csv(CHUNKS_PATH)

    embeddings = np.load(EMBEDDINGS_PATH)

    model = SentenceTransformer(MODEL_NAME)

    return index, chunks_df, embeddings, model


def semantic_search(
    query,
    chunks_df,
    embeddings,
    model,
    top_k=5,
    company=None,
    year=None,
    quarter=None
):

    # ---------------------------------------------
    # 1. Filter chunks using metadata
    # ---------------------------------------------

    filtered_df = chunks_df.copy()

    if company is not None:
        filtered_df = filtered_df[
            filtered_df["symbol"] == company
        ]

    if year is not None:
        filtered_df = filtered_df[
            filtered_df["year"] == year
        ]

    if quarter is not None:
        filtered_df = filtered_df[
            filtered_df["quarter"] == quarter
        ]

    if len(filtered_df) == 0:
        return []

    # ---------------------------------------------
    # 2. Get original embedding positions
    # ---------------------------------------------

    filtered_indices = filtered_df.index.to_numpy()

    filtered_embeddings = embeddings[
        filtered_indices
    ].copy()

    # ---------------------------------------------
    # 3. Normalize embeddings
    # ---------------------------------------------

    faiss.normalize_L2(filtered_embeddings)

    # ---------------------------------------------
    # 4. Create temporary FAISS index
    # ---------------------------------------------

    dimension = filtered_embeddings.shape[1]

    filtered_index = faiss.IndexFlatIP(
        dimension
    )

    filtered_index.add(
        filtered_embeddings
    )

    # ---------------------------------------------
    # 5. Embed user query
    # ---------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(
        query_embedding
    )

    # ---------------------------------------------
    # 6. Semantic search
    # ---------------------------------------------

    k = min(
        top_k,
        len(filtered_df)
    )

    scores, local_indices = (
        filtered_index.search(
            query_embedding,
            k
        )
    )

    # ---------------------------------------------
    # 7. Convert local indices to original rows
    # ---------------------------------------------

    results = []

    for score, local_idx in zip(
        scores[0],
        local_indices[0]
    ):

        original_idx = filtered_indices[
            local_idx
        ]

        result = chunks_df.iloc[
            original_idx
        ].to_dict()

        result[
            "similarity_score"
        ] = float(score)

        results.append(result)

    return results


if __name__ == "__main__":

    print("Loading retrieval system...")

    index, chunks_df, embeddings, model = (
        load_resources()
    )

    print("Retrieval system ready.")

    print("\nAvailable companies:")
    print(
        sorted(
            chunks_df["symbol"]
            .dropna()
            .unique()
        )[:50]
    )

    query = input(
        "\nEnter your question: "
    )

    company_input = input(
        "Enter company symbol "
        "(press Enter for all): "
    ).strip()

    year_input = input(
        "Enter year "
        "(press Enter for all): "
    ).strip()

    quarter_input = input(
        "Enter quarter 1-4 "
        "(press Enter for all): "
    ).strip()

    company = (
        company_input
        if company_input
        else None
    )

    year = (
        int(year_input)
        if year_input
        else None
    )

    quarter = (
        int(quarter_input)
        if quarter_input
        else None
    )

    results = semantic_search(
        query=query,
        chunks_df=chunks_df,
        embeddings=embeddings,
        model=model,
        top_k=4,
        company=company,
        year=year,
        quarter=quarter
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SEMANTIC SEARCH RESULTS"
    )

    print(
        "=" * 70
    )

    if not results:

        print(
            "\nNo matching results found."
        )

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nResult {i}"
        )

        print(
            "-" * 70
        )

        print(
            f"Company: "
            f"{result['company_name']}"
        )

        print(
            f"Quarter: "
            f"Q{int(result['quarter'])} "
            f"{int(result['year'])}"
        )

        print(
            f"Chunk: "
            f"{int(result['chunk_id'])}"
        )

        print(
            f"Similarity: "
            f"{result['similarity_score']:.4f}"
        )

        print("\nText:")

        print(
            result["text"][:1200]
        )