import os
from dotenv import load_dotenv
from groq import Groq

from retrieval import load_resources, semantic_search


load_dotenv()


# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_NAME = "openai/gpt-oss-120b"

# Limit the amount of transcript evidence sent to Groq
MAX_CHARS_PER_CHUNK = 1200

# Number of chunks retrieved for each period
TOP_K = 4


# ==========================================================
# GENERATE COMPARISON
# ==========================================================

def generate_comparison(
    query,
    results_q1,
    results_q2,
    period_1,
    period_2
):

    if not results_q1 or not results_q2:

        return (
            "Insufficient evidence was found "
            "for one or both periods."
        )


    # --------------------------------------------------
    # Build evidence for Period 1
    # --------------------------------------------------

    context_q1 = []

    for i, result in enumerate(
        results_q1,
        start=1
    ):

        context_q1.append(
            f"""
SOURCE {i}
Company: {result['company_name']}
Symbol: {result['symbol']}
Period: {period_1}
Chunk: {result['chunk_id']}

TEXT:
{result['text'][:MAX_CHARS_PER_CHUNK]}
"""
        )

    context_q1 = "\n".join(context_q1)


    # --------------------------------------------------
    # Build evidence for Period 2
    # --------------------------------------------------

    context_q2 = []

    for i, result in enumerate(
        results_q2,
        start=1
    ):

        context_q2.append(
            f"""
SOURCE {i}
Company: {result['company_name']}
Symbol: {result['symbol']}
Period: {period_2}
Chunk: {result['chunk_id']}

TEXT:
{result['text'][:MAX_CHARS_PER_CHUNK]}
"""
        )

    context_q2 = "\n".join(context_q2)


    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------

    prompt = f"""
You are a financial earnings-call analyst.

Compare {period_1} and {period_2} using ONLY
the supplied transcript evidence.

QUESTION:
{query}

PERIOD 1:
{period_1}

EVIDENCE:
{context_q1}

PERIOD 2:
{period_2}

EVIDENCE:
{context_q2}

RULES:
- Use only the supplied evidence.
- Do not invent facts or numbers.
- Clearly distinguish the two periods.
- Focus on meaningful changes.
- Explain why a change occurred only when supported
  by the evidence.
- Distinguish actual results from management outlook.
- If evidence is insufficient, say so.
- Cite important claims using Source X, Chunk Y.

Return exactly this structure:

OVERALL CHANGE:
Brief comparison of the most important change.

IMPROVED:
- What improved from Period 1 to Period 2.

DETERIORATED:
- What worsened from Period 1 to Period 2.

NEW FACTORS:
- Important factors appearing in Period 2.

PERSISTENT FACTORS:
- Factors present in both periods.

MANAGEMENT OUTLOOK:
- How management's outlook changed.

KEY TAKEAWAY:
One concise analytical conclusion.

EVIDENCE:

PERIOD 1:
- Source X, Chunk Y — supporting point.

PERIOD 2:
- Source X, Chunk Y — supporting point.
"""


    # --------------------------------------------------
    # Groq
    # --------------------------------------------------

    client = Groq(
        api_key=os.environ.get(
            "GROQ_API_KEY"
        )
    )


    response = client.chat.completions.create(
        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful financial analyst. "
                    "Ground every conclusion in the "
                    "supplied earnings-call evidence."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2,

        max_completion_tokens=1200
    )


    return response.choices[0].message.content


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    # --------------------------------------------------
    # Load retrieval system
    # --------------------------------------------------

    print("Loading retrieval system...")

    index, chunks_df, embeddings, model = (
        load_resources()
    )

    print("Retrieval system ready.")


    # --------------------------------------------------
    # User input
    # --------------------------------------------------

    company = input(
        "\nEnter company symbol: "
    ).strip().upper()


    year_1 = int(
        input(
            "Enter first year: "
        )
    )


    quarter_1 = int(
        input(
            "Enter first quarter (1-4): "
        )
    )


    year_2 = int(
        input(
            "Enter second year: "
        )
    )


    quarter_2 = int(
        input(
            "Enter second quarter (1-4): "
        )
    )


    query = input(
        "What would you like to compare? "
    ).strip()


    # --------------------------------------------------
    # Create period labels
    # --------------------------------------------------

    period_1 = (
        f"{company} Q{quarter_1} {year_1}"
    )

    period_2 = (
        f"{company} Q{quarter_2} {year_2}"
    )


    # --------------------------------------------------
    # Retrieve Period 1
    # --------------------------------------------------

    print(
        f"\nRetrieving evidence for {period_1}..."
    )


    results_q1 = semantic_search(
        query=query,
        chunks_df=chunks_df,
        embeddings=embeddings,
        model=model,

        top_k=TOP_K,

        company=company,
        year=year_1,
        quarter=quarter_1
    )


    # --------------------------------------------------
    # Retrieve Period 2
    # --------------------------------------------------

    print(
        f"Retrieving evidence for {period_2}..."
    )


    results_q2 = semantic_search(
        query=query,
        chunks_df=chunks_df,
        embeddings=embeddings,
        model=model,

        top_k=TOP_K,

        company=company,
        year=year_2,
        quarter=quarter_2
    )


    # --------------------------------------------------
    # Display retrieval results
    # --------------------------------------------------

    print(
        f"\nRetrieved {len(results_q1)} chunks "
        f"from {period_1}."
    )


    print(
        f"Retrieved {len(results_q2)} chunks "
        f"from {period_2}."
    )


    # --------------------------------------------------
    # Check evidence
    # --------------------------------------------------

    if not results_q1 or not results_q2:

        print(
            "\nCould not retrieve sufficient "
            "evidence for both periods."
        )

        exit()


    # --------------------------------------------------
    # Generate comparison
    # --------------------------------------------------

    print(
        "\nGenerating comparative insights with Groq..."
    )


    answer = generate_comparison(
        query=query,

        results_q1=results_q1,
        results_q2=results_q2,

        period_1=period_1,
        period_2=period_2
    )


    # --------------------------------------------------
    # Display final result
    # --------------------------------------------------

    print("\n")

    print("=" * 70)

    print(
        "COMPARATIVE EARNINGS ANALYSIS"
    )

    print("=" * 70)

    print(answer)

    print("=" * 70)