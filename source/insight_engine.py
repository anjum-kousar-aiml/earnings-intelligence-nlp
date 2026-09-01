import os
from dotenv import load_dotenv
from groq import Groq

from retrieval import load_resources, semantic_search


load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"


def generate_insights(query, results):

    if not results:
        return "No relevant information was found."


    # --------------------------------------------------
    # Build evidence context
    # --------------------------------------------------

    context_parts = []

    for i, result in enumerate(results, start=1):

        context_parts.append(
            f"""
SOURCE {i}
Company: {result['company_name']}
Symbol: {result['symbol']}
Quarter: Q{int(result['quarter'])} {int(result['year'])}
Chunk: {int(result['chunk_id'])}

TEXT:
{result['text']}
"""
        )

    context = "\n".join(context_parts)


    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------

    prompt = f"""
You are a financial earnings-call analyst.

Analyze the earnings-call evidence provided below and
answer the user's question.

IMPORTANT RULES:

1. Use ONLY the provided transcript evidence.
2. Do not invent facts, numbers, causes, or conclusions.
3. If the evidence does not support a claim, say so.
4. Distinguish revenue factors from profitability factors.
5. Distinguish actual results from management expectations.
6. Be concise but analytical.
7. Mention the evidence source for important claims.

USER QUESTION:
{query}


EARNINGS-CALL EVIDENCE:
{context}


Determine what type of analysis the question requires.

For example:
- revenue → revenue drivers and headwinds
- profit → profitability drivers, costs and margins
- growth → growth drivers and constraints
- risks → risks and headwinds
- guidance/outlook → management expectations
- strategy → strategic initiatives
- general question → provide the most relevant financial analysis


Return the answer using exactly this structure:

SUMMARY:
2-3 sentences directly answering the question.

KEY FACTORS:
- Factor 1
- Factor 2
- Factor 3
- Factor 4

POSITIVE IMPACT:
- Explain factors that helped the metric or business.
- If none are supported, write "No clear positive factor identified."

NEGATIVE / LIMITING FACTORS:
- Explain factors that negatively affected the metric or business.
- If none are supported, write "No clear negative factor identified."

MANAGEMENT OUTLOOK:
- Mention relevant forward-looking comments.
- If no outlook is present in the evidence, write
  "No clear outlook identified."

EVIDENCE:
- Source X, Chunk Y — brief explanation of what supports the claim.
- Source X, Chunk Y — brief explanation of what supports the claim.
"""


    # --------------------------------------------------
    # Groq
    # --------------------------------------------------

    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY")
    )


    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful financial analyst. "
                    "Ground every conclusion in the supplied "
                    "earnings-call evidence."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=1500
    )


    return response.choices[0].message.content


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("Loading retrieval system...")

    index, chunks_df, embeddings, model = load_resources()

    print("Retrieval system ready.")


    query = input(
        "\nEnter your question: "
    ).strip()


    company = input(
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


    company = (
        company
        if company
        else None
    )


    print(
        "\nRetrieving relevant transcript sections..."
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
        f"Retrieved {len(results)} relevant chunks."
    )


    if not results:

        print(
            "\nNo matching evidence found."
        )

        exit()


    print(
        "\nGenerating insights with Groq..."
    )


    answer = generate_insights(
        query=query,
        results=results
    )


    print("\n")
    print("=" * 70)
    print("EARNINGS CALL INSIGHTS")
    print("=" * 70)

    print(answer)

    print("=" * 70)