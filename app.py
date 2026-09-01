import streamlit as st
import os

from dotenv import load_dotenv
from groq import Groq

from source.retrieval import load_resources, semantic_search


# ==========================================================
# CONFIGURATION
# ==========================================================

load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"

TOP_K = 4

MAX_CHARS_PER_CHUNK = 1200


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Earnings Intelligence",
    page_icon="📊",
    layout="wide"
)


# ==========================================================
# LOAD RESOURCES
# ==========================================================

@st.cache_resource
def load_system():

    index, chunks_df, embeddings, model = (
        load_resources()
    )

    return index, chunks_df, embeddings, model


# ==========================================================
# GROQ ANALYSIS
# ==========================================================

def generate_insights(
    query,
    results,
    company,
    year,
    quarter
):

    if not results:

        return (
            "Insufficient evidence was found "
            "for this query."
        )


    context = []

    for i, result in enumerate(
        results,
        start=1
    ):

        context.append(
            f"""
SOURCE {i}
Company: {result['company_name']}
Symbol: {result['symbol']}
Period: Q{quarter} {year}
Chunk: {result['chunk_id']}

TEXT:
{result['text'][:MAX_CHARS_PER_CHUNK]}
"""
        )


    context = "\n".join(context)


    prompt = f"""
You are a financial earnings-call analyst.

Answer the user's question using ONLY the
supplied earnings-call evidence.

QUESTION:
{query}

COMPANY:
{company}

PERIOD:
Q{quarter} {year}

EVIDENCE:
{context}

RULES:
- Do not invent facts or numbers.
- Base conclusions only on the evidence.
- Explain important factors clearly.
- Distinguish positive and negative factors.
- Distinguish actual results from management outlook.
- If evidence is insufficient, say so.
- Cite important claims as Source X, Chunk Y.

Return exactly:

SUMMARY:
Brief answer.

KEY FACTORS:
- Important factors.

POSITIVE IMPACT:
- Positive factors supported by evidence.

NEGATIVE / LIMITING FACTORS:
- Negative factors supported by evidence.

MANAGEMENT OUTLOOK:
- Relevant management expectations.

KEY TAKEAWAY:
One concise conclusion.

EVIDENCE:
- Source X, Chunk Y — supporting point.
"""


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
                    "Ground every answer in the supplied "
                    "earnings-call evidence."
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
# COMPARISON ANALYSIS
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
    # Period 1
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
    # Period 2
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
- Focus on meaningful differences.
- Explain why a change occurred only when supported.
- Distinguish facts from management outlook.
- If evidence is insufficient, say so.
- Cite important claims as Source X, Chunk Y.

Return:

OVERALL CHANGE:
Brief comparison.

IMPROVED:
- What improved.

DETERIORATED:
- What worsened.

NEW FACTORS:
- Important factors appearing in Period 2.

PERSISTENT FACTORS:
- Factors present in both periods.

MANAGEMENT OUTLOOK:
- How the outlook changed.

KEY TAKEAWAY:
One concise conclusion.

EVIDENCE:

PERIOD 1:
- Source X, Chunk Y — supporting point.

PERIOD 2:
- Source X, Chunk Y — supporting point.
"""


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
                    "Ground every comparison in the supplied "
                    "earnings-call evidence."
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
# APP
# ==========================================================

st.title("📊 Earnings Intelligence System")

st.markdown(
    """
**NLP-powered analysis of company earnings calls**

Ask questions about a company's earnings call or
compare how business factors changed across two periods.
"""
)


# ==========================================================
# LOAD DATA
# ==========================================================

try:

    index, chunks_df, embeddings, model = (
        load_system()
    )

except Exception as e:

    st.error(
        f"Failed to load retrieval system: {e}"
    )

    st.stop()


companies = sorted(
    chunks_df["symbol"].dropna().unique()
)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("Analysis")

mode = st.sidebar.radio(
    "Analysis mode",
    [
        "Single Period",
        "Compare Periods"
    ]
)


# ==========================================================
# SINGLE PERIOD
# ==========================================================

if mode == "Single Period":

    st.subheader(
        "Single Earnings-Call Analysis"
    )


    company = st.selectbox(
        "Company",
        companies
    )


    col1, col2 = st.columns(2)


    with col1:

        year = st.number_input(
            "Year",
            min_value=2005,
            max_value=2025,
            value=2021
        )


    with col2:

        quarter = st.selectbox(
            "Quarter",
            [1, 2, 3, 4]
        )


    query = st.text_input(
        "What would you like to know?",
        placeholder=(
            "Example: What factors affected revenue?"
        )
    )


    if st.button(
        "Generate Insights",
        type="primary"
    ):

        if not query:

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Retrieving relevant evidence..."
            ):

                results = semantic_search(
                    query=query,
                    chunks_df=chunks_df,
                    embeddings=embeddings,
                    model=model,
                    top_k=TOP_K,
                    company=company,
                    year=year,
                    quarter=quarter
                )


            if not results:

                st.warning(
                    "No relevant evidence was found."
                )

            else:

                st.success(
                    f"Retrieved {len(results)} relevant sections."
                )


                with st.spinner(
                    "Generating insights..."
                ):

                    answer = generate_insights(
                        query=query,
                        results=results,
                        company=company,
                        year=year,
                        quarter=quarter
                    )


                st.markdown("---")

                st.markdown(
                    answer
                )


# ==========================================================
# COMPARISON
# ==========================================================

else:

    st.subheader(
        "Comparative Earnings Analysis"
    )


    company = st.selectbox(
        "Company",
        companies
    )


    st.markdown(
        "### Period 1"
    )


    col1, col2 = st.columns(2)


    with col1:

        year_1 = st.number_input(
            "Year",
            min_value=2005,
            max_value=2025,
            value=2019,
            key="year1"
        )


    with col2:

        quarter_1 = st.selectbox(
            "Quarter",
            [1, 2, 3, 4],
            key="quarter1"
        )


    st.markdown(
        "### Period 2"
    )


    col3, col4 = st.columns(2)


    with col3:

        year_2 = st.number_input(
            "Year",
            min_value=2005,
            max_value=2025,
            value=2021,
            key="year2"
        )


    with col4:

        quarter_2 = st.selectbox(
            "Quarter",
            [1, 2, 3, 4],
            key="quarter2"
        )


    query = st.text_input(
        "What would you like to compare?",
        placeholder=(
            "Example: How did the challenges change?"
        ),
        key="comparison_query"
    )


    if st.button(
        "Compare Periods",
        type="primary"
    ):

        if not query:

            st.warning(
                "Please enter a comparison question."
            )

        else:

            period_1 = (
                f"{company} Q{quarter_1} {year_1}"
            )

            period_2 = (
                f"{company} Q{quarter_2} {year_2}"
            )


            with st.spinner(
                "Retrieving evidence from both periods..."
            ):

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


            if not results_q1 or not results_q2:

                st.warning(
                    "Could not retrieve sufficient evidence "
                    "for both periods."
                )

            else:

                st.success(
                    f"Retrieved {len(results_q1)} sections "
                    f"from {period_1} and "
                    f"{len(results_q2)} sections "
                    f"from {period_2}."
                )


                with st.spinner(
                    "Generating comparative insights..."
                ):

                    answer = generate_comparison(
                        query=query,
                        results_q1=results_q1,
                        results_q2=results_q2,
                        period_1=period_1,
                        period_2=period_2
                    )


                st.markdown("---")

                st.markdown(
                    answer
                )