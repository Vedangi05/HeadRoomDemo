import os
import json
import time
from dotenv import load_dotenv

import streamlit as st
from google import genai

import headroom

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# A bulky, structured tool output (JSON log records) — the kind of context
# Headroom compresses out of the box via its SmartCrusher. Repeated fields
# (region, build, notice) and routine INFO lines are the redundancy it removes.
_SAMPLE_RECORDS = [
    {
        "id": i,
        "ts": f"2026-06-25T09:{i:02d}:00Z",
        "service": "checkout-api",
        "region": "us-east-1",
        "build": "4821-prod",
        "level": "ERROR" if i in (7, 18, 23) else "INFO",
        "message": (
            "payment gateway returned error code 502 - TLS certificate expired"
            if i in (7, 18, 23)
            else "request received for /checkout"
        ),
        "trace_id": f"abc{i:04d}",
        "notice": "Confidential - Copyright 2026 ExampleCorp. All rights reserved.",
    }
    for i in range(40)
]
SAMPLE_CONTEXT = json.dumps(_SAMPLE_RECORDS, indent=2)


def estimate_tokens(text):
    return max(1, len(text) // 4)
    

def headroom_compress_context(context, question, profile="agent-90"):
    """
    Compress the context with the REAL headroom-ai package.

    Headroom protects user messages and compresses tool/document content, so
    we place the context in a tool message and ask the question around it.
    Returns (compressed_context, headroom_result).
    """
    config = headroom.CompressConfig(savings_profile=profile)

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": "Let me look that up in the data."},
        {"role": "tool", "content": context},
        {"role": "user", "content": question},
    ]

    result = headroom.compress(
        messages,
        model="gemini-2.5-flash",
        model_limit=1_000_000,
        config=config,
    )

    compressed_context = context
    for message in result.messages:
        if message.get("role") == "tool":
            compressed_context = str(message.get("content", context))
            break

    return compressed_context, result


def ask_gemini(question, context):
    prompt = f"""
You are a helpful assistant.

Use the context below if it is relevant to the question.
If the context does not contain the answer, answer using your own knowledge.

Context:
{context}

Question:
{question}
"""

    start_time = time.time()

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    latency = round(time.time() - start_time, 2)
    answer = response.text

    # Real token counts reported by the Gemini API.
    usage = getattr(response, "usage_metadata", None)
    tokens = {
        "prompt": getattr(usage, "prompt_token_count", None),
        "output": getattr(usage, "candidates_token_count", None),
        "total": getattr(usage, "total_token_count", None),
    }

    # Fall back to an estimate if the API did not return usage metadata.
    if tokens["prompt"] is None:
        tokens["prompt"] = estimate_tokens(prompt)
    if tokens["total"] is None:
        tokens["total"] = tokens["prompt"] + estimate_tokens(answer or "")

    return answer, latency, tokens


st.set_page_config(
    page_title="With vs Without Headroom",
    layout="wide"
)

st.title("LLM Answer: With vs Without Headroom")
st.write(
    "Paste context, ask a question. The app compresses the context with the real "
    "**headroom-ai** package, then calls Gemini twice — once with the full context "
    "and once with the Headroom-compressed context — and compares the real tokens used."
)

question = st.text_input(
    "Ask a question",
    value="Which requests failed and what was the error?"
)

profile = st.selectbox(
    "Headroom savings profile",
    options=["agent-90", "balanced"],
    index=0,
    help="agent-90 targets ~90% savings (more aggressive); balanced targets ~70%.",
)

context = st.text_area(
    "Context (the text Headroom compresses before the LLM call)",
    value=SAMPLE_CONTEXT,
    height=300,
)

if st.button("Run Comparison"):
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY missing. Add it in your .env file.")
        st.stop()

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    full_context = context

    with st.spinner("Compressing context with Headroom..."):
        compressed_context, hr_result = headroom_compress_context(
            full_context, question, profile
        )

    # Headroom's own reported compression of the context block.
    st.subheader("Headroom Context Compression")
    hc1, hc2, hc3 = st.columns(3)
    hc1.metric("Context tokens before", hr_result.tokens_before)
    hc2.metric("Context tokens after", hr_result.tokens_after)
    hc3.metric("Headroom reduction", f"{hr_result.compression_ratio * 100:.1f}%")
    st.caption(f"Transforms applied: {hr_result.transforms_applied}")

    if hr_result.tokens_saved == 0:
        st.info(
            "Headroom made no change to this content. Structured content "
            "(JSON, logs, code) compresses out of the box; plain prose needs the "
            "Kompress model from the optional `[ml]` extra. Try the sample JSON, "
            "or paste structured tool output."
        )

    st.subheader("Context Comparison")
    left, right = st.columns(2)
    with left:
        st.markdown("### Without Headroom (full context)")
        st.text_area("Full context", full_context, height=300)
    with right:
        st.markdown("### With Headroom (compressed context)")
        st.text_area("Compressed context", compressed_context, height=300)

    st.subheader("Gemini Answer Comparison")

    with st.spinner("Calling Gemini without Headroom..."):
        normal_answer, normal_latency, normal_tokens = ask_gemini(
            question, full_context
        )

    with st.spinner("Calling Gemini with Headroom..."):
        compressed_answer, compressed_latency, compressed_tokens = ask_gemini(
            question, compressed_context
        )

    # Token comparison based on real prompt tokens sent to the LLM.
    normal_prompt = normal_tokens["prompt"]
    compressed_prompt = compressed_tokens["prompt"]
    reduction = round((1 - compressed_prompt / normal_prompt) * 100, 2)

    st.subheader("Gemini Prompt Tokens (real usage, end to end)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Without Headroom — prompt tokens", normal_prompt)
    col2.metric("With Headroom — prompt tokens", compressed_prompt)
    col3.metric("Prompt token reduction", f"{reduction}%")

    st.markdown("**Full token breakdown**")
    st.table({
        "": ["Prompt tokens", "Output tokens", "Total tokens"],
        "Without Headroom": [
            normal_tokens["prompt"],
            normal_tokens["output"],
            normal_tokens["total"],
        ],
        "With Headroom": [
            compressed_tokens["prompt"],
            compressed_tokens["output"],
            compressed_tokens["total"],
        ],
    })

    ans1, ans2 = st.columns(2)
    with ans1:
        st.markdown("### Without Headroom — Answer")
        st.write(normal_answer)
        st.caption(f"Latency: {normal_latency} seconds")
    with ans2:
        st.markdown("### With Headroom — Answer")
        st.write(compressed_answer)
        st.caption(f"Latency: {compressed_latency} seconds")

    st.success(
        f"Without Headroom: {normal_prompt} prompt tokens. "
        f"With Headroom: {compressed_prompt} prompt tokens. "
        f"Reduction: {reduction}%."
    )
