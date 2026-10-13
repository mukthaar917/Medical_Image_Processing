from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import hmac

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from inference import ChestXRayAnalyzer
from rag import SimpleRAG
from llm import generate_with_ollama, fallback_summary


st.set_page_config(
    page_title="AI-Assisted Chest X-ray Review",
    page_icon="🩻",
    layout="wide",
)


def check_login() -> bool:
    if st.session_state.get("authenticated", False):
        return True

    st.title("🔐 AI-Assisted Chest X-ray Review")
    st.caption("Authorized demonstration access")

    try:
        expected_username = str(st.secrets["auth"]["username"])
        expected_password = str(st.secrets["auth"]["password"])
    except Exception:
        st.error(
            "Login credentials are not configured. "
            "Create .streamlit/secrets.toml with [auth] username and password."
        )
        st.stop()

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login", type="primary"):
        username_ok = hmac.compare_digest(username, expected_username)
        password_ok = hmac.compare_digest(password, expected_password)

        if username_ok and password_ok:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid username or password.")

    return False


if not check_login():
    st.stop()


@st.cache_resource
def load_analyzer():
    return ChestXRayAnalyzer()


@st.cache_resource
def load_rag():
    return SimpleRAG("knowledge_base.md")


def display_uploaded_image(file_name: str, data: bytes):
    suffix = Path(file_name).suffix.lower()

    if suffix == ".dcm":
        try:
            import pydicom

            ds = pydicom.dcmread(BytesIO(data))
            arr = ds.pixel_array.astype(np.float32)

            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                arr = arr.max() - arr

            lo, hi = np.percentile(arr, (1, 99))
            arr = np.clip(arr, lo, hi)
            arr = (arr - lo) / (hi - lo + 1e-8)

            st.image(
                arr,
                caption="Uploaded DICOM preview",
                clamp=True,
            )

        except Exception as exc:
            st.info(f"DICOM preview unavailable: {exc}")

    else:
        image = Image.open(BytesIO(data)).convert("L")
        st.image(
            image,
            caption="Uploaded chest X-ray",
            use_container_width=True,
        )


def make_radiologist_report(
    file_name: str,
    predictions: dict[str, float],
    rag_background_summary: str,
    findings: str,
    impression: str,
    final_diagnosis: str,
    reviewer: str,
) -> str:
    ranked = sorted(
        predictions.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    scores = "\n".join(
        f"- {name}: {score:.3f}"
        for name, score in ranked
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""FINAL RADIOLOGIST-REVIEWED DEMO REPORT

Generated: {now}
Study: Chest X-ray
File: {file_name}
Reviewed by: {reviewer or "Not specified"}

RAW MODEL OUTPUTS
These values are produced directly by TorchXRayVision and are rendered by the
application without LLM rewriting. They are not clinical probabilities, disease
severity measures, or confirmed findings.

{scores}

RAG BACKGROUND SUMMARY
The following section summarizes retrieved reference material only. It does not
interpret the uploaded radiograph.

{rag_background_summary}

RADIOLOGIST FINDINGS
{findings}

RADIOLOGIST IMPRESSION
{impression}

FINAL DIAGNOSIS / CLINICAL CONCLUSION
{final_diagnosis}

STATUS
Radiologist-reviewed / demo workflow.

NOTE
The AI components provide model outputs and background information only.
The final diagnosis / clinical conclusion above is radiologist-entered and approved.
"""


st.title("🩻 AI-Assisted Chest X-ray Review Dashboard")
st.caption(
    "TorchXRayVision DenseNet121 → raw model scores → RAG background → "
    "LLM background summary only → radiologist review."
)

st.info(
    "Important separation of responsibilities: TorchXRayVision produces the raw "
    "scores. Python displays those scores exactly. Ollama receives only retrieved "
    "RAG text and does not receive or rewrite model scores."
)

with st.sidebar:
    st.header("Demo settings")

    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

    top_n_display = st.slider(
        "Number of model scores to highlight",
        min_value=3,
        max_value=18,
        value=10,
    )

    rag_findings = st.slider(
        "Top model labels used to form RAG query",
        min_value=1,
        max_value=6,
        value=3,
    )

    top_k = st.slider(
        "RAG chunks returned",
        min_value=1,
        max_value=5,
        value=3,
    )

    use_llm = st.toggle(
        "Use Ollama for RAG background summary",
        value=True,
    )

    ollama_model = st.text_input(
        "Ollama model",
        value="llama3.2:1b",
    )

    st.warning(
        "No generic 0.5 diagnostic threshold is used. "
        "Model scores are displayed as raw outputs only."
    )


uploaded = st.file_uploader(
    "Upload a chest X-ray",
    type=["png", "jpg", "jpeg", "dcm"],
    help="Use a de-identified chest X-ray for this demonstration.",
)

if uploaded is None:
    st.info("Upload a chest X-ray to begin.")
    st.stop()

image_bytes = uploaded.getvalue()

left, right = st.columns(
    [1, 1.35],
    gap="large",
)

with left:
    st.subheader("Input image")
    display_uploaded_image(
        uploaded.name,
        image_bytes,
    )

with st.spinner("Running TorchXRayVision model..."):
    analyzer = load_analyzer()
    predictions = analyzer.predict_bytes(
        image_bytes,
        uploaded.name,
    )

ranked = sorted(
    predictions.items(),
    key=lambda x: x[1],
    reverse=True,
)

all_scores_df = pd.DataFrame(
    [
        {
            "Rank": i + 1,
            "Pathology label": name,
            "Raw model score": score,
        }
        for i, (name, score) in enumerate(ranked)
    ]
)

highlight_df = all_scores_df.head(
    top_n_display
).copy()

with right:
    st.subheader("Raw model outputs")
    st.caption(
        "Rendered directly from TorchXRayVision output by Python. "
        "The LLM does not generate, rename, or interpret these values."
    )

    st.dataframe(
        highlight_df.style.format(
            {"Raw model score": "{:.3f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    chart_df = (
        highlight_df[
            ["Pathology label", "Raw model score"]
        ]
        .set_index("Pathology label")
    )

    st.bar_chart(chart_df)

with st.expander("View all 18 raw model outputs"):
    st.dataframe(
        all_scores_df.style.format(
            {"Raw model score": "{:.3f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

st.download_button(
    "Download all raw model scores (.csv)",
    data=all_scores_df.to_csv(index=False),
    file_name="chest_xray_raw_model_scores.csv",
    mime="text/csv",
)

# The model labels are used only to formulate the retrieval query in Python.
# Raw scores are NOT sent to the LLM.
rag_labels = [
    name
    for name, _ in ranked[:rag_findings]
]

query = (
    "Clinical background for these exact chest X-ray pathology labels: "
    + ", ".join(rag_labels)
)

with st.spinner("Retrieving clinical background..."):
    rag = load_rag()
    retrieved = rag.retrieve(
        query,
        top_k=top_k,
    )

st.subheader("RAG clinical background")
st.caption(
    "Retrieved material provides background only. "
    "It is not used to classify the image."
)

for chunk in retrieved:
    with st.expander(
        f"{chunk.title} · retrieval similarity {chunk.score:.3f}"
    ):
        st.write(chunk.text)

st.subheader("LLM summary of RAG background only")
st.caption(
    "Ollama receives the retrieved reference text only. "
    "It does not receive model scores and is not asked to diagnose the image."
)

if use_llm:
    try:
        with st.spinner("Summarizing retrieved background..."):
            rag_background_summary = generate_with_ollama(
                retrieved,
                model=ollama_model,
            )

    except Exception as exc:
        st.warning(
            "Ollama background summary was unavailable; "
            "showing deterministic output instead. "
            f"Details: {exc}"
        )

        rag_background_summary = fallback_summary(
            retrieved
        )
else:
    rag_background_summary = fallback_summary(
        retrieved
    )

st.markdown(rag_background_summary)

st.divider()
st.subheader("Radiologist review")

st.info(
    "The AI does not create the final diagnosis. "
    "A final report becomes available only after radiologist-entered review."
)

reviewer = st.text_input(
    "Radiologist / reviewer name"
)

radiologist_findings = st.text_area(
    "Radiologist findings",
    height=150,
    placeholder=(
        "Enter image findings after reviewing "
        "the original radiograph..."
    ),
)

radiologist_impression = st.text_area(
    "Radiologist impression",
    height=120,
    placeholder="Enter the radiologist impression...",
)

final_diagnosis = st.text_area(
    "Final diagnosis / clinical conclusion",
    height=100,
    placeholder=(
        "Enter the final clinician-approved conclusion..."
    ),
)

approved = st.checkbox(
    "I have reviewed the original image and approve "
    "the final report content."
)

if approved:
    if (
        not radiologist_impression.strip()
        and not final_diagnosis.strip()
    ):
        st.warning(
            "Enter a radiologist impression or final clinical "
            "conclusion before producing the final report."
        )

    else:
        final_report = make_radiologist_report(
            uploaded.name,
            predictions,
            rag_background_summary,
            radiologist_findings,
            radiologist_impression,
            final_diagnosis,
            reviewer,
        )

        st.success(
            "Radiologist-reviewed report is ready."
        )

        st.download_button(
            "Download final radiologist-reviewed report (.txt)",
            data=final_report,
            file_name="radiologist_reviewed_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
else:
    st.caption(
        "Final report status: Pending radiologist review."
    )

st.divider()

st.error(
    "Research/demo only. Raw model scores are not diagnoses or calibrated "
    "clinical probabilities. Clinical interpretation requires appropriate "
    "validation and radiologist review."
)
