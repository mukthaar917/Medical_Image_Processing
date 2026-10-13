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
        if (
            hmac.compare_digest(username, expected_username)
            and hmac.compare_digest(password, expected_password)
        ):
            st.session_state["authenticated"] = True
            st.rerun()
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
                caption="Uploaded chest X-ray",
                clamp=True,
                use_container_width=True,
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


def build_score_table(predictions: dict[str, float]) -> pd.DataFrame:
    ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    return pd.DataFrame(
        [
            {
                "Rank": i + 1,
                "Pathology label": pathology,
                "Raw model score": score,
            }
            for i, (pathology, score) in enumerate(ranked)
        ]
    )


def build_final_report(
    file_name: str,
    score_table: pd.DataFrame,
    assessments: dict[str, str],
    comments: dict[str, str],
    rag_summary: str,
    findings: str,
    impression: str,
    final_diagnosis: str,
    reviewer: str,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    condition_lines = []
    for _, row in score_table.iterrows():
        pathology = str(row["Pathology label"])
        score = float(row["Raw model score"])
        assessment = assessments.get(pathology, "Not reviewed")
        note = comments.get(pathology, "").strip()

        line = (
            f"- {pathology}: raw model score {score:.3f}; "
            f"radiologist assessment: {assessment}"
        )
        if note:
            line += f"; note: {note}"
        condition_lines.append(line)

    condition_text = "\n".join(condition_lines)

    return f"""FINAL RADIOLOGIST-REVIEWED CHEST X-RAY DEMO REPORT

Generated: {now}
Study: Chest X-ray
File: {file_name}
Reviewed by: {reviewer or "Not specified"}

IMPORTANT
The raw AI model scores below are not calibrated clinical probabilities and are
not diagnoses. The radiologist assessment column is the clinical review step.

ALL MODEL CONDITIONS
{condition_text}

RAG / LLM BACKGROUND SUMMARY
The LLM received retrieved background material only. It did not receive the
uploaded image or model scores.

{rag_summary}

RADIOLOGIST FINDINGS
{findings}

RADIOLOGIST IMPRESSION
{impression}

FINAL DIAGNOSIS / CLINICAL CONCLUSION
{final_diagnosis}

STATUS
Radiologist-reviewed / demo workflow.

NOTE
The final diagnosis/clinical conclusion above is radiologist-entered and approved.
"""


st.title("🩻 AI-Assisted Chest X-ray Review Dashboard")
st.caption(
    "TorchXRayVision DenseNet121 → all raw chest pathology scores → "
    "RAG background → LLM background summary → radiologist review → final report."
)

with st.sidebar:
    st.header("Demo settings")

    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

    top_k = st.slider("RAG chunks returned", 1, 5, 3)

    rag_labels_count = st.slider(
        "Top model labels used only for RAG retrieval",
        1,
        6,
        3,
    )

    use_llm = st.toggle(
        "Use Ollama for RAG-background summary",
        value=True,
    )

    ollama_model = st.text_input(
        "Ollama model",
        value="llama3.2:1b",
    )

    st.warning(
        "Chest X-ray only. No generic 0.5 diagnostic threshold is used. "
        "Model scores are raw outputs and the LLM does not receive them."
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

left, right = st.columns([1, 1.35], gap="large")

with left:
    st.subheader("Input image")
    display_uploaded_image(uploaded.name, image_bytes)

with st.spinner("Running TorchXRayVision DenseNet121..."):
    analyzer = load_analyzer()
    predictions = analyzer.predict_bytes(image_bytes, uploaded.name)

score_table = build_score_table(predictions)

with right:
    st.subheader("All raw model outputs")
    st.caption(
        "All pathology labels are rendered directly by Python from TorchXRayVision. "
        "The LLM does not generate, rename, or reinterpret them."
    )

    st.dataframe(
        score_table.style.format({"Raw model score": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
        height=620,
    )

st.download_button(
    "Download all raw model scores (.csv)",
    data=score_table.to_csv(index=False),
    file_name="chest_xray_all_raw_model_scores.csv",
    mime="text/csv",
)

st.divider()
st.subheader("RAG clinical background")

rag_labels = (
    score_table["Pathology label"]
    .head(rag_labels_count)
    .tolist()
)

query = (
    "Clinical reference background for chest X-ray terminology: "
    + ", ".join(rag_labels)
)

with st.spinner("Retrieving reference background..."):
    rag = load_rag()
    retrieved = rag.retrieve(query, top_k=top_k)

st.caption(
    "Top model labels are used only as search terms for retrieval. "
    "Retrieval similarity does not indicate disease presence."
)

for chunk in retrieved:
    with st.expander(
        f"{chunk.title} · retrieval similarity {chunk.score:.3f}"
    ):
        st.write(chunk.text)

st.subheader("LLM summary of retrieved background only")
st.caption(
    "Ollama receives retrieved reference text only. "
    "It does not receive the image or model scores."
)

if use_llm:
    try:
        with st.spinner("Summarizing retrieved reference material..."):
            rag_summary = generate_with_ollama(
                retrieved,
                model=ollama_model,
            )
    except Exception as exc:
        st.warning(
            "Ollama was unavailable; showing deterministic RAG summary instead. "
            f"Details: {exc}"
        )
        rag_summary = fallback_summary(retrieved)
else:
    rag_summary = fallback_summary(retrieved)

st.markdown(rag_summary)

st.divider()
st.subheader("Radiologist review of all model conditions")

st.info(
    "Review each model condition against the original radiograph. "
    "The model score itself does not determine whether the condition is present."
)

assessment_options = [
    "Not reviewed",
    "Confirmed",
    "Not confirmed",
    "Indeterminate",
]

assessments: dict[str, str] = {}
comments: dict[str, str] = {}

for _, row in score_table.iterrows():
    pathology = str(row["Pathology label"])
    score = float(row["Raw model score"])

    with st.expander(f"{pathology} · raw model score {score:.3f}"):
        assessments[pathology] = st.selectbox(
            "Radiologist assessment",
            assessment_options,
            key=f"assessment_{pathology}",
        )
        comments[pathology] = st.text_input(
            "Optional comment",
            key=f"comment_{pathology}",
        )

st.divider()
st.subheader("Final radiologist report")

reviewer = st.text_input("Radiologist / reviewer name")

radiologist_findings = st.text_area(
    "Radiologist findings",
    height=160,
    placeholder="Enter the complete chest X-ray findings...",
)

radiologist_impression = st.text_area(
    "Radiologist impression",
    height=120,
    placeholder="Enter the radiologist impression...",
)

final_diagnosis = st.text_area(
    "Final diagnosis / clinical conclusion",
    height=100,
    placeholder="Enter the final clinician-approved diagnosis/conclusion...",
)

approved = st.checkbox(
    "I have reviewed the original image and approve the final report content."
)

if approved:
    if not radiologist_impression.strip() and not final_diagnosis.strip():
        st.warning(
            "Enter a radiologist impression or final diagnosis before "
            "producing the final report."
        )
    else:
        report = build_final_report(
            uploaded.name,
            score_table,
            assessments,
            comments,
            rag_summary,
            radiologist_findings,
            radiologist_impression,
            final_diagnosis,
            reviewer,
        )

        st.success("Radiologist-reviewed final report is ready.")

        st.download_button(
            "Download final diagnosis report (.txt)",
            data=report,
            file_name="final_radiologist_chest_xray_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
else:
    st.caption("Final report status: Pending radiologist review.")

st.divider()
st.error(
    "Research/demo only. Raw model scores are not diagnoses or calibrated "
    "probabilities. Final clinical interpretation requires radiologist review."
)
