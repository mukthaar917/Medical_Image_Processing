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
from xray_router import BODY_PART_OPTIONS, route_xray


st.set_page_config(
    page_title="AI-Assisted X-ray Review",
    page_icon="🩻",
    layout="wide",
)


def check_login() -> bool:
    if st.session_state.get("authenticated", False):
        return True

    st.title("🔐 AI-Assisted X-ray Review")
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
def load_chest_analyzer():
    return ChestXRayAnalyzer()


@st.cache_resource
def load_chest_rag():
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
                caption="Uploaded X-ray",
                clamp=True,
                use_container_width=True,
            )
        except Exception as exc:
            st.info(f"DICOM preview unavailable: {exc}")
    else:
        image = Image.open(BytesIO(data)).convert("L")
        st.image(
            image,
            caption="Uploaded X-ray",
            use_container_width=True,
        )


def build_score_table(predictions: dict[str, float]) -> pd.DataFrame:
    ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    return pd.DataFrame(
        [
            {
                "Rank": i + 1,
                "Pathology label": name,
                "Raw model score": score,
            }
            for i, (name, score) in enumerate(ranked)
        ]
    )


def build_final_report(
    file_name: str,
    body_part: str,
    route_description: str,
    model_section: str,
    rag_summary: str,
    findings: str,
    impression: str,
    final_diagnosis: str,
    reviewer: str,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""FINAL RADIOLOGIST-REVIEWED DEMO REPORT

Generated: {now}
Study: X-ray
Detected/selected body region: {body_part}
Routing: {route_description}
File: {file_name}
Reviewed by: {reviewer or "Not specified"}

AI MODEL SECTION
{model_section}

RAG / LLM BACKGROUND SUMMARY
{rag_summary or "Not available for this route."}

RADIOLOGIST FINDINGS
{findings}

RADIOLOGIST IMPRESSION
{impression}

FINAL DIAGNOSIS / CLINICAL CONCLUSION
{final_diagnosis}

STATUS
Radiologist-reviewed / demo workflow.

NOTE
AI outputs are decision-support information only. The final diagnosis/clinical
conclusion above is radiologist-entered and approved.
"""


st.title("🩻 AI-Assisted X-ray Review Dashboard")
st.caption(
    "Body-part routing → task-specific model when configured → "
    "RAG background → LLM background summary → radiologist review."
)

with st.sidebar:
    st.header("Demo settings")

    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

    top_n_display = st.slider(
        "Chest: model scores to highlight",
        min_value=3,
        max_value=18,
        value=10,
    )

    rag_findings = st.slider(
        "Chest: labels used only for RAG retrieval",
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
        "Use Ollama for RAG-background summary",
        value=True,
    )

    ollama_model = st.text_input(
        "Ollama model",
        value="llama3.2:1b",
    )


uploaded = st.file_uploader(
    "Upload an X-ray",
    type=["png", "jpg", "jpeg", "dcm"],
    help="Use a de-identified X-ray for this demonstration.",
)

if uploaded is None:
    st.info("Upload an X-ray to begin.")
    st.stop()

image_bytes = uploaded.getvalue()
suffix = Path(uploaded.name).suffix.lower()

if suffix == ".dcm":
    manual_body_part = st.selectbox(
        "Body region override / fallback",
        BODY_PART_OPTIONS,
        index=0,
        help=(
            "For DICOM, metadata is used first. This selection is used as a "
            "fallback and can also be used to override after reviewing the route."
        ),
    )
else:
    manual_body_part = st.selectbox(
        "Select X-ray body region",
        BODY_PART_OPTIONS,
        index=0,
        help=(
            "JPEG/PNG files usually do not contain reliable DICOM body-part "
            "metadata, so select the body region manually."
        ),
    )

route = route_xray(
    uploaded.name,
    image_bytes,
    manual_body_part=manual_body_part,
)

st.subheader("Study routing")
c1, c2, c3 = st.columns(3)
c1.metric("Body region", route.body_part)
c2.metric("Model route", route.model_name or "No model configured")
c3.metric("Routing source", route.confidence)

with st.expander("Routing details"):
    st.write(route.source)
    st.write(route.reason)

left, right = st.columns([1, 1.35], gap="large")

with left:
    st.subheader("Input image")
    display_uploaded_image(uploaded.name, image_bytes)

score_table = None
rag_summary = ""
model_section = ""

if route.supported and route.model_key == "chest_torchxrayvision":
    with st.spinner("Running chest X-ray model..."):
        analyzer = load_chest_analyzer()
        predictions = analyzer.predict_bytes(
            image_bytes,
            uploaded.name,
        )

    score_table = build_score_table(predictions)

    with right:
        st.subheader("Raw chest-model outputs")
        st.caption(
            "Rendered directly by Python from TorchXRayVision output. "
            "The LLM does not generate or rewrite these labels or values."
        )

        highlight_df = score_table.head(top_n_display)

        st.dataframe(
            highlight_df.style.format(
                {"Raw model score": "{:.3f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            highlight_df[
                ["Pathology label", "Raw model score"]
            ].set_index("Pathology label")
        )

    with st.expander("View all 18 raw chest-model outputs"):
        st.dataframe(
            score_table.style.format(
                {"Raw model score": "{:.3f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        "Download all raw chest-model scores (.csv)",
        data=score_table.to_csv(index=False),
        file_name="chest_xray_raw_model_scores.csv",
        mime="text/csv",
    )

    ranked_labels = (
        score_table["Pathology label"]
        .head(rag_findings)
        .tolist()
    )

    query = (
        "Clinical reference background for chest X-ray terminology: "
        + ", ".join(ranked_labels)
    )

    st.divider()
    st.subheader("RAG clinical background")
    st.caption(
        "Selected model labels are used only as retrieval terms. "
        "RAG similarity does not indicate disease presence."
    )

    with st.spinner("Retrieving chest reference background..."):
        rag = load_chest_rag()
        retrieved = rag.retrieve(
            query,
            top_k=top_k,
        )

    for chunk in retrieved:
        with st.expander(
            f"{chunk.title} · retrieval similarity {chunk.score:.3f}"
        ):
            st.write(chunk.text)

    st.subheader("LLM summary of RAG background only")
    st.caption(
        "Ollama receives retrieved reference text only. "
        "It does not receive model scores or the uploaded image."
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

    score_lines = "\n".join(
        f"- {row['Pathology label']}: {row['Raw model score']:.3f}"
        for _, row in score_table.iterrows()
    )

    model_section = (
        "TorchXRayVision DenseNet121 raw scores. "
        "No generic 0.5 diagnostic threshold applied.\n\n"
        + score_lines
    )

else:
    with right:
        st.subheader("AI model status")
        st.warning(
            f"{route.body_part}: no validated task-specific model is "
            "configured in this prototype."
        )
        st.write(
            "No chest-model scores are generated for this image. "
            "This prevents inappropriate use of the chest model on other body regions."
        )

    model_section = (
        f"No task-specific AI model configured for {route.body_part}. "
        "No automated image-classification scores were generated."
    )

    st.divider()
    st.info(
        "RAG/LLM background is disabled for this route because the current "
        "knowledge base is chest-X-ray specific. Add a body-part-specific model "
        "and knowledge base before enabling AI assistance for this region."
    )

st.divider()
st.subheader("Radiologist review")

st.info(
    "The AI components do not create the final diagnosis. "
    "A final report becomes available only after radiologist-entered review."
)

reviewer = st.text_input("Radiologist / reviewer name")

radiologist_findings = st.text_area(
    "Radiologist findings",
    height=150,
    placeholder="Enter findings after reviewing the original radiograph...",
)

radiologist_impression = st.text_area(
    "Radiologist impression",
    height=120,
    placeholder="Enter the radiologist impression...",
)

final_diagnosis = st.text_area(
    "Final diagnosis / clinical conclusion",
    height=100,
    placeholder="Enter the final clinician-approved conclusion...",
)

approved = st.checkbox(
    "I have reviewed the original image and approve the final report content."
)

if approved:
    if (
        not radiologist_impression.strip()
        and not final_diagnosis.strip()
    ):
        st.warning(
            "Enter a radiologist impression or final clinical conclusion "
            "before producing the final report."
        )
    else:
        final_report = build_final_report(
            uploaded.name,
            route.body_part,
            route.source,
            model_section,
            rag_summary,
            radiologist_findings,
            radiologist_impression,
            final_diagnosis,
            reviewer,
        )

        st.success("Radiologist-reviewed report is ready.")

        st.download_button(
            "Download final radiologist-reviewed report (.txt)",
            data=final_report,
            file_name="radiologist_reviewed_xray_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
else:
    st.caption("Final report status: Pending radiologist review.")

st.divider()
st.error(
    "Research/demo only. A task-specific validated model is required for each "
    "body region. The chest model is not used for non-chest X-rays."
)
