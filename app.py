\
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from inference import ChestXRayAnalyzer
from rag import SimpleRAG
from llm import generate_with_ollama, fallback_summary


st.set_page_config(
    page_title="Medical Imaging AI Research Dashboard",
    page_icon="🩻",
    layout="wide",
)

st.title("🩻 Medical Imaging Processing — Research Dashboard")
st.caption(
    "Chest X-ray model inference + RAG + LLM summary. "
    "For research/demo use only — not for diagnosis or clinical decision-making."
)

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Display threshold (model score)", 0.0, 1.0, 0.50, 0.05
    )
    top_k = st.slider("RAG chunks", 1, 5, 3)
    use_llm = st.toggle("Use local Ollama LLM", value=True)
    ollama_model = st.text_input("Ollama model", value="llama3.2:3b")
    st.divider()
    st.warning(
        "Do not upload identifiable patient data to a public deployment. "
        "Use de-identified research images."
    )


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
            st.image(arr, caption="Uploaded DICOM preview", clamp=True)
        except Exception as exc:
            st.info(f"DICOM preview unavailable: {exc}")
    else:
        image = Image.open(BytesIO(data)).convert("L")
        st.image(image, caption="Uploaded X-ray", use_container_width=True)


uploaded = st.file_uploader(
    "Upload a chest X-ray",
    type=["png", "jpg", "jpeg", "dcm"],
    help="Use a de-identified chest X-ray for this research demo.",
)

if uploaded is None:
    st.info("Upload a chest X-ray to start.")
    st.stop()

image_bytes = uploaded.getvalue()

left, right = st.columns([1, 1.25], gap="large")

with left:
    st.subheader("Input image")
    display_uploaded_image(uploaded.name, image_bytes)

with st.spinner("Running chest X-ray model..."):
    analyzer = load_analyzer()
    predictions = analyzer.predict_bytes(image_bytes, uploaded.name)

ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
visible = [(name, score) for name, score in ranked if score >= threshold]
if not visible:
    visible = ranked[:5]

df = pd.DataFrame(visible, columns=["Finding", "Model score"])

with right:
    st.subheader("Model outputs")
    st.dataframe(
        df.style.format({"Model score": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.bar_chart(df.set_index("Finding")["Model score"])

query = "Chest X-ray findings: " + ", ".join(
    f"{name} {score:.3f}" for name, score in ranked[:6]
)

with st.spinner("Retrieving reference context..."):
    rag = load_rag()
    retrieved = rag.retrieve(query, top_k=top_k)

st.subheader("Retrieved context")
for chunk in retrieved:
    with st.expander(f"{chunk.title} · similarity {chunk.score:.3f}"):
        st.write(chunk.text)

st.subheader("AI-generated research summary")
if use_llm:
    try:
        with st.spinner("Generating summary with local LLM..."):
            summary = generate_with_ollama(
                predictions, retrieved, model=ollama_model
            )
        st.markdown(summary)
    except Exception as exc:
        st.warning(
            "Local LLM was unavailable, so a deterministic fallback summary "
            f"is shown instead. Details: {exc}"
        )
        st.markdown(fallback_summary(predictions, retrieved))
else:
    st.markdown(fallback_summary(predictions, retrieved))

st.divider()
st.error(
    "Research/demo only. This application does not establish a diagnosis, "
    "exclude disease, or replace interpretation by a qualified radiologist."
)
