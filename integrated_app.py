from __future__ import annotations

import os
import hmac
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import torch
import open_clip

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# Keep your existing RAG import
from rag import SimpleRAG 

st.set_page_config(
    page_title="AI-Assisted Chest X-ray Review",
    page_icon="🩻",
    layout="wide",
)

# ---------------------------------------------------------
# AUTHENTICATION
# ---------------------------------------------------------
def check_login() -> bool:
    if st.session_state.get("authenticated", False):
        return True

    st.title("🔐 AI-Assisted Chest X-ray Review")
    st.caption("Authorized clinical validation access")

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


# ---------------------------------------------------------
# BIOMEDCLIP & RAG PIPELINE
# ---------------------------------------------------------
class ClinicalRadiologyPipeline:
    def __init__(self, ollama_model_name="llama3"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load BiomedCLIP
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 2. ChestXray8 Pathologies
        self.pathologies = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", 
            "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", 
            "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"
        ]
        
        # 3. Calibrated Thresholds 
        # Update these values with the exact output from validate_thresholds.py
        self.thresholds = {
            "Atelectasis": 0.62, "Cardiomegaly": 0.58, "Effusion": 0.65, 
            "Infiltration": 0.55, "Mass": 0.71, "Nodule": 0.68, 
            "Pneumonia": 0.60, "Pneumothorax": 0.75, "Consolidation": 0.64, 
            "Edema": 0.59, "Emphysema": 0.66, "Fibrosis": 0.70, 
            "Pleural Thickening": 0.63, "Hernia": 0.80
        }
        
        # 4. Initialize Local LLM
        self.llm = Ollama(model=ollama_model_name) 
        
    def extract_validated_findings(self, image: Image.Image):
        # Convert incoming image to RGB tensor for BiomedCLIP
        image_tensor = self.preprocess(image.convert('RGB')).unsqueeze(0).to(self.device)
        
        texts = [f"Chest X-ray showing {path}" for path in self.pathologies]
        text_tokens = self.tokenizer(texts).to(self.device)
        
        with torch.no_grad():
            image_features, text_features, logit_scale = self.model(image_tensor, text_tokens)
            logits = (logit_scale * image_features @ text_features.T).softmax(dim=-1)
            raw_scores = logits[0].cpu().tolist()
            
        findings = {"Positive": [], "Negative": [], "RawScores": {}}
        for path, score in zip(self.pathologies, raw_scores):
            findings["RawScores"][path] = score
            # Apply thresholding validation
            if score >= self.thresholds[path]:
                findings["Positive"].append(path)
            else:
                findings["Negative"].append(path)
                
        return findings

    def generate_draft_report(self, positive_findings: list, rag_context: str):
        template = """
        You are an expert radiologist writing a final diagnostic report.
        
        STRICT RULES:
        1. Do NOT modify or rename the provided pathologies.
        2. Do NOT invent or infer disease severity (e.g., do not use words like "severe", "high", "moderate", or "significant").
        3. Formulate the 'Impression' based ONLY on the provided Positive Findings and the supporting Clinical Context.
        4. Do NOT diagnose or mention conditions not listed in the Positive Findings.
        
        VALIDATED MODEL FINDINGS:
        Positive: {positive_findings}
        
        SUPPORTING CLINICAL CONTEXT (RAG):
        {rag_context}
        
        Generate a professional, highly structured medical report containing strictly a 'Findings' section and an 'Impression' section.
        """
        prompt = PromptTemplate(input_variables=["positive_findings", "rag_context"], template=template)
        pos_str = ", ".join(positive_findings) if positive_findings else "None"
        
        chain = prompt | self.llm
        return chain.invoke({"positive_findings": pos_str, "rag_context": rag_context})

@st.cache_resource
def load_pipeline(ollama_model):
    return ClinicalRadiologyPipeline(ollama_model_name=ollama_model)

@st.cache_resource
def load_rag():
    return SimpleRAG("knowledge_base.md")


# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------
def process_uploaded_image(file_name: str, data: bytes) -> Image.Image:
    """Safely reads PNG, JPG, or DICOM into a PIL Image for model ingestion."""
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
            arr = (arr * 255).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")
        except Exception as exc:
            st.error(f"DICOM processing error: {exc}")
            st.stop()
    else:
        return Image.open(BytesIO(data)).convert("RGB")


def make_radiologist_report(
    file_name: str,
    positive_findings: list[str],
    draft_report: str,
    radiologist_findings: str,
    radiologist_impression: str,
    final_diagnosis: str,
    reviewer: str,
) -> str:
    pos_str = ", ".join(positive_findings) if positive_findings else "None"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""FINAL RADIOLOGIST-REVIEWED REPORT

Generated: {now}
Study: Chest X-ray
File: {file_name}
Reviewed by: {reviewer or "Not specified"}

VALIDATED MODEL FINDINGS (Positive)
{pos_str}

AI DRAFT REPORT (RAG + LLM)
{draft_report}

--- RADIOLOGIST FINAL REVIEW ---
FINDINGS:
{radiologist_findings}

IMPRESSION:
{radiologist_impression}

CLINICAL CONCLUSION:
{final_diagnosis}

STATUS
Radiologist-reviewed and approved.
"""


# ---------------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------------
st.title("🩻 AI-Assisted Chest X-ray Review Dashboard")
st.caption(
    "BiomedCLIP zero-shot → Calibrated Thresholds → Conditioned RAG → "
    "Constrained LLM Draft → Radiologist Review."
)

with st.sidebar:
    st.header("Pipeline Settings")
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

    top_k = st.slider("RAG chunks returned per finding", min_value=1, max_value=5, value=2)
    ollama_model = st.text_input("Ollama model", value="llama3")

    st.warning("Pathology thresholds are dynamically applied. Only clinically positive classifications trigger RAG context and LLM interpretation.")


uploaded = st.file_uploader(
    "Upload a chest X-ray",
    type=["png", "jpg", "jpeg", "dcm"],
    help="Upload PNG, JPG, or DICOM scan.",
)

if uploaded is None:
    st.info("Upload a chest X-ray to begin.")
    st.stop()

image_bytes = uploaded.getvalue()
pil_image = process_uploaded_image(uploaded.name, image_bytes)

left, right = st.columns([1, 1.35], gap="large")

with left:
    st.subheader("Input image")
    st.image(pil_image, caption="Processed Upload", use_container_width=True)

with st.spinner("Running BiomedCLIP model & thresholding..."):
    pipeline = load_pipeline(ollama_model)
    findings = pipeline.extract_validated_findings(pil_image)

with right:
    st.subheader("Validated Clinical Findings")
    
    if findings["Positive"]:
        st.error(f"🚨 **Positive Pathologies Detected:** {', '.join(findings['Positive'])}")
    else:
        st.success("✅ **No Positive Pathologies Detected** (All scores below threshold)")
        
    with st.expander("View Negative Findings"):
        st.write(", ".join(findings["Negative"]))
        
    with st.expander("View Raw Model Probabilities"):
        st.json(findings["RawScores"])

# --- RAG & LLM Execution ---
st.divider()
st.subheader("Conditioned RAG & LLM Synthesis")

rag_context = ""
retrieved_chunks = []

if findings["Positive"]:
    with st.spinner("Retrieving clinical background for positive findings..."):
        rag = load_rag()
        query = "Clinical background for these chest X-ray pathology labels: " + ", ".join(findings["Positive"])
        retrieved_chunks = rag.retrieve(query, top_k=top_k)
        
        for chunk in retrieved_chunks:
            rag_context += f"{chunk.text}\n\n"
            with st.expander(f"{chunk.title} · similarity {chunk.score:.3f}"):
                st.write(chunk.text)
else:
    rag_context = "No abnormal findings supported by validated model thresholds."
    st.info("RAG skipped: No positive findings to evaluate.")

with st.spinner("Synthesizing Draft Report with Llama3..."):
    draft_report = pipeline.generate_draft_report(findings["Positive"], rag_context)

st.markdown("### AI Draft Report")
st.write(draft_report)

# --- Radiologist Review ---
st.divider()
st.subheader("Radiologist Review")
st.info("A final report becomes available only after radiologist-entered review.")

reviewer = st.text_input("Radiologist / reviewer name")

radiologist_findings = st.text_area(
    "Radiologist findings",
    height=150,
    placeholder="Enter image findings after reviewing the original radiograph...",
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

approved = st.checkbox("I have reviewed the original image and approve the final report content.")

if approved:
    if not radiologist_impression.strip() and not final_diagnosis.strip():
        st.warning("Enter a radiologist impression or final clinical conclusion before producing the final report.")
    else:
        final_report = make_radiologist_report(
            uploaded.name,
            findings["Positive"],
            draft_report,
            radiologist_findings,
            radiologist_impression,
            final_diagnosis,
            reviewer,
        )

        st.success("Radiologist-reviewed report is ready.")
        st.download_button(
            "Download final radiologist-reviewed report (.txt)",
            data=final_report,
            file_name="radiologist_reviewed_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
else:
    st.caption("Final report status: Pending radiologist review.")