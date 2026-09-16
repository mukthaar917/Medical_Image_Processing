\
from __future__ import annotations

import requests


SYSTEM_RULES = """
You are writing a research-only medical imaging AI summary.
Never claim that the model has made a definitive diagnosis.
Never recommend medication, treatment, or emergency disposition.
Do not invent patient history, demographics, symptoms, measurements, or findings.
Only discuss the supplied model outputs and retrieved reference context.
Call model values "model scores" or "signals", not clinical probabilities.
State that a qualified clinician/radiologist must review the original image.
Keep the output concise and structured with:
1. Model-assisted findings
2. RAG context
3. Limitations
""".strip()


def build_prompt(predictions: dict[str, float], retrieved_chunks) -> str:
    top = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:6]
    pred_text = "\n".join(f"- {name}: {score:.3f}" for name, score in top)
    context = "\n\n".join(
        f"[{i+1}] {chunk.title}\n{chunk.text}"
        for i, chunk in enumerate(retrieved_chunks)
    )

    return f"""
{SYSTEM_RULES}

MODEL OUTPUTS (top signals):
{pred_text}

RETRIEVED CONTEXT:
{context}

Create the research-only summary now.
""".strip()


def generate_with_ollama(
    predictions: dict[str, float],
    retrieved_chunks,
    model: str = "llama3.2:3b",
    endpoint: str = "http://localhost:11434/api/generate",
) -> str:
    prompt = build_prompt(predictions, retrieved_chunks)
    response = requests.post(
        endpoint,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def fallback_summary(predictions: dict[str, float], retrieved_chunks) -> str:
    top = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]
    findings = "\n".join(
        f"- {name}: model score {score:.3f}" for name, score in top
    )
    refs = "\n".join(f"- {c.title}" for c in retrieved_chunks)
    return f"""### Model-assisted findings
{findings}

### RAG context
Retrieved sections:
{refs}

### Limitations
This is a research/demo output from a pretrained chest X-ray model. Model scores are not calibrated diagnoses and may not generalize to a new institution, scanner, patient population, projection, or image quality. A qualified radiologist/clinician must review the original image.
"""
