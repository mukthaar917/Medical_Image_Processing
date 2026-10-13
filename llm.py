from __future__ import annotations

import requests


SYSTEM_RULES = """
You are summarizing retrieved medical reference material for a radiologist-facing
research/demo application.

STRICT RULES:
1. You do NOT receive image model scores and must never invent or estimate them.
2. Do not decide whether any pathology is present or absent in the image.
3. Do not diagnose, classify, rank, or assign severity to any condition.
4. Do not infer laterality, location, measurements, patient history, symptoms,
   demographics, urgency, treatment, or management.
5. Retrieved RAG material is background information only.
6. Preserve medical terminology from the retrieved text. Do not rename conditions.
7. Make clear that the retrieved context does not determine what is present in the image.
8. The final clinical interpretation belongs to the radiologist.

OUTPUT FORMAT:

### Retrieved clinical background
Provide a concise summary of the retrieved reference material only.

### Context limitations
State that the material is background information and cannot establish whether a
finding is present in the uploaded radiograph.
""".strip()


def build_prompt(retrieved_chunks) -> str:
    context = "\n\n".join(
        f"[{i + 1}] {chunk.title}\n{chunk.text}"
        for i, chunk in enumerate(retrieved_chunks)
    )

    return f"""
{SYSTEM_RULES}

RETRIEVED REFERENCE MATERIAL:
{context}

Summarize only the retrieved reference material.
""".strip()


def generate_with_ollama(
    retrieved_chunks,
    model: str = "llama3.2:1b",
    endpoint: str = "http://localhost:11434/api/generate",
) -> str:
    prompt = build_prompt(retrieved_chunks)

    response = requests.post(
        endpoint,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
            },
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def fallback_summary(retrieved_chunks) -> str:
    refs = "\n".join(
        f"- {chunk.title}"
        for chunk in retrieved_chunks
    )

    return f"""### Retrieved clinical background
Retrieved reference sections:
{refs}

### Context limitations
This material is background information only. It does not determine whether any
pathology is present or absent in the uploaded radiograph. Image interpretation
and the final clinical conclusion require radiologist review.
"""
