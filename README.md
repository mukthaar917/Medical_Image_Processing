# Medical Imaging Processing Dashboard — Research MVP

This project is a **research/demo application**, not a medical device and not intended for diagnosis or clinical decision-making.

## MVP scope

- Upload chest X-ray PNG/JPG/JPEG/DICOM
- Preprocess with TorchXRayVision utilities
- Run a pretrained DenseNet121 chest X-ray model
- Show model output scores
- Retrieve relevant local text using Sentence Transformers
- Generate a constrained summary using a local Ollama LLM
- Streamlit dashboard

## Architecture

X-ray upload
→ preprocessing
→ TorchXRayVision DenseNet121
→ pathology model scores
→ semantic RAG retrieval
→ constrained LLM prompt
→ research summary + dashboard

## 1. Create environment

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

The first run may download pretrained TorchXRayVision and Sentence Transformer weights.

## 2. Optional local LLM

Install Ollama, then pull a small model:

```bash
ollama pull llama3.2:3b
```

Start Ollama if it is not already running. If Ollama is unavailable, the app still displays a deterministic fallback summary.

## 3. Run

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## 4. What to demo

1. Upload a de-identified chest X-ray.
2. Show the image.
3. Show the ranked model outputs.
4. Open the retrieved RAG context.
5. Show the LLM-generated research summary.
6. Explain that the system is a prototype and requires validation before clinical use.

## MRI extension

Do not route arbitrary MRI studies into the chest-X-ray model.

For MRI, create a separate task-specific pipeline, for example:
- brain-tumor segmentation
- brain lesion segmentation
- knee abnormality classification

MONAI Model Zoo bundles are a good starting point for task-specific medical image models. The MRI pipeline should have its own preprocessing, model, labels, validation set, and summary template.

## Suggested project title

**MedVision RAG: Deep Learning and LLM-Based Medical Imaging Research Dashboard**
