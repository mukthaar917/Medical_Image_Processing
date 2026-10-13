from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional


@dataclass
class RouteResult:
    body_part: str
    source: str
    confidence: str
    model_key: Optional[str]
    model_name: Optional[str]
    supported: bool
    reason: str


BODY_PART_OPTIONS = [
    "Chest",
    "Upper Extremity",
    "Lower Extremity",
    "Spine",
    "Abdomen/Pelvis",
    "Other/Unknown",
]


def _normalize(text: str) -> str:
    return " ".join(str(text or "").upper().replace("_", " ").split())


def _map_text_to_body_part(text: str) -> Optional[str]:
    t = _normalize(text)

    chest_terms = [
        "CHEST", "THORAX", "CXR", "LUNG", "RIB",
    ]
    upper_terms = [
        "HAND", "WRIST", "FOREARM", "ELBOW", "HUMERUS",
        "SHOULDER", "CLAVICLE", "FINGER",
    ]
    lower_terms = [
        "FOOT", "ANKLE", "TIBIA", "FIBULA", "KNEE",
        "FEMUR", "HIP", "TOE",
    ]
    spine_terms = [
        "SPINE", "CERVICAL", "THORACIC SPINE", "LUMBAR",
        "LUMBOSACRAL", "SACRUM", "COCCYX",
    ]
    abdomen_terms = [
        "ABDOMEN", "PELVIS", "KUB",
    ]

    groups = [
        ("Chest", chest_terms),
        ("Upper Extremity", upper_terms),
        ("Lower Extremity", lower_terms),
        ("Spine", spine_terms),
        ("Abdomen/Pelvis", abdomen_terms),
    ]

    for body_part, terms in groups:
        if any(term in t for term in terms):
            return body_part

    return None


def detect_dicom_body_part(data: bytes) -> tuple[Optional[str], str]:
    """
    Detect body region from DICOM metadata only.

    Returns:
        (body_part, metadata_text)
    """
    try:
        import pydicom

        ds = pydicom.dcmread(BytesIO(data), stop_before_pixels=True, force=True)

        fields = {
            "BodyPartExamined": getattr(ds, "BodyPartExamined", ""),
            "StudyDescription": getattr(ds, "StudyDescription", ""),
            "SeriesDescription": getattr(ds, "SeriesDescription", ""),
            "ProtocolName": getattr(ds, "ProtocolName", ""),
            "ViewPosition": getattr(ds, "ViewPosition", ""),
        }

        metadata_text = " | ".join(
            f"{k}={v}" for k, v in fields.items() if str(v).strip()
        )

        # Prefer BodyPartExamined when available.
        direct = _map_text_to_body_part(fields["BodyPartExamined"])
        if direct:
            return direct, metadata_text

        # Fall back to descriptions.
        combined = " ".join(str(v) for v in fields.values())
        detected = _map_text_to_body_part(combined)
        return detected, metadata_text

    except Exception:
        return None, ""


def route_xray(
    file_name: str,
    data: bytes,
    manual_body_part: Optional[str] = None,
) -> RouteResult:
    suffix = Path(file_name).suffix.lower()

    detected = None
    source = "manual selection"
    confidence = "manual"

    if suffix == ".dcm":
        detected, metadata_text = detect_dicom_body_part(data)
        if detected:
            source = f"DICOM metadata ({metadata_text})"
            confidence = "metadata"
        elif manual_body_part:
            detected = manual_body_part
            source = "manual fallback"
            confidence = "manual"
    else:
        detected = manual_body_part
        source = "manual selection"
        confidence = "manual"

    body_part = detected or "Other/Unknown"

    if body_part == "Chest":
        return RouteResult(
            body_part=body_part,
            source=source,
            confidence=confidence,
            model_key="chest_torchxrayvision",
            model_name="TorchXRayVision DenseNet121 (densenet121-res224-all)",
            supported=True,
            reason="Chest X-ray route is configured.",
        )

    return RouteResult(
        body_part=body_part,
        source=source,
        confidence=confidence,
        model_key=None,
        model_name=None,
        supported=False,
        reason=(
            "No validated task-specific model is configured for this body region yet. "
            "The app will not generate chest-model scores for this image."
        ),
    )
