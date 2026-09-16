\
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
import torchvision


class ChestXRayAnalyzer:
    """
    Research/demo inference wrapper around TorchXRayVision.
    The returned values are model outputs and must not be treated as a diagnosis.
    """

    def __init__(self, weights: str = "densenet121-res224-all"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = xrv.models.DenseNet(weights=weights).to(self.device)
        self.model.eval()
        self.transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
        ])

    def _prepare_from_path(self, path: str) -> torch.Tensor:
        # TorchXRayVision load_image handles PNG/JPG/DICOM and normalizes
        # to the range expected by its pretrained models.
        img = xrv.utils.load_image(path)
        img = self.transform(img)
        tensor = torch.from_numpy(img).float().unsqueeze(0)
        return tensor.to(self.device)

    def predict_bytes(self, image_bytes: bytes, original_name: str) -> dict[str, float]:
        suffix = Path(original_name).suffix.lower() or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            tensor = self._prepare_from_path(tmp_path)
            with torch.no_grad():
                output = self.model(tensor)[0].detach().cpu().numpy()
            return {
                pathology: float(score)
                for pathology, score in zip(self.model.pathologies, output)
            }
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
