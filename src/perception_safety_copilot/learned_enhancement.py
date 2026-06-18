from __future__ import annotations

from pathlib import Path
from urllib import request

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
ZERO_DCE_WEIGHTS_PATH = MODEL_DIR / "zero_dce" / "Epoch99.pth"
ZERO_DCE_WEIGHTS_URL = "https://huggingface.co/spaces/IanNathaniel/Zero-DCE/resolve/main/Epoch99.pth"


class ZeroDceModelUnavailable(RuntimeError):
    pass


def ensure_zero_dce_weights(weights_path: Path = ZERO_DCE_WEIGHTS_PATH) -> Path:
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    if weights_path.exists():
        return weights_path
    request.urlretrieve(ZERO_DCE_WEIGHTS_URL, weights_path)
    return weights_path


def _get_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:
        raise ZeroDceModelUnavailable("PyTorch is required for Zero-DCE learned enhancement.") from exc
    return torch, nn, F


def build_zero_dce_network():
    torch, nn, F = _get_torch()

    class EnhanceNetNoPool(nn.Module):
        def __init__(self):
            super().__init__()
            number_f = 32
            self.relu = nn.ReLU(inplace=True)
            self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
            self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
            self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
            self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
            self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
            self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
            self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

        def forward(self, x):
            x1 = self.relu(self.e_conv1(x))
            x2 = self.relu(self.e_conv2(x1))
            x3 = self.relu(self.e_conv3(x2))
            x4 = self.relu(self.e_conv4(x3))
            x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
            x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
            x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

            r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
            x = x + r1 * (torch.pow(x, 2) - x)
            x = x + r2 * (torch.pow(x, 2) - x)
            x = x + r3 * (torch.pow(x, 2) - x)
            enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
            x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
            x = x + r6 * (torch.pow(x, 2) - x)
            x = x + r7 * (torch.pow(x, 2) - x)
            enhance_image = x + r8 * (torch.pow(x, 2) - x)
            return enhance_image

    return EnhanceNetNoPool()


def load_zero_dce_model(weights_path: Path = ZERO_DCE_WEIGHTS_PATH):
    torch, _, _ = _get_torch()
    resolved_path = ensure_zero_dce_weights(weights_path)
    model = build_zero_dce_network()
    state_dict = torch.load(resolved_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def apply_zero_dce_low_light(image_rgb: np.ndarray, model=None) -> np.ndarray:
    torch, _, _ = _get_torch()
    if model is None:
        model = load_zero_dce_model()

    image_float = image_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_float).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        enhanced = model(tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()

    enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
    return enhanced
