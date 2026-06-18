from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class VisibilityAssessment:
    brightness_mean: float
    contrast_std: float
    sharpness_laplacian_var: float
    visibility_score: float
    visibility_level: str


@dataclass(frozen=True)
class EnhancementVariant:
    name: str
    image_rgb: np.ndarray


def apply_clahe(image_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def apply_gamma_correction(image_rgb: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    gamma = max(gamma, 0.1)
    inverse_gamma = 1.0 / gamma
    table = np.array([((index / 255.0) ** inverse_gamma) * 255 for index in range(256)], dtype=np.uint8)
    return cv2.LUT(image_rgb, table)


def apply_sharpening(image_rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
    strength = max(strength, 0.0)
    base_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    identity_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    kernel = identity_kernel * (1.0 - strength) + base_kernel * strength
    sharpened = cv2.filter2D(image_rgb, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_deraining(image_rgb: np.ndarray) -> np.ndarray:
    # Lightweight MVP deraining approximation: denoise streak-like texture and recover edges.
    denoised = cv2.fastNlMeansDenoisingColored(image_rgb, None, 8, 8, 7, 21)
    return apply_sharpening(denoised, strength=0.8)


def apply_dehazing(image_rgb: np.ndarray) -> np.ndarray:
    # Contrast restoration approximation for haze / spray / washed-out rain scenes.
    float_img = image_rgb.astype(np.float32) / 255.0
    min_per_channel = float_img.min(axis=(0, 1), keepdims=True)
    max_per_channel = float_img.max(axis=(0, 1), keepdims=True)
    stretched = (float_img - min_per_channel) / np.maximum(max_per_channel - min_per_channel, 1e-6)
    stretched = np.clip(stretched * 255.0, 0, 255).astype(np.uint8)
    return apply_clahe(stretched, clip_limit=2.5, tile_grid_size=8)


def apply_low_light_enhancement(image_rgb: np.ndarray) -> np.ndarray:
    enhanced = apply_gamma_correction(image_rgb, gamma=1.4)
    enhanced = apply_clahe(enhanced, clip_limit=2.5, tile_grid_size=8)
    return apply_sharpening(enhanced, strength=0.7)


def enhance_image_for_visibility(
    image_rgb: np.ndarray,
    *,
    use_clahe: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8,
    use_gamma: bool = True,
    gamma: float = 1.2,
    use_sharpening: bool = True,
    sharpening_strength: float = 1.0,
) -> np.ndarray:
    enhanced = image_rgb.copy()
    if use_clahe:
        enhanced = apply_clahe(enhanced, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid_size)
    if use_gamma:
        enhanced = apply_gamma_correction(enhanced, gamma=gamma)
    if use_sharpening:
        enhanced = apply_sharpening(enhanced, strength=sharpening_strength)
    return enhanced


def assess_visibility(image_rgb: np.ndarray) -> VisibilityAssessment:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    brightness_mean = float(gray.mean())
    contrast_std = float(gray.std())
    sharpness_laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    brightness_score = min(brightness_mean / 140.0, 1.0)
    contrast_score = min(contrast_std / 55.0, 1.0)
    sharpness_score = min(sharpness_laplacian_var / 250.0, 1.0)
    visibility_score = round((0.35 * brightness_score + 0.35 * contrast_score + 0.30 * sharpness_score) * 100, 1)

    if visibility_score < 40:
        visibility_level = "Poor"
    elif visibility_score < 70:
        visibility_level = "Moderate"
    else:
        visibility_level = "Good"

    return VisibilityAssessment(
        brightness_mean=round(brightness_mean, 1),
        contrast_std=round(contrast_std, 1),
        sharpness_laplacian_var=round(sharpness_laplacian_var, 1),
        visibility_score=visibility_score,
        visibility_level=visibility_level,
    )


def build_enhancement_variants(
    image_rgb: np.ndarray,
    *,
    gamma: float = 1.2,
    sharpening_strength: float = 1.0,
) -> list[EnhancementVariant]:
    return [
        EnhancementVariant("Original", image_rgb.copy()),
        EnhancementVariant("CLAHE", apply_clahe(image_rgb, clip_limit=2.0, tile_grid_size=8)),
        EnhancementVariant("Gamma", apply_gamma_correction(image_rgb, gamma=gamma)),
        EnhancementVariant("Sharpening", apply_sharpening(image_rgb, strength=sharpening_strength)),
        EnhancementVariant(
            "Combined",
            enhance_image_for_visibility(
                image_rgb,
                use_clahe=True,
                use_gamma=True,
                gamma=gamma,
                use_sharpening=True,
                sharpening_strength=sharpening_strength,
            ),
        ),
        EnhancementVariant("Deraining", apply_deraining(image_rgb)),
        EnhancementVariant("Dehazing", apply_dehazing(image_rgb)),
        EnhancementVariant("Low-Light", apply_low_light_enhancement(image_rgb)),
    ]
