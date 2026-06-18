import numpy as np

from src.perception_safety_copilot.preprocessing import (
    apply_gamma_correction,
    assess_visibility,
    build_enhancement_variants,
    enhance_image_for_visibility,
)


def test_gamma_correction_changes_dark_image():
    image = np.full((20, 20, 3), 40, dtype=np.uint8)
    corrected = apply_gamma_correction(image, gamma=1.4)

    assert corrected.shape == image.shape
    assert corrected.mean() > image.mean()


def test_visibility_assessment_returns_expected_fields():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:24, 8:24] = 180

    assessment = assess_visibility(image)

    assert assessment.visibility_level in {"Poor", "Moderate", "Good"}
    assert 0.0 <= assessment.visibility_score <= 100.0
    assert assessment.contrast_std >= 0.0


def test_enhancement_pipeline_preserves_image_shape_and_dtype():
    image = np.random.default_rng(7).integers(0, 255, size=(40, 50, 3), dtype=np.uint8)

    enhanced = enhance_image_for_visibility(
        image,
        use_clahe=True,
        use_gamma=True,
        gamma=1.2,
        use_sharpening=True,
        sharpening_strength=1.0,
    )

    assert enhanced.shape == image.shape
    assert enhanced.dtype == np.uint8


def test_build_enhancement_variants_includes_weather_variants():
    image = np.random.default_rng(3).integers(0, 255, size=(24, 24, 3), dtype=np.uint8)
    variants = build_enhancement_variants(image, gamma=1.2, sharpening_strength=1.0)

    names = [variant.name for variant in variants]
    assert "Original" in names
    assert "Deraining" in names
    assert "Dehazing" in names
    assert "Low-Light" in names
