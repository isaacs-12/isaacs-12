# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "scipy>=1.11", "opencv-python-headless>=4.8", "matplotlib>=3.8", "pytest>=8"]
# ///
"""Synthetic ground-truth test: a textured patch oscillating at a known rate
on a textured background, filmed by a camera that pans and zooms."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import breathscope

W, H, FPS, SECONDS = 480, 360, 30, 24
ROI = (170, 120, 140, 120)


def _texture(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    img = cv2.GaussianBlur(rng.uniform(0, 255, (h, w)).astype(np.float32), (0, 0), 2.0)
    return cv2.normalize(img, None, 20, 235, cv2.NORM_MINMAX)


def _render(path: Path, true_bpm: float, amp_px: float) -> None:
    rng = np.random.default_rng(0)
    scene = _texture(rng, H + 80, W + 80)
    x, y, w, h = ROI
    patch = _texture(rng, h, w)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n = FPS * SECONDS
    for i in range(n):
        t = i / FPS
        frame = scene.copy()
        dy = amp_px * np.sin(2 * np.pi * true_bpm / 60 * t)
        shift = np.float32([[1, 0, x + 40], [0, 1, y + 40 - dy]])
        cv2.warpAffine(patch, shift, (W + 80, H + 80), dst=frame,
                       flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        # Camera: slow pan plus 6% zoom over the clip.
        zoom = 1 + 0.06 * i / n
        cam = cv2.getRotationMatrix2D(((W + 80) / 2, (H + 80) / 2), 0, zoom)
        cam[:, 2] += (-40 + 15 * i / n, -40 - 8 * i / n)
        view = cv2.warpAffine(frame, cam, (W, H), flags=cv2.INTER_CUBIC)
        view += rng.normal(0, 1.5, view.shape).astype(np.float32)
        writer.write(cv2.cvtColor(np.clip(view, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    writer.release()


@pytest.mark.parametrize("true_bpm", [12.0, 17.0, 24.0])
def test_recovers_known_rate_under_camera_motion(tmp_path: Path, true_bpm: float) -> None:
    video = tmp_path / "synthetic.mp4"
    _render(video, true_bpm, amp_px=0.4)
    out = tmp_path / "out"
    code = breathscope.main([str(video), "--roi", ",".join(map(str, ROI)), "-o", str(out)])
    assert code == 0
    result = json.loads((out / "result.json").read_text())
    assert result["rate"]["spectral_bpm"] == pytest.approx(true_bpm, abs=1.0)
    assert result["rate"]["interval_bpm"] == pytest.approx(true_bpm, abs=1.5)
    assert result["subject_peak_amplitude_px"] == pytest.approx(0.4, rel=0.35)
    assert result["subject_to_scene_ratio"] > 5
    assert (out / "report.png").stat().st_size > 0
    assert (out / "magnified.mp4").stat().st_size > 0


def test_without_roi_writes_reference_frame(tmp_path: Path) -> None:
    video = tmp_path / "synthetic.mp4"
    _render(video, 15.0, amp_px=0.4)
    out = tmp_path / "out"
    assert breathscope.main([str(video), "-o", str(out)]) == 2
    assert (out / "reference_frame.png").exists()
