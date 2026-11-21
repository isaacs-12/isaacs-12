#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.26",
#   "scipy>=1.11",
#   "opencv-python>=4.8",
#   "matplotlib>=3.8",
# ]
# ///
"""breathscope - estimate respiratory rate from ordinary video, visually.

Pipeline
--------
1. Shot detection: only one continuous shot is analysed (cuts break tracking).
2. Sparse feature tracking (Shi-Tomasi + pyramidal Lucas-Kanade), always from
   the reference frame to frame t, so tracking error does not accumulate into
   low-frequency drift - which is exactly the band breathing lives in.
3. Camera compensation: a RANSAC similarity transform (pan / zoom / roll) is
   fitted per frame to the points *outside* the subject ROI. What is left over
   on each point is motion relative to the static scene, in reference pixels.
4. Subject signal: residual trajectories of the points inside the ROI are
   detrended, band-passed to the respiratory band and reduced with PCA. PC1 is
   the dominant coherent motion of the subject. The identical computation on
   the static scene gives the noise floor ("the surroundings").
5. Rate: spectral peak, autocorrelation and peak-to-peak intervals, reported
   together with the frequency resolution the clip length actually permits.
6. Visuals: a report figure (maps of where the periodic motion is, trace,
   spectrum) and a stabilised, Eulerian motion-magnified side-by-side video.

Usage
-----
    uv run breathscope.py clip.mp4                # writes a gridded reference frame
    uv run breathscope.py clip.mp4 --roi X,Y,W,H  # full analysis
    uv run breathscope.py clip.mp4 --pick         # draw the ROI with the mouse
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy import signal as sps

# Chart palette (light surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SUBJECT = "#2a78d6"
SCENE = "#8a8983"
WARM = "#eb6834"
SEQ_RAMP = ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"]
DIV_RAMP = [WARM, "#f0efec", SUBJECT]

# Typical adult ranges, breaths/min. Awake-at-rest and asleep overlap almost
# completely, which is why a rate alone cannot establish sleep.
ADULT_REST_BPM = (12.0, 20.0)


@dataclass(frozen=True)
class Roi:
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def parse(cls, text: str) -> Roi:
        try:
            x, y, w, h = (int(round(float(v))) for v in text.split(","))
        except ValueError as exc:
            raise argparse.ArgumentTypeError("ROI must be X,Y,W,H in pixels") from exc
        if w <= 0 or h <= 0:
            raise argparse.ArgumentTypeError("ROI width and height must be positive")
        return cls(x, y, w, h)

    def contains(self, pts: np.ndarray, margin: float = 0.0) -> np.ndarray:
        return (
            (pts[:, 0] >= self.x - margin)
            & (pts[:, 0] <= self.x + self.w + margin)
            & (pts[:, 1] >= self.y - margin)
            & (pts[:, 1] <= self.y + self.h + margin)
        )


@dataclass
class RateEstimate:
    spectral_bpm: float | None
    autocorr_bpm: float | None
    interval_bpm: float | None
    intervals_s: list[float]
    cycles_observed: float
    resolution_bpm: float
    peak_amplitude_px: float
    peak_to_band_median: float
    peak_times_s: list[float] = field(default_factory=list)


@dataclass
class Tracks:
    p0: np.ndarray  # (N, 2) reference positions
    residual: np.ndarray  # (T, N, 2) motion relative to static scene, ref px
    valid: np.ndarray  # (T, N)
    transforms: np.ndarray  # (T, 2, 3) frame t -> reference
    static: np.ndarray  # (N,) consistently RANSAC-inlying scene points
    in_roi: np.ndarray  # (N,)


# --------------------------------------------------------------------------
# Video I/O
# --------------------------------------------------------------------------


def read_gray(path: Path, start_s: float, end_s: float | None) -> tuple[np.ndarray, float, int]:
    """Return (frames[T,H,W] uint8, fps, index of first returned frame)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        raise SystemExit("video reports no frame rate; re-mux it with ffmpeg first")
    first = int(round(start_s * fps))
    last = int(round(end_s * fps)) if end_s is not None else None
    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok or (last is not None and idx >= last):
            break
        if idx >= first:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        idx += 1
    cap.release()
    if len(frames) < 2:
        raise SystemExit("no frames in the requested time range")
    return np.stack(frames), float(fps), first


def detect_shots(gray: np.ndarray) -> list[tuple[int, int]]:
    """Split on hard cuts using mean absolute frame difference."""
    small = gray[:, ::4, ::4].astype(np.int16)
    diff = np.abs(np.diff(small, axis=0)).mean(axis=(1, 2))
    thresh = max(12.0, 8.0 * float(np.median(diff)))
    cuts = [0, *(np.flatnonzero(diff > thresh) + 1).tolist(), len(gray)]
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


# --------------------------------------------------------------------------
# Tracking and camera compensation
# --------------------------------------------------------------------------


def _detect_features(ref: np.ndarray, roi: Roi | None) -> np.ndarray:
    kw = dict(qualityLevel=0.004, blockSize=7)
    if roi is None:
        pts = cv2.goodFeaturesToTrack(ref, maxCorners=2500, minDistance=7, **kw)
        return pts.reshape(-1, 2)
    inside = np.zeros_like(ref)
    inside[max(roi.y, 0) : roi.y + roi.h, max(roi.x, 0) : roi.x + roi.w] = 255
    # Detect separately so a textured background cannot starve the subject.
    a = cv2.goodFeaturesToTrack(ref, maxCorners=900, minDistance=5, mask=inside, **kw)
    b = cv2.goodFeaturesToTrack(ref, maxCorners=1800, minDistance=7, mask=255 - inside, **kw)
    parts = [p.reshape(-1, 2) for p in (a, b) if p is not None]
    if not parts:
        raise SystemExit("no trackable features found")
    return np.concatenate(parts)


def track(gray: np.ndarray, roi: Roi | None, fb_tol: float = 1.0) -> Tracks:
    ref = gray[0]
    p0 = _detect_features(ref, roi).astype(np.float32)
    n, t_len = len(p0), len(gray)
    lk = dict(
        winSize=(25, 25),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.005),
    )
    pos = np.empty((t_len, n, 2), np.float32)
    valid = np.zeros((t_len, n), bool)
    guess = p0.copy()
    for t in range(t_len):
        # Reference -> t directly (seeded with the previous solution): no drift.
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(
            ref, gray[t], p0, guess.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk
        )
        back, st2, _ = cv2.calcOpticalFlowPyrLK(
            gray[t], ref, p1, p0.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk
        )
        fb = np.linalg.norm(back - p0, axis=1)
        ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < fb_tol)
        pos[t], valid[t] = p1, ok
        guess = np.where(ok[:, None], p1, guess)

    in_roi = roi.contains(p0) if roi is not None else np.zeros(n, bool)
    # Keep a margin around the ROI out of the camera fit (hair, clothing edges).
    anchor = ~roi.contains(p0, margin=12) if roi is not None else np.ones(n, bool)

    transforms = np.empty((t_len, 2, 3), np.float64)
    residual = np.full((t_len, n, 2), np.nan, np.float32)
    inlier_hits = np.zeros(n, np.int32)
    last = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    for t in range(t_len):
        use = anchor & valid[t]
        m = None
        if use.sum() >= 12:
            m, inl = cv2.estimateAffinePartial2D(
                pos[t][use], p0[use], method=cv2.RANSAC, ransacReprojThreshold=1.0,
                maxIters=4000, confidence=0.999, refineIters=20,
            )
            if m is not None:
                inlier_hits[np.flatnonzero(use)[inl.ravel() == 1]] += 1
        last = m if m is not None else last
        transforms[t] = last
        mapped = pos[t] @ last[:, :2].T + last[:, 2]
        residual[t] = np.where(valid[t][:, None], mapped - p0, np.nan)

    static = anchor & (inlier_hits >= 0.9 * t_len)
    if static.sum() < 12:
        raise SystemExit(
            "could not lock onto a static scene; give --roi so the subject is "
            "excluded from the camera fit, or pick a shot with visible static objects"
        )
    return Tracks(p0, residual, valid, transforms, static, in_roi)


# --------------------------------------------------------------------------
# Signal extraction
# --------------------------------------------------------------------------


def _complete(tracks: Tracks, select: np.ndarray, min_valid: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Trajectories (T, K, 2) for selected points, short gaps interpolated."""
    idx = np.flatnonzero(select & (tracks.valid.mean(axis=0) >= min_valid))
    traj = tracks.residual[:, idx].astype(np.float64)
    t = np.arange(traj.shape[0])
    for k in range(traj.shape[1]):
        bad = np.isnan(traj[:, k, 0])
        if bad.any():
            for c in range(2):
                traj[bad, k, c] = np.interp(t[bad], t[~bad], traj[~bad, k, c])
    return traj, idx


def _detrend(x: np.ndarray, order: int = 2) -> np.ndarray:
    """Remove a low-order polynomial along axis 0 (postural drift)."""
    t = np.linspace(-1, 1, x.shape[0])
    v = np.vander(t, order + 1)
    flat = x.reshape(x.shape[0], -1)
    coef, *_ = np.linalg.lstsq(v, flat, rcond=None)
    return (flat - v @ coef).reshape(x.shape)


def _bandpass(x: np.ndarray, fps: float, band_hz: tuple[float, float]) -> np.ndarray:
    sos = sps.butter(2, band_hz, btype="bandpass", fs=fps, output="sos")
    return sps.sosfiltfilt(sos, x, axis=0)


@dataclass
class Component:
    """Dominant coherent in-band motion of a set of points."""

    idx: np.ndarray  # indices into Tracks.p0
    band: np.ndarray  # (T,) band-passed PC1 time series, px
    wide: np.ndarray  # (T,) detrended-only projection onto PC1, px
    explained: float
    point_rms: np.ndarray  # (K,) per-point in-band RMS displacement, px
    direction: np.ndarray  # (2,) mean unit motion direction in image coords


def dominant_component(tracks: Tracks, select: np.ndarray, fps: float, band_hz: tuple[float, float]) -> Component:
    traj, idx = _complete(tracks, select)
    if traj.shape[1] < 8:
        raise SystemExit(
            f"only {traj.shape[1]} trackable points in the selection; enlarge the ROI "
            "or include textured edges (shoulder line, collar, hair)"
        )
    det = _detrend(traj)
    bp = _bandpass(det, fps, band_hz)
    t_len, k, _ = bp.shape
    flat = bp.reshape(t_len, 2 * k)
    _, s, vt = np.linalg.svd(flat - flat.mean(axis=0), full_matrices=False)
    w = vt[0]  # unit loading vector over (point, axis)
    # Express the component as mean per-point displacement in pixels.
    scale = np.linalg.norm(w.reshape(k, 2), axis=1).mean()
    w2 = w.reshape(k, 2)
    mean_dir = w2.mean(axis=0)
    if mean_dir[1] > 0:  # orient so that positive = upward in the image
        w, w2, mean_dir = -w, -w2, -mean_dir
    band = (flat @ w) * scale
    wide = (det.reshape(t_len, 2 * k) @ w) * scale
    norm = np.linalg.norm(mean_dir)
    return Component(
        idx=idx,
        band=band,
        wide=wide,
        explained=float(s[0] ** 2 / np.sum(s**2)),
        point_rms=np.sqrt((bp**2).sum(axis=2).mean(axis=0)),
        direction=mean_dir / norm if norm > 0 else mean_dir,
    )


def amplitude_spectrum(x: np.ndarray, fps: float, pad_to_bpm: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed, zero-padded amplitude spectrum. Returns (bpm, amplitude px)."""
    n = len(x)
    win = np.hanning(n)
    nfft = int(2 ** np.ceil(np.log2(max(n, fps * 60.0 / pad_to_bpm))))
    amp = np.abs(np.fft.rfft((x - x.mean()) * win, nfft)) * 2.0 / win.sum()
    return np.fft.rfftfreq(nfft, 1.0 / fps) * 60.0, amp


def estimate_rate(comp: Component, fps: float, band_hz: tuple[float, float]) -> RateEstimate:
    n = len(comp.band)
    duration = n / fps
    lo, hi = band_hz[0] * 60.0, band_hz[1] * 60.0

    bpm, amp = amplitude_spectrum(comp.wide, fps)
    in_band = (bpm >= lo) & (bpm <= hi)
    i = int(np.flatnonzero(in_band)[np.argmax(amp[in_band])])
    interior = 0 < i < len(amp) - 1 and amp[i] >= amp[i - 1] and amp[i] >= amp[i + 1]
    spectral = float(bpm[i]) if interior and in_band[i - 1] and in_band[i + 1] else None
    peak_amp = float(amp[i])
    ratio = float(peak_amp / np.median(amp[in_band]))

    x = comp.band - comp.band.mean()
    ac = sps.correlate(x, x, mode="full")[n - 1 :]
    ac /= np.arange(n, 0, -1)  # unbiased
    ac /= ac[0]
    lag_lo, lag_hi = int(fps / band_hz[1]), min(int(fps / band_hz[0]), n // 2)
    autocorr = None
    if lag_hi > lag_lo + 2:
        pk, _ = sps.find_peaks(ac[lag_lo:lag_hi], height=0.2)
        if len(pk):
            autocorr = float(60.0 * fps / (pk[0] + lag_lo))

    min_dist = max(1, int(0.6 * fps / band_hz[1]))
    prom = 0.4 * x.std()
    peaks, _ = sps.find_peaks(x, distance=min_dist, prominence=prom)
    troughs, _ = sps.find_peaks(-x, distance=min_dist, prominence=prom)
    intervals = np.concatenate([np.diff(peaks), np.diff(troughs)]) / fps
    interval_bpm = float(60.0 / np.median(intervals)) if len(intervals) else None
    best = spectral or interval_bpm or autocorr
    return RateEstimate(
        spectral_bpm=spectral,
        autocorr_bpm=autocorr,
        interval_bpm=interval_bpm,
        intervals_s=[round(float(v), 2) for v in intervals],
        cycles_observed=float(duration * best / 60.0) if best else 0.0,
        resolution_bpm=60.0 / duration,
        peak_amplitude_px=peak_amp,
        peak_to_band_median=ratio,
        peak_times_s=[round(float(p / fps), 2) for p in peaks],
    )


# --------------------------------------------------------------------------
# Visual outputs
# --------------------------------------------------------------------------


def save_reference_grid(ref_bgr: np.ndarray, out: Path, roi: Roi | None = None) -> None:
    img = ref_bgr.copy()
    h, w = img.shape[:2]
    for x in range(0, w, 100):
        cv2.line(img, (x, 0), (x, h), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, str(x), (x + 3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    for y in range(100, h, 100):
        cv2.line(img, (0, y), (w, y), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, str(y), (3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if roi:
        cv2.rectangle(img, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (214, 120, 42), 2)
    cv2.imwrite(str(out), img)


def read_color_frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {index}")
    return frame


def make_report(
    out: Path,
    ref_bgr: np.ndarray,
    roi: Roi,
    tracks: Tracks,
    subj: Component,
    scene: Component,
    rate: RateEstimate,
    fps: float,
    band_hz: tuple[float, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })
    seq = LinearSegmentedColormap.from_list("seq", SEQ_RAMP)
    div = LinearSegmentedColormap.from_list("div", DIV_RAMP)
    rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    dim = (0.45 * rgb + 0.55 * 255).astype(np.uint8)  # recessive backdrop for the dots
    t = np.arange(len(subj.band)) / fps
    lo, hi = band_hz[0] * 60, band_hz[1] * 60

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.6), constrained_layout=True)

    # Per-point in-band amplitude and correlation with the subject signal, for
    # every well-tracked point in the frame (subject and surroundings alike).
    traj, idx = _complete(tracks, np.ones(len(tracks.p0), bool))
    bp = _bandpass(_detrend(traj), fps, band_hz)
    rms = np.sqrt((bp**2).sum(axis=2).mean(axis=0))
    along = bp @ subj.direction  # motion along the subject's breathing axis
    ref_sig = subj.band - subj.band.mean()
    denom = np.linalg.norm(along - along.mean(axis=0), axis=0) * np.linalg.norm(ref_sig)
    corr = ((along - along.mean(axis=0)) * ref_sig[:, None]).sum(axis=0) / np.where(denom > 0, denom, 1)
    pts = tracks.p0[idx]

    def frame_axes(ax, title):
        ax.imshow(dim)
        ax.add_patch(Rectangle((roi.x, roi.y), roi.w, roi.h, fill=False, ec=INK, lw=1.2, ls="--"))
        ax.set_title(title)
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)

    ax = axes[0, 0]
    frame_axes(ax, f"Where is {lo:.0f}-{hi:.0f}/min motion?  In-band RMS displacement per tracked point")
    vmax = float(np.percentile(rms, 97))
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=rms, cmap=seq, vmin=0, vmax=vmax, s=16, lw=0.4, ec="white")
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01, label="px (camera motion removed)")

    ax = axes[0, 1]
    frame_axes(ax, "Which points move in step with the subject signal?  Correlation")
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=corr, cmap=div, vmin=-1, vmax=1, s=16, lw=0.4, ec="white")
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01, label="correlation (dashed box = subject ROI)")

    ax = axes[1, 0]
    ax.plot(t, scene.band, color=SCENE, lw=1.4, label="static scene (noise floor)")
    ax.plot(t, subj.wide - subj.wide.mean(), color=SUBJECT, lw=1, alpha=0.35, label="subject, detrended only")
    ax.plot(t, subj.band, color=SUBJECT, lw=2, label="subject, band-passed")
    pk = np.array(rate.peak_times_s)
    if len(pk):
        ax.plot(pk, np.interp(pk, t, subj.band), "o", ms=8, mfc=SUBJECT, mec=SURFACE, mew=2)
    ax.set_title(f"Coherent subject motion (PC1, {subj.explained:.0%} of in-band variance, {len(subj.idx)} points)")
    ax.set_xlabel("time in shot (s)"), ax.set_ylabel("displacement, + = up (px)")
    ax.legend(frameon=False, loc="upper right", ncols=3, fontsize=9)

    ax = axes[1, 1]
    bpm, a_subj = amplitude_spectrum(subj.wide, fps)
    _, a_scene = amplitude_spectrum(scene.wide, fps)
    keep = bpm <= 90
    ax.axvspan(*ADULT_REST_BPM, color=WARM, alpha=0.12, lw=0)
    ax.axvline(lo, color=INK_2, lw=0.8, ls=":"), ax.axvline(hi, color=INK_2, lw=0.8, ls=":")
    ax.plot(bpm[keep], a_scene[keep], color=SCENE, lw=1.4, label="static scene (noise floor)")
    ax.plot(bpm[keep], a_subj[keep], color=SUBJECT, lw=2, label="subject")
    top = float(a_subj[keep].max())
    ax.set_ylim(0, top * 1.25)
    ax.text(np.mean(ADULT_REST_BPM), top * 1.19, "adult at rest 12-20/min\n(awake or asleep)",
            ha="center", va="top", fontsize=8.5, color=INK_2)
    if rate.spectral_bpm:
        ax.annotate(f"{rate.spectral_bpm:.1f}/min", (rate.spectral_bpm, rate.peak_amplitude_px),
                    xytext=(8, 4), textcoords="offset points", fontweight="bold")
        half = rate.resolution_bpm / 2
        ax.hlines(rate.peak_amplitude_px * 0.5, rate.spectral_bpm - half, rate.spectral_bpm + half,
                  color=INK, lw=1.5)
        ax.text(rate.spectral_bpm + half + 1, rate.peak_amplitude_px * 0.5,
                f"resolution limit ±{half:.1f}/min", va="center", fontsize=8.5, color=INK_2)
    ax.set_title(f"Amplitude spectrum ({len(t) / fps:.1f} s shot; dotted = search band)")
    ax.set_xlabel("cycles per minute"), ax.set_ylabel("amplitude (px)")
    ax.legend(frameon=False, loc="upper right", fontsize=9)

    fig.savefig(out, dpi=130)
    plt.close(fig)


def _temporal_bandpass_fft(stack: np.ndarray, fps: float, band_hz: tuple[float, float]) -> np.ndarray:
    """Ideal band-pass along axis 0, with the end-to-end ramp removed first."""
    n = stack.shape[0]
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None, None]
    x = stack - (stack[0] + ramp * (stack[-1] - stack[0]))
    spec = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(n, 1.0 / fps)
    spec[(f < band_hz[0]) | (f > band_hz[1])] = 0
    return np.fft.irfft(spec, n=n, axis=0).astype(np.float32)


def make_magnified_video(
    video: Path,
    out: Path,
    first_frame: int,
    tracks: Tracks,
    subj: Component,
    rate: RateEstimate,
    roi: Roi,
    fps: float,
    band_hz: tuple[float, float],
    alpha: float,
    width: int = 640,
) -> None:
    """Stabilised original next to a Laplacian-pyramid Eulerian magnification."""
    t_len = len(tracks.transforms)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = min(1.0, width / src_w)
    w, h = int(round(src_w * s)) // 16 * 16, int(round(src_h * s)) // 16 * 16
    sx, sy = w / src_w, h / src_h
    to_small = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1.0]])

    ycc = np.empty((t_len, h, w, 3), np.uint8)
    common = np.ones((h, w), np.float32)
    ones = np.ones((src_h, src_w), np.float32)
    for t in range(t_len):
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("video ended early while rendering")
        m = (to_small @ np.vstack([tracks.transforms[t], [0, 0, 1]]))[:2]
        warped = cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
        ycc[t] = cv2.cvtColor(warped, cv2.COLOR_BGR2YCrCb)
        common = np.minimum(common, cv2.warpAffine(ones, m, (w, h)))
    cap.release()
    # Only magnify where every frame has real pixels, feathered at the edge.
    common = cv2.GaussianBlur(cv2.erode((common > 0.99).astype(np.float32), np.ones((9, 9))), (0, 0), 6)

    luma = ycc[..., 0].astype(np.float32)
    levels = 4
    gauss = [luma]
    for _ in range(levels):
        gauss.append(np.stack([cv2.pyrDown(f) for f in gauss[-1]]))
    delta = np.zeros_like(luma)
    for lvl in range(1, levels):  # skip the finest level: mostly codec noise
        up = np.stack([cv2.pyrUp(f, dstsize=gauss[lvl].shape[:0:-1][:2]) for f in gauss[lvl + 1]])
        filt = _temporal_bandpass_fft(gauss[lvl] - up, fps, band_hz)
        delta += np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) for f in filt])
    delta = np.clip(alpha * delta, -60, 60) * common

    strip_h = 150
    canvas_w, canvas_h = 2 * w, h + strip_h
    sig = subj.band - subj.band.mean()
    span = float(np.abs(sig).max()) or 1.0
    xs = np.linspace(12, canvas_w - 12, t_len)
    ys = h + strip_h / 2 + 14 - sig / span * (strip_h / 2 - 30)
    curve = np.stack([xs, ys], axis=1).astype(np.int32)
    headline = (
        f"{rate.spectral_bpm:.1f} breaths/min" if rate.spectral_bpm
        else f"~{rate.interval_bpm:.0f} breaths/min" if rate.interval_bpm else "no clear rate"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    blue, ink, grey = (214, 120, 42), (240, 239, 236), (150, 150, 150)
    r = (int(roi.x * sx), int(roi.y * sy), int((roi.x + roi.w) * sx), int((roi.y + roi.h) * sy))

    tmp = out.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_w, canvas_h))
    for t in range(t_len):
        left = cv2.cvtColor(ycc[t], cv2.COLOR_YCrCb2BGR)
        mag = ycc[t].copy()
        mag[..., 0] = np.clip(luma[t] + delta[t], 0, 255).astype(np.uint8)
        right = cv2.cvtColor(mag, cv2.COLOR_YCrCb2BGR)
        cv2.rectangle(left, r[:2], r[2:], blue, 1, cv2.LINE_AA)
        for img, label in ((left, "stabilised"), (right, f"motion x{alpha:g}, {band_hz[0]*60:.0f}-{band_hz[1]*60:.0f}/min")):
            cv2.putText(img, label, (10, h - 12), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (10, h - 12), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        canvas = np.full((canvas_h, canvas_w, 3), 26, np.uint8)
        canvas[:h, :w], canvas[:h, w:] = left, right
        cv2.line(canvas, (12, h + strip_h // 2 + 14), (canvas_w - 12, h + strip_h // 2 + 14), (60, 60, 60), 1)
        cv2.polylines(canvas, [curve], False, grey, 1, cv2.LINE_AA)
        cv2.polylines(canvas, [curve[: t + 1]], False, blue, 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(int(v) for v in curve[t]), 5, ink, -1, cv2.LINE_AA)
        cv2.putText(canvas, f"subject motion (px, + up)   {headline}   t={t / fps:4.1f}s",
                    (12, h + 22), font, 0.55, ink, 1, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()

    # mp4v does not play in browsers; transcode to H.264 when ffmpeg is around.
    if shutil.which("ffmpeg"):
        done = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-crf", "18",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
            check=False,
        )
        if done.returncode == 0:
            tmp.unlink()
            return
    tmp.replace(out)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def pick_roi(ref_bgr: np.ndarray) -> Roi:
    try:
        x, y, w, h = cv2.selectROI("Drag a box over the subject's head/shoulders/torso, then ENTER", ref_bgr)
        cv2.destroyAllWindows()
    except cv2.error as exc:
        raise SystemExit("--pick needs a display and GUI OpenCV; use --roi X,Y,W,H instead") from exc
    if w == 0 or h == 0:
        raise SystemExit("no ROI selected")
    return Roi(int(x), int(y), int(w), int(h))


def choose_shot(shots: list[tuple[int, int]], index: int | None) -> tuple[int, int]:
    if index is not None:
        if not 0 <= index < len(shots):
            raise SystemExit(f"--shot must be in 0..{len(shots) - 1}")
        return shots[index]
    return max(shots, key=lambda s: s[1] - s[0])


def interpret(rate: RateEstimate, snr: float, explained: float) -> list[str]:
    lines = []
    best = rate.spectral_bpm or rate.interval_bpm or rate.autocorr_bpm
    if best is None:
        return ["No periodic component could be isolated in the search band."]
    lo, hi = best - rate.resolution_bpm / 2, best + rate.resolution_bpm / 2
    lines.append(
        f"Best estimate {best:.1f} breaths/min; the clip length limits resolution to "
        f"about ±{rate.resolution_bpm / 2:.1f} (roughly {lo:.0f}-{hi:.0f}/min)."
    )
    lines.append(f"Only {rate.cycles_observed:.1f} cycles were observed.")
    if rate.cycles_observed < 5:
        lines.append("Fewer than ~5 cycles: treat the rate as indicative, not as a measurement.")
    others = [v for v in (rate.interval_bpm, rate.autocorr_bpm) if v is not None and v != best]
    if any(abs(v - best) > rate.resolution_bpm for v in others):
        lines.append(
            "The estimators disagree by more than the resolution limit: the motion is irregular "
            "(uneven breath-to-breath intervals, or posture shifts mixed in)."
        )
    if snr < 3:
        lines.append(f"Subject motion is only {snr:.1f}x the static-scene floor: weak evidence of real motion.")
    if explained < 0.4:
        lines.append(f"PC1 explains only {explained:.0%} of in-band variance: the ROI is not moving as one body.")
    lines.append(
        "A rate inside 12-20/min is normal for an adult at rest whether awake or asleep; "
        "respiratory rate alone cannot establish sleep."
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--roi", type=Roi.parse, help="subject box X,Y,W,H in pixels of the shot's first frame")
    ap.add_argument("--pick", action="store_true", help="draw the ROI interactively (needs a display)")
    ap.add_argument("--shot", type=int, help="index of the shot to analyse (default: longest)")
    ap.add_argument("--start", type=float, default=0.0, help="start time, seconds")
    ap.add_argument("--end", type=float, help="end time, seconds")
    ap.add_argument("--band", type=float, nargs=2, default=(6.0, 40.0), metavar=("LO", "HI"),
                    help="search band in breaths/min")
    ap.add_argument("--alpha", type=float, default=10.0, help="motion magnification factor")
    ap.add_argument("--no-video", action="store_true", help="skip the magnified video")
    ap.add_argument("-o", "--out", type=Path, help="output directory (default: <video stem>_breathscope)")
    args = ap.parse_args(argv)

    if not 0 < args.band[0] < args.band[1]:
        ap.error("--band must satisfy 0 < LO < HI")
    out = args.out or args.video.with_name(args.video.stem + "_breathscope")
    out.mkdir(parents=True, exist_ok=True)
    band_hz = (args.band[0] / 60.0, args.band[1] / 60.0)

    gray, fps, offset = read_gray(args.video, args.start, args.end)
    if band_hz[1] >= fps / 2:
        ap.error("--band upper edge is above the video's Nyquist frequency")
    shots = detect_shots(gray)
    a, b = choose_shot(shots, args.shot)
    print(f"{len(gray)} frames @ {fps:.2f} fps; {len(shots)} shot(s): "
          + ", ".join(f"[{i}] {(s + offset) / fps:.2f}-{(e + offset) / fps:.2f}s" for i, (s, e) in enumerate(shots)))
    print(f"analysing shot {shots.index((a, b))}: {(b - a) / fps:.2f} s")
    gray = gray[a:b]
    first = offset + a
    duration = len(gray) / fps
    if duration < 2.5 / band_hz[0] / 4:
        print("warning: shot is very short for this band", file=sys.stderr)

    ref_bgr = read_color_frame(args.video, first)
    roi = pick_roi(ref_bgr) if args.pick else args.roi
    if roi is None:
        grid = out / "reference_frame.png"
        save_reference_grid(ref_bgr, grid)
        print(f"\nNo ROI given. Wrote {grid} with a pixel grid.\n"
              "Re-run with --roi X,Y,W,H around the subject's head, shoulders and torso (or use --pick).")
        return 2
    save_reference_grid(ref_bgr, out / "reference_frame.png", roi)

    tracks = track(gray, roi)
    subj = dominant_component(tracks, tracks.in_roi, fps, band_hz)
    scene = dominant_component(tracks, tracks.static, fps, band_hz)
    rate = estimate_rate(subj, fps, band_hz)
    scene_rate = estimate_rate(scene, fps, band_hz)
    snr = rate.peak_amplitude_px / max(scene_rate.peak_amplitude_px, 1e-9)
    notes = interpret(rate, snr, subj.explained)

    zoom = np.hypot(tracks.transforms[:, 0, 0], tracks.transforms[:, 1, 0])
    result = {
        "video": args.video.name,
        "shot_start_s": round(first / fps, 3),
        "shot_duration_s": round(duration, 3),
        "fps": fps,
        "roi": asdict(roi),
        "band_bpm": list(args.band),
        "subject_points": int(len(subj.idx)),
        "static_scene_points": int(len(scene.idx)),
        "camera_zoom_range": [round(float(1 / zoom.max()), 4), round(float(1 / zoom.min()), 4)],
        "camera_pan_px": round(float(np.ptp(tracks.transforms[:, :, 2], axis=0).max()), 1),
        "pc1_explained_variance": round(subj.explained, 3),
        "subject_peak_amplitude_px": round(rate.peak_amplitude_px, 4),
        "scene_peak_amplitude_px": round(scene_rate.peak_amplitude_px, 4),
        "subject_to_scene_ratio": round(float(snr), 2),
        "rate": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in asdict(rate).items()},
        "notes": notes,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    make_report(out / "report.png", ref_bgr, roi, tracks, subj, scene, rate, fps, band_hz)
    if not args.no_video:
        make_magnified_video(args.video, out / "magnified.mp4", first, tracks, subj, rate, roi, fps, band_hz, args.alpha)

    def fmt(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.1f}"

    print(f"\ntracked {len(subj.idx)} subject points, {len(scene.idx)} static scene points")
    print(f"camera: zoom x{result['camera_zoom_range'][0]}-{result['camera_zoom_range'][1]}, pan {result['camera_pan_px']} px (removed)")
    print(f"rate (breaths/min): spectral {fmt(rate.spectral_bpm)} | intervals {fmt(rate.interval_bpm)} | autocorr {fmt(rate.autocorr_bpm)}")
    print(f"amplitude: subject {rate.peak_amplitude_px:.3f} px vs scene floor {scene_rate.peak_amplitude_px:.3f} px ({snr:.1f}x); "
          f"PC1 {subj.explained:.0%}")
    for line in notes:
        print(" -", line)
    print(f"\noutputs in {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
