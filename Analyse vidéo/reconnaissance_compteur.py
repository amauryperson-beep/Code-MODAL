"""
Extraction automatique d'un compteur 7-segments sur une video.

Pipeline:
1) Lecture de la video image par image
2) Detection de la zone rouge du compteur (automatique ou ROI manuelle)
3) Decoupage en N chiffres (7 par defaut)
4) Reconnaissance de chaque chiffre via un decodeur 7-segments
5) Export CSV + trace valeur en fonction du temps

Exemple:
    python reconnaissance_compteur.py video.mp4 --orientation vertical
"""

from __future__ import annotations

import argparse
import collections
import csv
import difflib
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Le module 'opencv-python' est requis. "
        "Installez-le avec: python3 -m pip install opencv-python"
    ) from exc


SEGMENT_TO_DIGIT: dict[tuple[int, int, int, int, int, int, int], str] = {
    (1, 1, 1, 1, 1, 1, 0): "0",  # a b c d e f
    (0, 1, 1, 0, 0, 0, 0): "1",  # b c
    (1, 1, 0, 1, 1, 0, 1): "2",  # a b d e g
    (1, 1, 1, 1, 0, 0, 1): "3",  # a b c d g
    (0, 1, 1, 0, 0, 1, 1): "4",  # b c f g
    (1, 0, 1, 1, 0, 1, 1): "5",  # a c d f g
    (1, 0, 1, 1, 1, 1, 1): "6",  # a c d e f g
    (1, 1, 1, 0, 0, 0, 0): "7",  # a b c
    (1, 1, 1, 1, 1, 1, 1): "8",  # a b c d e f g
    (1, 1, 1, 1, 0, 1, 1): "9",  # a b c d f g
}


@dataclass(frozen=True)
class ExtractionConfig:
    n_digits: int = 7
    orientation: str = "vertical"
    red_sat_min: int = 70
    red_val_min: int = 70
    segment_threshold: float = 0.10
    min_red_pixels: int = 400
    min_component_pixels: int = 60
    counter_padding: int = 8
    track_roi: bool = True
    track_search_scale: float = 2.5
    roi_smoothing: float = 0.65
    lit_percentile: float = 70.0


@dataclass(frozen=True)
class FrameMeasure:
    frame_index: int
    time_s: float
    value_str: str
    value: float
    uncertain_digits: int


def _normalize_filename(name: str) -> str:
    return unicodedata.normalize("NFKC", name).casefold()


def resolve_video_path(path: Path) -> Path:
    """
    Resolve un chemin de video de maniere robuste:
    - chemin direct
    - equivalence Unicode (NFC/NFD) frequente sur macOS
    - suggestion de nom proche si introuvable
    """
    if path.exists():
        return path

    search_dir = path.parent if str(path.parent) not in {"", "."} else Path(".")
    if not search_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable: {search_dir}")

    target_norm = _normalize_filename(path.name)
    files = [p for p in search_dir.iterdir() if p.is_file()]

    exact_norm = [p for p in files if _normalize_filename(p.name) == target_norm]
    if exact_norm:
        return exact_norm[0]

    stem_norm = _normalize_filename(path.stem)
    suffix_norm = _normalize_filename(path.suffix)
    stem_matches = [
        p
        for p in files
        if _normalize_filename(p.stem) == stem_norm and (not suffix_norm or _normalize_filename(p.suffix) == suffix_norm)
    ]
    if stem_matches:
        return stem_matches[0]

    close = difflib.get_close_matches(path.name, [p.name for p in files], n=3, cutoff=0.6)
    if close:
        raise FileNotFoundError(
            f"Video introuvable: {path}\n"
            f"Suggestions: {', '.join(close)}"
        )
    raise FileNotFoundError(f"Video introuvable: {path}")


def open_video_capture(video_path: Path) -> tuple[cv2.VideoCapture, Path]:
    """
    Ouvre la video avec resolution de chemin robuste + diagnostics.
    """
    resolved = resolve_video_path(video_path)
    cap = cv2.VideoCapture(str(resolved))
    if cap.isOpened():
        return cap, resolved

    # Fallback: URI file:// (parfois utile selon backend OpenCV)
    cap.release()
    cap = cv2.VideoCapture(resolved.resolve().as_uri())
    if cap.isOpened():
        return cap, resolved

    raise RuntimeError(
        "Impossible d'ouvrir la video avec OpenCV.\n"
        f"Chemin fourni: {video_path}\n"
        f"Chemin resolu: {resolved}\n"
        "Verifiez le codec (H.264/H.265), ou convertissez la video en MP4 H.264."
    )


def red_mask(image_bgr: np.ndarray, sat_min: int, val_min: int) -> np.ndarray:
    """Construit un masque binaire des pixels rouges."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, sat_min, val_min], dtype=np.uint8)
    upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([160, sat_min, val_min], dtype=np.uint8)
    upper_red_2 = np.array([179, 255, 255], dtype=np.uint8)

    mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask_1, mask_2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def lit_red_mask(image_bgr: np.ndarray, sat_min: int, val_min: int, percentile_v: float = 78.0) -> np.ndarray:
    """
    Masque des segments vraiment allumes.
    On part du rouge global puis on conserve les pixels rouges les plus lumineux.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    base_red = red_mask(image_bgr, sat_min=sat_min, val_min=val_min)

    red_values = hsv[:, :, 2][base_red > 0].astype(np.float32)
    if red_values.size < 20:
        return base_red

    values = red_values.reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
    compactness, labels, centers = cv2.kmeans(
        values,
        2,
        None,
        criteria,
        4,
        cv2.KMEANS_PP_CENTERS,
    )
    _ = compactness, labels
    c0, c1 = sorted(float(c) for c in centers.flatten())
    if (c1 - c0) >= 12.0:
        thr_v = int(round((c0 + c1) * 0.5))
    else:
        dyn_threshold = int(np.percentile(red_values, np.clip(percentile_v, 50.0, 98.0)))
        thr_v = dyn_threshold

    v = hsv[:, :, 2]
    lit = np.zeros_like(base_red, dtype=np.uint8)
    lit[(base_red > 0) & (v >= max(val_min, thr_v))] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    lit = cv2.morphologyEx(lit, cv2.MORPH_OPEN, kernel, iterations=1)
    lit = cv2.morphologyEx(lit, cv2.MORPH_CLOSE, kernel, iterations=2)
    lit = cv2.dilate(lit, kernel, iterations=1)
    return lit


def clamp_roi(roi: tuple[int, int, int, int], frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """Force la ROI a rester dans les bornes de l'image."""
    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = roi

    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    x2 = max(x + 1, min(x + w, frame_w))
    y2 = max(y + 1, min(y + h, frame_h))
    return x, y, x2 - x, y2 - y


def pad_roi(roi: tuple[int, int, int, int], padding: int, frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = roi
    return clamp_roi((x - padding, y - padding, w + 2 * padding, h + 2 * padding), frame_shape)


def expand_roi(roi: tuple[int, int, int, int], scale: float, frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = roi
    cx = x + w / 2.0
    cy = y + h / 2.0
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    new_x = int(round(cx - (new_w / 2.0)))
    new_y = int(round(cy - (new_h / 2.0)))
    return clamp_roi((new_x, new_y, new_w, new_h), frame_shape)


def candidate_mask(red: np.ndarray, orientation: str) -> np.ndarray:
    """Fusionne les segments rouges voisins pour obtenir des boites candidates."""
    if orientation == "vertical":
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 31))
    else:
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9))

    merged = cv2.dilate(red, kernel_dilate, iterations=1)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    return merged


def roi_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    inter_x0 = max(ax, bx)
    inter_y0 = max(ay, by)
    inter_x1 = min(ax + aw, bx + bw)
    inter_y1 = min(ay + ah, by + bh)
    inter_w = max(0, inter_x1 - inter_x0)
    inter_h = max(0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    union_area = (aw * ah) + (bw * bh) - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area) / float(union_area)


def roi_center(roi: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = roi
    return x + (w / 2.0), y + (h / 2.0)


def score_counter_candidate(
    roi: tuple[int, int, int, int],
    red_pixels: int,
    frame_shape: tuple[int, int, int],
    config: ExtractionConfig,
    previous_roi: tuple[int, int, int, int] | None,
    search_roi: tuple[int, int, int, int] | None,
) -> float:
    x, y, w, h = roi
    if w <= 1 or h <= 1:
        return -1e9

    bbox_area = float(w * h)
    fill = red_pixels / max(1.0, bbox_area)
    elongation = (h / max(1.0, w)) if config.orientation == "vertical" else (w / max(1.0, h))
    frame_area = float(frame_shape[0] * frame_shape[1])
    area_ratio = bbox_area / max(1.0, frame_area)

    # Compteur attendu: assez allonge + densite rouge moderee (7-segments)
    score = 0.0
    score += min(4.0, 1.4 * elongation)
    score += min(3.0, 3.0 * red_pixels / max(1.0, config.min_red_pixels))
    score -= 4.0 * abs(fill - 0.25)
    score -= 10.0 * abs(area_ratio - 0.05)

    if previous_roi is not None:
        cx, cy = roi_center(roi)
        px, py = roi_center(previous_roi)
        diag = math.hypot(frame_shape[1], frame_shape[0])
        dist = math.hypot(cx - px, cy - py) / max(1e-6, diag)
        score += 3.0 * roi_iou(roi, previous_roi)
        score += 1.5 * max(0.0, 1.0 - (4.0 * dist))

    if search_roi is not None:
        sx, sy, sw, sh = search_roi
        margin_x = min(abs(x - sx), abs((sx + sw) - (x + w)))
        margin_y = min(abs(y - sy), abs((sy + sh) - (y + h)))
        if margin_x <= 1 or margin_y <= 1:
            score -= 0.2

    return score


def detect_counter_roi_in_window(
    frame: np.ndarray,
    config: ExtractionConfig,
    search_roi: tuple[int, int, int, int] | None,
    previous_roi: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    frame_h, frame_w = frame.shape[:2]

    if search_roi is None:
        sx, sy, sw, sh = 0, 0, frame_w, frame_h
    else:
        sx, sy, sw, sh = clamp_roi(search_roi, frame.shape)

    search_img = frame[sy : sy + sh, sx : sx + sw]
    red = red_mask(search_img, config.red_sat_min, config.red_val_min)
    if int(np.count_nonzero(red)) < max(20, config.min_component_pixels):
        return None

    merged = candidate_mask(red, config.orientation)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)

    best_score = -1e9
    best_roi: tuple[int, int, int, int] | None = None

    for label in range(1, num_labels):
        bx = int(stats[label, cv2.CC_STAT_LEFT])
        by = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])

        if bw <= 3 or bh <= 3:
            continue

        local_red = red[by : by + bh, bx : bx + bw]
        red_pixels = int(np.count_nonzero(local_red))
        if red_pixels < config.min_component_pixels:
            continue

        gx = sx + bx
        gy = sy + by
        candidate_roi = pad_roi((gx, gy, bw, bh), config.counter_padding, frame.shape)

        score = score_counter_candidate(
            roi=candidate_roi,
            red_pixels=red_pixels,
            frame_shape=frame.shape,
            config=config,
            previous_roi=previous_roi,
            search_roi=search_roi,
        )
        if score > best_score:
            best_score = score
            best_roi = candidate_roi

    return best_roi


def detect_counter_roi(
    frame: np.ndarray,
    config: ExtractionConfig,
    previous_roi: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Detecte automatiquement la ROI du compteur."""
    local_candidate = None
    if previous_roi is not None:
        local_search = expand_roi(previous_roi, config.track_search_scale, frame.shape)
        local_candidate = detect_counter_roi_in_window(
            frame=frame,
            config=config,
            search_roi=local_search,
            previous_roi=previous_roi,
        )
        if local_candidate is not None:
            return local_candidate

    global_candidate = detect_counter_roi_in_window(
        frame=frame,
        config=config,
        search_roi=None,
        previous_roi=previous_roi,
    )
    if global_candidate is not None:
        return global_candidate

    raise RuntimeError(
        "Detection automatique du compteur impossible. "
        "Essayez --roi X Y W H, ou ajustez --red-sat-min / --red-val-min."
    )


def blend_rois(
    previous_roi: tuple[int, int, int, int],
    detected_roi: tuple[int, int, int, int],
    alpha: float,
    frame_shape: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    """Lisse la ROI pour limiter le jitter."""
    ax = float(np.clip(alpha, 0.0, 1.0))
    px, py, pw, ph = previous_roi
    dx, dy, dw, dh = detected_roi

    bx = int(round((1.0 - ax) * px + ax * dx))
    by = int(round((1.0 - ax) * py + ax * dy))
    bw = int(round((1.0 - ax) * pw + ax * dw))
    bh = int(round((1.0 - ax) * ph + ax * dh))
    return clamp_roi((bx, by, bw, bh), frame_shape)


def split_digits(counter_img: np.ndarray, n_digits: int, orientation: str) -> list[np.ndarray]:
    """Decoupe l'image du compteur en sous-images (une par chiffre)."""
    h, w = counter_img.shape[:2]
    digit_images: list[np.ndarray] = []

    if orientation == "vertical":
        boundaries = np.linspace(0, h, n_digits + 1, dtype=int)
        for i in range(n_digits):
            y0, y1 = boundaries[i], boundaries[i + 1]
            y1 = max(y1, y0 + 1)
            digit_images.append(counter_img[y0:y1, :])
    else:
        boundaries = np.linspace(0, w, n_digits + 1, dtype=int)
        for i in range(n_digits):
            x0, x1 = boundaries[i], boundaries[i + 1]
            x1 = max(x1, x0 + 1)
            digit_images.append(counter_img[:, x0:x1])

    return digit_images


def seven_segment_regions(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """
    Renvoie les 7 regions de test dans l'ordre:
    a, b, c, d, e, f, g.
    """
    margin_x = max(1, int(round(width * 0.08)))
    margin_y = max(1, int(round(height * 0.08)))
    seg_h = max(2, int(round(height * 0.16)))
    seg_w = max(2, int(round(width * 0.22)))
    mid_h = max(2, int(round(height * 0.10)))

    top = (margin_x, 0, width - margin_x, seg_h)  # a
    upper_right = (width - seg_w, margin_y, width, (height // 2) - (mid_h // 2))  # b
    lower_right = (width - seg_w, (height // 2) + (mid_h // 2), width, height - margin_y)  # c
    bottom = (margin_x, height - seg_h, width - margin_x, height)  # d
    lower_left = (0, (height // 2) + (mid_h // 2), seg_w, height - margin_y)  # e
    upper_left = (0, margin_y, seg_w, (height // 2) - (mid_h // 2))  # f
    middle = (margin_x, (height // 2) - (mid_h // 2), width - margin_x, (height // 2) + (mid_h // 2))  # g

    regions = [top, upper_right, lower_right, bottom, lower_left, upper_left, middle]
    clipped_regions: list[tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in regions:
        clipped_regions.append(
            (
                max(0, min(x0, width - 1)),
                max(0, min(y0, height - 1)),
                max(1, min(x1, width)),
                max(1, min(y1, height)),
            )
        )
    return clipped_regions


def fill_ratio(mask: np.ndarray, rect: tuple[int, int, int, int]) -> float:
    """Taux de pixels actifs dans un rectangle."""
    x0, y0, x1, y1 = rect
    if x1 <= x0 or y1 <= y0:
        return 0.0
    region = mask[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0
    return float(np.count_nonzero(region)) / float(region.size)


def decode_digit(digit_img: np.ndarray, config: ExtractionConfig) -> tuple[str, int]:
    """
    Decode un chiffre 7-segments.
    Retourne (caractere, distance_hamming_au_pattern_le_plus_proche).
    """
    mask_lit = lit_red_mask(
        digit_img,
        config.red_sat_min,
        config.red_val_min,
        percentile_v=config.lit_percentile,
    )
    mask_base = red_mask(
        digit_img,
        max(8, config.red_sat_min - 45),
        max(8, config.red_val_min - 50),
    )

    # Fallback si le seuil lumineux est trop strict.
    if np.count_nonzero(mask_lit) < 8:
        hsv = cv2.cvtColor(digit_img, cv2.COLOR_BGR2HSV)
        red_pixels = hsv[:, :, 2][mask_base > 0]
        if red_pixels.size > 0:
            dynamic_v = int(np.percentile(red_pixels, 60))
            mask_lit = red_mask(
                digit_img,
                max(25, config.red_sat_min - 10),
                max(20, dynamic_v),
            )
        else:
            mask_lit = mask_base

    h, w = mask_lit.shape
    if h < 4 or w < 4:
        return "?", 7
    regions = seven_segment_regions(w, h)

    states: list[int] = []
    for rect in regions:
        lit_ratio = fill_ratio(mask_lit, rect)
        base_ratio = fill_ratio(mask_base, rect)
        score = (0.75 * lit_ratio) + (0.25 * base_ratio)
        states.append(1 if score >= config.segment_threshold else 0)

    pattern = tuple(states)
    if pattern in SEGMENT_TO_DIGIT:
        return SEGMENT_TO_DIGIT[pattern], 0

    # Fallback: pattern le plus proche (utile en cas de bruit / segment partiel)
    best_digit = "?"
    best_distance = 7
    for known_pattern, known_digit in SEGMENT_TO_DIGIT.items():
        distance = sum(int(a != b) for a, b in zip(pattern, known_pattern))
        if distance < best_distance:
            best_distance = distance
            best_digit = known_digit

    return best_digit, best_distance


def decode_counter_crop(counter_crop: np.ndarray, config: ExtractionConfig) -> tuple[str, int]:
    """
    Decode directement une ROI compteur (7 chiffres).
    Retourne (value_str, nb_chiffres_incertain).
    """
    digits = split_digits(counter_crop, config.n_digits, config.orientation)
    chars: list[str] = []
    uncertain = 0
    for digit_img in digits:
        char, distance = decode_digit(digit_img, config)
        chars.append(char)
        if distance > 0 or char == "?":
            uncertain += 1
    return "".join(chars), uncertain


def _digit_change_count(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(int(x != y) for x, y in zip(a, b))


def auto_tune_config(
    video_path: Path,
    config: ExtractionConfig,
    initial_roi: tuple[int, int, int, int] | None,
    frame_step: int,
    fallback_fps: float,
    max_samples: int = 80,
) -> ExtractionConfig:
    """
    Choisit automatiquement des hyperparametres robustes sur un sous-ensemble de frames.
    """
    cap, resolved = open_video_capture(video_path)
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        return config

    if initial_roi is None:
        try:
            roi = detect_counter_roi(first_frame, config, previous_roi=None)
        except RuntimeError:
            cap.release()
            return config
    else:
        roi = clamp_roi(initial_roi, first_frame.shape)

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = fallback_fps

    candidate_segment_thresholds = sorted(
        {
            round(float(np.clip(config.segment_threshold + delta, 0.05, 0.35)), 3)
            for delta in (-0.04, -0.02, 0.0, 0.02, 0.04)
        }
    )
    candidate_lit_percentiles = sorted(
        {
            float(np.clip(config.lit_percentile + delta, 50.0, 95.0))
            for delta in (-15.0, -8.0, 0.0, 8.0, 15.0)
        }
    )

    sample_interval = max(frame_step, int(max(1.0, fps // 2.0)))
    candidate_configs: list[ExtractionConfig] = []
    for seg_thr in candidate_segment_thresholds:
        for lit_pct in candidate_lit_percentiles:
            candidate_configs.append(
                ExtractionConfig(
                    n_digits=config.n_digits,
                    orientation=config.orientation,
                    red_sat_min=config.red_sat_min,
                    red_val_min=config.red_val_min,
                    segment_threshold=seg_thr,
                    min_red_pixels=config.min_red_pixels,
                    min_component_pixels=config.min_component_pixels,
                    counter_padding=config.counter_padding,
                    track_roi=False,
                    track_search_scale=config.track_search_scale,
                    roi_smoothing=config.roi_smoothing,
                    lit_percentile=lit_pct,
                )
            )

    stats = [
        {
            "valid": 0,
            "unknown": 0,
            "uncertain": 0,
            "samples": 0,
            "changes": 0,
            "prev": None,
        }
        for _ in candidate_configs
    ]

    frame_idx = 0
    samples = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while samples < max_samples:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_idx % sample_interval != 0:
            frame_idx += 1
            continue
        frame_idx += 1
        samples += 1

        x, y, w, h = roi
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            continue

        for i, candidate in enumerate(candidate_configs):
            value_str, uncertain = decode_counter_crop(crop, candidate)
            entry = stats[i]
            entry["samples"] += 1
            entry["uncertain"] += uncertain
            entry["unknown"] += value_str.count("?")
            if "?" not in value_str:
                entry["valid"] += 1
            prev = entry["prev"]
            if prev is not None:
                entry["changes"] += _digit_change_count(prev, value_str)
            entry["prev"] = value_str

    cap.release()
    if samples == 0:
        return config

    best_score = -1e18
    best_cfg = config
    for candidate, entry in zip(candidate_configs, stats):
        n = max(1, int(entry["samples"]))
        valid_ratio = float(entry["valid"]) / n
        unknown_ratio = float(entry["unknown"]) / max(1, n * config.n_digits)
        uncertain_ratio = float(entry["uncertain"]) / max(1, n * config.n_digits)
        change_rate = float(entry["changes"]) / max(1, n - 1)

        # Score empirique: maximiser la validite, minimiser inconnus/bruit temporel.
        score = (
            (4.0 * valid_ratio)
            - (3.0 * unknown_ratio)
            - (1.2 * uncertain_ratio)
            - (0.04 * change_rate)
        )
        if score > best_score:
            best_score = score
            best_cfg = candidate

    print(
        "Auto-tune: "
        f"segment_threshold={best_cfg.segment_threshold:.3f}, "
        f"lit_percentile={best_cfg.lit_percentile:.1f}, "
        f"video={resolved.name}"
    )
    return best_cfg


def extract_series(
    video_path: Path,
    config: ExtractionConfig,
    roi: tuple[int, int, int, int] | None,
    frame_step: int,
    fallback_fps: float,
    debug_frame_path: Path | None,
    debug_digits_dir: Path | None,
) -> tuple[list[FrameMeasure], tuple[int, int, int, int], float, int]:
    """Extrait la serie temporelle du compteur depuis la video."""
    cap, resolved_video_path = open_video_capture(video_path)

    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise RuntimeError("La video est vide ou illisible.")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = fallback_fps

    if roi is None:
        current_roi = detect_counter_roi(first_frame, config, previous_roi=None)
    else:
        current_roi = clamp_roi(roi, first_frame.shape)

    if debug_frame_path is not None:
        debug = first_frame.copy()
        x, y, w, h = current_roi
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if config.orientation == "vertical":
            boundaries = np.linspace(0, h, config.n_digits + 1, dtype=int)
            for b in boundaries[1:-1]:
                yy = y + int(b)
                cv2.line(debug, (x, yy), (x + w, yy), (255, 0, 0), 1)
        else:
            boundaries = np.linspace(0, w, config.n_digits + 1, dtype=int)
            for b in boundaries[1:-1]:
                xx = x + int(b)
                cv2.line(debug, (xx, y), (xx, y + h), (255, 0, 0), 1)

        cv2.imwrite(str(debug_frame_path), debug)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    measures: list[FrameMeasure] = []
    total_frames = 0
    debug_digits_written = False

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        total_frames += 1
        frame_idx = total_frames - 1
        if frame_idx % frame_step != 0:
            continue

        if config.track_roi:
            try:
                detected_roi = detect_counter_roi(frame, config, previous_roi=current_roi)
                current_roi = blend_rois(
                    previous_roi=current_roi,
                    detected_roi=detected_roi,
                    alpha=config.roi_smoothing,
                    frame_shape=frame.shape,
                )
            except RuntimeError:
                # Si la detection rate ponctuellement, on garde la derniere ROI stable.
                pass

        x, y, w, h = current_roi
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            continue

        digits = split_digits(crop, config.n_digits, config.orientation)

        if debug_digits_dir is not None and not debug_digits_written:
            debug_digits_dir.mkdir(parents=True, exist_ok=True)
            for i, digit_img in enumerate(digits):
                raw_path = debug_digits_dir / f"digit_{i:02d}_raw.png"
                lit_path = debug_digits_dir / f"digit_{i:02d}_litmask.png"
                cv2.imwrite(str(raw_path), digit_img)
                cv2.imwrite(
                    str(lit_path),
                    lit_red_mask(
                        digit_img,
                        sat_min=config.red_sat_min,
                        val_min=config.red_val_min,
                        percentile_v=config.lit_percentile,
                    ),
                )
            debug_digits_written = True

        value_str, uncertain = decode_counter_crop(crop, config)
        if value_str.isdigit():
            value = float(int(value_str))
        else:
            value = math.nan

        measures.append(
            FrameMeasure(
                frame_index=frame_idx,
                time_s=frame_idx / fps,
                value_str=value_str,
                value=value,
                uncertain_digits=uncertain,
            )
        )

    cap.release()
    if total_frames == 0:
        raise RuntimeError(f"Aucune frame lue dans la video: {resolved_video_path}")
    return measures, current_roi, fps, total_frames


def smooth_measures(
    measures: list[FrameMeasure],
    n_digits: int,
    window: int,
    enforce_monotonic: bool,
) -> list[FrameMeasure]:
    """
    Stabilise la sequence lue:
    - vote majoritaire local par position de chiffre
    - option monotone non-decroissante sur la valeur complete
    """
    if window <= 0 or not measures:
        return measures

    half = window // 2
    values = [m.value_str for m in measures]
    smoothed_strings: list[str] = []

    for i in range(len(values)):
        chars: list[str] = []
        for j in range(n_digits):
            votes: collections.Counter[str] = collections.Counter()
            for k in range(max(0, i - half), min(len(values), i + half + 1)):
                if j >= len(values[k]):
                    continue
                c = values[k][j]
                if c.isdigit():
                    votes[c] += 1
            chars.append(votes.most_common(1)[0][0] if votes else values[i][j])
        smoothed_strings.append("".join(chars))

    if enforce_monotonic:
        running_max = -1
        for i, txt in enumerate(smoothed_strings):
            if txt.isdigit():
                val = int(txt)
                if val < running_max:
                    smoothed_strings[i] = f"{running_max:0{n_digits}d}"[-n_digits:]
                else:
                    running_max = val

    output: list[FrameMeasure] = []
    for m, txt in zip(measures, smoothed_strings):
        if txt.isdigit():
            value = float(int(txt))
            uncertain = m.uncertain_digits
        else:
            value = math.nan
            uncertain = max(m.uncertain_digits, txt.count("?"))
        output.append(
            FrameMeasure(
                frame_index=m.frame_index,
                time_s=m.time_s,
                value_str=txt,
                value=value,
                uncertain_digits=uncertain,
            )
        )
    return output


def apply_first_value_calibration(
    measures: list[FrameMeasure],
    known_first_value: str,
    n_digits: int,
) -> list[FrameMeasure]:
    """
    Calibre un remappage symbolique par position a partir de la 1ere valeur connue.
    Utile si certains segments sont systematiquement confondus (ex: 3 lu a la place de 0).
    """
    if not measures:
        return measures
    if len(known_first_value) != n_digits or not known_first_value.isdigit():
        raise ValueError("--known-first-value doit contenir exactement n_digits chiffres.")

    first_pred = measures[0].value_str
    mapping_by_pos: list[dict[str, str]] = []
    for j in range(n_digits):
        src = first_pred[j] if j < len(first_pred) else "?"
        dst = known_first_value[j]
        mapping_by_pos.append({src: dst})

    calibrated: list[FrameMeasure] = []
    for m in measures:
        chars = list(m.value_str)
        out_chars: list[str] = []
        for j in range(n_digits):
            c = chars[j] if j < len(chars) else "?"
            out_chars.append(mapping_by_pos[j].get(c, c))
        txt = "".join(out_chars)
        value = float(int(txt)) if txt.isdigit() else math.nan
        calibrated.append(
            FrameMeasure(
                frame_index=m.frame_index,
                time_s=m.time_s,
                value_str=txt,
                value=value,
                uncertain_digits=m.uncertain_digits,
            )
        )
    return calibrated


def export_csv(path: Path, measures: list[FrameMeasure]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "time_s", "value_str", "value", "uncertain_digits"])
        for m in measures:
            writer.writerow(
                [
                    m.frame_index,
                    f"{m.time_s:.6f}",
                    m.value_str,
                    "" if math.isnan(m.value) else f"{m.value:.0f}",
                    m.uncertain_digits,
                ]
            )


def export_plot(path: Path, measures: list[FrameMeasure], show_plot: bool) -> None:
    valid = [m for m in measures if not math.isnan(m.value)]
    if not valid:
        print("Avertissement: aucun point numerique valide pour la courbe.")
        return

    times = np.array([m.time_s for m in valid], dtype=float)
    values = np.array([m.value for m in valid], dtype=float)

    plt.figure(figsize=(10, 4))
    plt.step(times, values, where="post")
    plt.xlabel("Temps (s)")
    plt.ylabel("Compteur")
    plt.title("Evolution temporelle du compteur")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    if show_plot:
        plt.show()
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconnaissance d'un compteur rouge 7-segments dans une video.",
    )
    parser.add_argument("video", type=Path, help="Chemin de la video a analyser")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("compteur_temps.csv"),
        help="Fichier CSV de sortie",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("compteur_temps.png"),
        help="Image PNG de la courbe compteur(t)",
    )
    parser.add_argument(
        "--debug-frame",
        type=Path,
        default=Path("debug_detection.png"),
        help="Image de debug (ROI + decoupage) sur la premiere frame",
    )
    parser.add_argument(
        "--debug-digits-dir",
        type=Path,
        default=None,
        help="Dossier optionnel pour exporter les 7 sous-images et leurs masques lumineux",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="ROI manuelle en pixels",
    )
    parser.add_argument(
        "--orientation",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="Orientation des 7 chiffres dans la ROI",
    )
    parser.add_argument(
        "--n-digits",
        type=int,
        default=7,
        help="Nombre de chiffres a decouper dans la ROI",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Traiter une frame sur N",
    )
    parser.add_argument(
        "--segment-threshold",
        type=float,
        default=0.10,
        help="Seuil d'activation d'un segment (0-1)",
    )
    parser.add_argument(
        "--red-sat-min",
        type=int,
        default=70,
        help="Saturation minimale pour detecter le rouge",
    )
    parser.add_argument(
        "--red-val-min",
        type=int,
        default=70,
        help="Valeur (luminosite) minimale pour detecter le rouge",
    )
    parser.add_argument(
        "--lit-percentile",
        type=float,
        default=70.0,
        help="Percentile de luminosite pour isoler les segments allumes (50-98)",
    )
    parser.add_argument(
        "--min-red-pixels",
        type=int,
        default=400,
        help="Nb minimal de pixels rouges pour valider la detection auto de ROI",
    )
    parser.add_argument(
        "--min-component-pixels",
        type=int,
        default=60,
        help="Nb minimal de pixels rouges pour conserver un composant candidat",
    )
    parser.add_argument(
        "--counter-padding",
        type=int,
        default=8,
        help="Padding ajoute autour de la ROI detectee",
    )
    parser.add_argument(
        "--no-track-roi",
        action="store_true",
        help="Desactive le suivi dynamique de la ROI sur toute la video",
    )
    parser.add_argument(
        "--track-search-scale",
        type=float,
        default=2.5,
        help="Taille de la fenetre locale de recherche autour de la ROI precedente",
    )
    parser.add_argument(
        "--roi-smoothing",
        type=float,
        default=0.65,
        help="Lissage de ROI (0 = pas de MAJ, 1 = ROI detectee brute)",
    )
    parser.add_argument(
        "--fallback-fps",
        type=float,
        default=25.0,
        help="FPS utilise si les metadonnees video sont absentes",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Affiche la figure matplotlib en plus de l'export PNG",
    )
    parser.add_argument(
        "--no-auto-tune",
        action="store_true",
        help="Desactive l'ajustement automatique des seuils sur un echantillon de frames",
    )
    parser.add_argument(
        "--auto-tune-max-samples",
        type=int,
        default=80,
        help="Nb maximal de frames echantillonnees pour l'auto-tune",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Taille de la fenetre de lissage temporel (0 = desactive)",
    )
    parser.add_argument(
        "--enforce-monotonic",
        action="store_true",
        help="Force une evolution non-decroissante de la valeur lue apres lissage",
    )
    parser.add_argument(
        "--known-first-value",
        type=str,
        default=None,
        help="Valeur reelle de la 1ere frame analysee (ex: 0008888) pour calibration rapide",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.n_digits <= 0:
        parser.error("--n-digits doit etre > 0")
    if args.frame_step <= 0:
        parser.error("--frame-step doit etre > 0")
    if not (0.0 < args.segment_threshold < 1.0):
        parser.error("--segment-threshold doit etre dans ]0, 1[")
    if args.min_component_pixels <= 0:
        parser.error("--min-component-pixels doit etre > 0")
    if args.counter_padding < 0:
        parser.error("--counter-padding doit etre >= 0")
    if args.track_search_scale <= 1.0:
        parser.error("--track-search-scale doit etre > 1.0")
    if not (0.0 <= args.roi_smoothing <= 1.0):
        parser.error("--roi-smoothing doit etre dans [0, 1]")
    if not (50.0 <= args.lit_percentile <= 98.0):
        parser.error("--lit-percentile doit etre dans [50, 98]")
    if args.auto_tune_max_samples <= 0:
        parser.error("--auto-tune-max-samples doit etre > 0")
    if args.smooth_window < 0:
        parser.error("--smooth-window doit etre >= 0")
    if args.known_first_value is not None:
        if len(args.known_first_value) != args.n_digits or not args.known_first_value.isdigit():
            parser.error("--known-first-value doit contenir exactement n_digits chiffres")

    config = ExtractionConfig(
        n_digits=args.n_digits,
        orientation=args.orientation,
        red_sat_min=args.red_sat_min,
        red_val_min=args.red_val_min,
        segment_threshold=args.segment_threshold,
        min_red_pixels=args.min_red_pixels,
        min_component_pixels=args.min_component_pixels,
        counter_padding=args.counter_padding,
        track_roi=not args.no_track_roi,
        track_search_scale=args.track_search_scale,
        roi_smoothing=args.roi_smoothing,
        lit_percentile=args.lit_percentile,
    )

    roi = tuple(args.roi) if args.roi is not None else None
    if not args.no_auto_tune:
        config = auto_tune_config(
            video_path=args.video,
            config=config,
            initial_roi=roi,
            frame_step=args.frame_step,
            fallback_fps=args.fallback_fps,
            max_samples=args.auto_tune_max_samples,
        )

    measures, roi_used, fps, total_frames = extract_series(
        video_path=args.video,
        config=config,
        roi=roi,
        frame_step=args.frame_step,
        fallback_fps=args.fallback_fps,
        debug_frame_path=args.debug_frame,
        debug_digits_dir=args.debug_digits_dir,
    )

    measures = smooth_measures(
        measures=measures,
        n_digits=config.n_digits,
        window=args.smooth_window,
        enforce_monotonic=args.enforce_monotonic,
    )
    if args.known_first_value is not None:
        measures = apply_first_value_calibration(measures, args.known_first_value, config.n_digits)

    export_csv(args.output_csv, measures)
    export_plot(args.output_plot, measures, show_plot=args.show)

    valid_points = sum(0 if math.isnan(m.value) else 1 for m in measures)
    uncertain_ratio = 100.0 * sum(m.uncertain_digits > 0 for m in measures) / max(1, len(measures))

    print(f"Video analysee: {args.video}")
    print(f"Frames lues: {total_frames}, frames traitees: {len(measures)}, fps: {fps:.3f}")
    print(f"ROI utilisee (x, y, w, h): {roi_used}")
    print(f"Points numeriques valides: {valid_points}/{len(measures)}")
    print(f"Frames avec au moins 1 chiffre incertain: {uncertain_ratio:.1f}%")
    print(f"CSV: {args.output_csv}")
    print(f"Courbe: {args.output_plot}")
    print(f"Debug frame: {args.debug_frame}")
    if args.debug_digits_dir is not None:
        print(f"Debug digits: {args.debug_digits_dir}")
    print(f"Lissage: window={args.smooth_window}, monotone={args.enforce_monotonic}")
    if args.known_first_value is not None:
        print(f"Calibration 1ere valeur: {args.known_first_value}")


if __name__ == "__main__":
    main()
