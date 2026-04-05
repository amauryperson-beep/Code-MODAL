"""
Generation d'une video synthétique de compteur 7-segments (7 chiffres verticaux).

Permet de tester la robustesse de reconnaissance_compteur.py avec une verite terrain.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


SEGMENTS_BY_DIGIT: dict[int, tuple[int, int, int, int, int, int, int]] = {
    0: (1, 1, 1, 1, 1, 1, 0),
    1: (0, 1, 1, 0, 0, 0, 0),
    2: (1, 1, 0, 1, 1, 0, 1),
    3: (1, 1, 1, 1, 0, 0, 1),
    4: (0, 1, 1, 0, 0, 1, 1),
    5: (1, 0, 1, 1, 0, 1, 1),
    6: (1, 0, 1, 1, 1, 1, 1),
    7: (1, 1, 1, 0, 0, 0, 0),
    8: (1, 1, 1, 1, 1, 1, 1),
    9: (1, 1, 1, 1, 0, 1, 1),
}


def draw_digit_7seg(canvas: np.ndarray, x: int, y: int, w: int, h: int, digit: int) -> None:
    on = SEGMENTS_BY_DIGIT[digit]
    t = max(2, int(round(min(w, h) * 0.15)))

    # a, b, c, d, e, f, g
    segments = [
        ((x + int(0.2 * w), y + int(0.05 * h)), (x + int(0.8 * w), y + int(0.05 * h))),  # a
        ((x + int(0.82 * w), y + int(0.16 * h)), (x + int(0.82 * w), y + int(0.48 * h))),  # b
        ((x + int(0.82 * w), y + int(0.52 * h)), (x + int(0.82 * w), y + int(0.84 * h))),  # c
        ((x + int(0.2 * w), y + int(0.95 * h)), (x + int(0.8 * w), y + int(0.95 * h))),  # d
        ((x + int(0.18 * w), y + int(0.52 * h)), (x + int(0.18 * w), y + int(0.84 * h))),  # e
        ((x + int(0.18 * w), y + int(0.16 * h)), (x + int(0.18 * w), y + int(0.48 * h))),  # f
        ((x + int(0.2 * w), y + int(0.50 * h)), (x + int(0.8 * w), y + int(0.50 * h))),  # g
    ]

    for active, (p0, p1) in zip(on, segments):
        if active:
            cv2.line(canvas, p0, p1, (10, 10, 240), t, lineType=cv2.LINE_AA)
        else:
            cv2.line(canvas, p0, p1, (12, 12, 40), max(1, t // 2), lineType=cv2.LINE_AA)


def make_frame(width: int, height: int, value_str: str, jitter_xy: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    frame = np.full((height, width, 3), 218, dtype=np.uint8)

    # fond bruité léger
    noise = np.random.normal(0, 7, frame.shape).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # quelques distractions rouges
    cv2.circle(frame, (int(width * 0.75), int(height * 0.25)), 16, (20, 20, 180), -1)
    cv2.rectangle(
        frame,
        (int(width * 0.68), int(height * 0.72)),
        (int(width * 0.95), int(height * 0.80)),
        (15, 15, 130),
        -1,
    )

    panel_w = 86
    panel_h = 410
    x0 = int(width * 0.36) + jitter_xy[0]
    y0 = int(height * 0.14) + jitter_xy[1]
    x1, y1 = x0 + panel_w, y0 + panel_h

    cv2.rectangle(frame, (x0, y0), (x1, y1), (34, 34, 34), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (120, 120, 120), 2)

    digit_h = panel_h // 7
    for i, char in enumerate(value_str):
        draw_digit_7seg(frame, x0 + 12, y0 + i * digit_h + 6, panel_w - 24, digit_h - 12, int(char))

    # bloom global modéré
    blur = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.5, sigmaY=1.5)
    frame = cv2.addWeighted(frame, 0.85, blur, 0.35, 0.0)
    return frame, (x0, y0, panel_w, panel_h)


def generate_video(
    output_video: Path,
    output_truth_csv: Path,
    fps: float,
    n_frames: int,
    start_value: int,
    increment_every: int,
) -> None:
    width, height = 720, 960
    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_truth_csv.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Impossible de creer la video: {output_video}")

    value = int(start_value)
    rows: list[tuple[int, float, str]] = []
    for frame_idx in range(n_frames):
        if frame_idx > 0 and frame_idx % increment_every == 0:
            value += 1
        value_str = f"{value:07d}"[-7:]
        jitter = (int(np.random.randint(-2, 3)), int(np.random.randint(-2, 3)))
        frame, _ = make_frame(width, height, value_str, jitter)
        writer.write(frame)
        rows.append((frame_idx, frame_idx / fps, value_str))

    writer.release()

    with output_truth_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "time_s", "value_str"])
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genere une video synthétique de compteur 7-segments.")
    parser.add_argument("--output-video", type=Path, default=Path("synthetic_counter.mp4"))
    parser.add_argument("--output-truth-csv", type=Path, default=Path("synthetic_counter_truth.csv"))
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--n-frames", type=int, default=450)
    parser.add_argument("--start-value", type=int, default=1234560)
    parser.add_argument("--increment-every", type=int, default=15)
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps doit etre > 0")
    if args.n_frames <= 0:
        parser.error("--n-frames doit etre > 0")
    if args.increment_every <= 0:
        parser.error("--increment-every doit etre > 0")

    generate_video(
        output_video=args.output_video,
        output_truth_csv=args.output_truth_csv,
        fps=args.fps,
        n_frames=args.n_frames,
        start_value=args.start_value,
        increment_every=args.increment_every,
    )
    print(f"Video synthétique: {args.output_video}")
    print(f"Verite terrain: {args.output_truth_csv}")


if __name__ == "__main__":
    main()
