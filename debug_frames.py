"""Save a few annotated frames: kept players green, removed players red."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from demo import players_inside_court  # noqa: E402
from tune_detection import load_court  # noqa: E402
from ml_manager.ml_manager import MLManager  # noqa: E402
from ml_manager.utils.tools import det2supervision  # noqa: E402

CLIP = ROOT / "baseline_clip.mp4"
OUT = ROOT / "output" / "debug"
FRAMES = [60, 200, 350, 500, 650]
CONF, MARGIN = 0.25, 60


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    court_poly = load_court()
    ml = MLManager()
    cap = cv2.VideoCapture(str(CLIP))

    for idx in FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        _, ball, players = ml.detect_all(frame, conf_threshold=CONF, iou_threshold=0.45)
        p = det2supervision(detections=players)

        cv2.polylines(frame, [court_poly.astype(np.int32)], True, (255, 255, 0), 2)
        if len(p) > 0:
            keep = players_inside_court(p.xyxy, court_poly, margin=MARGIN)
            for box, k in zip(p.xyxy, keep):
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0) if k else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if ball:
            cx, cy = map(int, ball.bbox.center)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 255), 2)

        cv2.imwrite(str(OUT / f"frame_{idx}.jpg"), frame)
        print("saved", OUT / f"frame_{idx}.jpg")
    cap.release()


if __name__ == "__main__":
    main()
    