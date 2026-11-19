"""Compare detection settings on the baseline clip (no video output)."""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from demo import court_polygon, players_inside_court  # noqa: E402
from ml_manager.ml_manager import MLManager  # noqa: E402
from ml_manager.utils.tools import det2supervision  # noqa: E402
from vb_gui.calibration.database import CalibrationDB  # noqa: E402

CLIP = ROOT / "baseline_clip.mp4"
DB_PATH = ROOT / "db" / "calibration.db"
STEP = 5                              # use every 5th frame
CONF_VALUES = [0.25, 0.35, 0.5, 0.65]
MARGINS = [0, 30, 60, 100]


def load_court():
    db = CalibrationDB(db_path=str(DB_PATH))
    names = [n for n in db.list_videos() if "baseline_clip" in n]
    if not names:
        sys.exit("No saved calibration for baseline_clip. Run `python src/demo.py` once and calibrate it.")
    data = db.load(names[0])
    return court_polygon(data["court_corners"])


def main():
    court_poly = court_polygon_value = load_court()
    ml = MLManager()
    cap = cv2.VideoCapture(str(CLIP))

    stats = {c: {"frames": 0, "ball": 0, "actions": 0, "players_all": 0,
                 "players_in": {m: 0 for m in MARGINS}} for c in CONF_VALUES}

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % STEP:
            continue

        for conf in CONF_VALUES:
            actions, ball, players = ml.detect_all(frame, conf_threshold=conf, iou_threshold=0.45)
            s = stats[conf]
            s["frames"] += 1
            s["ball"] += 1 if ball else 0
            s["actions"] += len(det2supervision(detections=actions))
            p = det2supervision(detections=players)
            s["players_all"] += len(p)
            if len(p) > 0:
                for m in MARGINS:
                    keep = players_inside_court(p.xyxy, court_poly, margin=m)
                    s["players_in"][m] += int(keep.sum())
    cap.release()

    print("\nPer-frame averages (sampled frames)\n")
    header = f"{'conf':>5} | {'ball rate':>9} | {'actions':>7} | {'players all':>11} | " + " | ".join(f"in m={m:<3}" for m in MARGINS)
    print(header)
    print("-" * len(header))
    for conf in CONF_VALUES:
        s = stats[conf]
        n = max(s["frames"], 1)
        row = (f"{conf:>5} | {s['ball']/n:>9.2f} | {s['actions']/n:>7.2f} | {s['players_all']/n:>11.2f} | "
               + " | ".join(f"{s['players_in'][m]/n:>7.2f}" for m in MARGINS))
        print(row)
    print(f"\nSampled frames: {stats[CONF_VALUES[0]]['frames']}")


if __name__ == "__main__":
    main()