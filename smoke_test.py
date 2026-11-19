"""Smoke test: run the pipeline on the baseline clip and verify the outputs."""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from demo import run_object_detection, run_video_classification  # noqa: E402
from ml_manager.ml_manager import MLManager  # noqa: E402
from vb_gui.calibration.database import CalibrationDB  # noqa: E402

CLIP = ROOT / "baseline_clip.mp4"
OUT_DIR = ROOT / "output"
DB_PATH = ROOT / "db" / "calibration.db"

results = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} {detail}")
    results.append((name, ok))
    return ok


def skip(name, reason):
    print(f"[SKIP] {name}: {reason}")


def frame_count(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def verify_output(name, path, expected_frames):
    exists = Path(path).exists()
    if not check(f"{name}: output file exists", exists, str(path)):
        return
    size_kb = Path(path).stat().st_size / 1024
    check(f"{name}: file size > 100 KB", size_kb > 100, f"({size_kb:.0f} KB)")
    n = frame_count(path)
    check(
        f"{name}: frame count matches input",
        abs(n - expected_frames) <= 1,
        f"(got {n}, expected {expected_frames})",
    )


def main():
    if not check("baseline clip exists", CLIP.exists(), str(CLIP)):
        sys.exit(1)
    expected = frame_count(CLIP)
    print(f"Input frames: {expected}\n")

    OUT_DIR.mkdir(exist_ok=True)
    ml_manager = MLManager()

    # 1) Video classification
    cls_out = OUT_DIR / "smoke_classification.mp4"
    run_video_classification(ml_manager, str(CLIP), str(cls_out))
    verify_output("classification", cls_out, expected)

    # 2) Object detection (needs a saved court calibration)
    calibration = None
    if DB_PATH.exists():
        db = CalibrationDB(db_path=str(DB_PATH))
        names = [n for n in db.list_videos() if "baseline_clip" in n]
        if names:
            calibration = db.load(names[0])

    if calibration is None:
        skip("object detection", "no saved calibration for baseline_clip. Run `python src/demo.py` once and calibrate it.")
    elif "court_corners" not in calibration:
        check("calibration has court_corners", False, f"(keys: {list(calibration.keys())})")
    else:
        det_out = OUT_DIR / "smoke_object_detection.mp4"
        run_object_detection(ml_manager, str(CLIP), str(det_out), calibration)
        verify_output("object detection", det_out, expected)

    failed = [name for name, ok in results if not ok]
    print("\n" + "=" * 50)
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()