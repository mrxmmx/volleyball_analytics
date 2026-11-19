import cv2
cap = cv2.VideoCapture("src/output/tokyo2020-poland-vs-iran.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter("baseline_clip.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
for _ in range(int(fps * 15)):
    ok, f = cap.read()
    if not ok:
        break
    out.write(f)
cap.release()
out.release()