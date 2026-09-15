"""Live AprilTag wrist-tag diagnostic. Hold the printed tag in front of the RealSense."""
import sys, os, time
sys.path.insert(0, os.path.expanduser("~/v2s2r_isaaclab"))
import numpy as np, cv2
from sim_teleop.live_teleop import RealSenseSource

OUT = "/tmp/claude-1202/-home-jianuoqiu-v2s2r-isaaclab/71e0e212-69d8-4e97-9b1d-40c6254efda6/scratchpad"
dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
pA = cv2.aruco.DetectorParameters(); pA.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
detA = cv2.aruco.ArucoDetector(dic, pA)                       # teleop's exact config
pB = cv2.aruco.DetectorParameters()                          # relaxed
pB.adaptiveThreshWinSizeMin = 3; pB.adaptiveThreshWinSizeMax = 53; pB.adaptiveThreshWinSizeStep = 8
pB.minMarkerPerimeterRate = 0.01
detB = cv2.aruco.ArucoDetector(dic, pB)

print("\n>>> HOLD THE PRINTED TAG FLAT, FACING THE CAMERA, ~30-50 cm AWAY, and move it slowly <<<\n")
src = RealSenseSource(fps=30)
time.sleep(1.0)
print("intrinsics:", [round(v,1) for v in src.intrinsics])
seenA = seenB = 0; N = 150; saved = 0
for k in range(N):
    bgr, _ = src.read()
    if bgr is None:
        time.sleep(0.02); continue
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bright = gray.mean(); sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    cA, iA, _ = detA.detectMarkers(gray)
    cB, iB, _ = detB.detectMarkers(gray)
    idsA = [] if iA is None else iA.ravel().tolist()
    idsB = [] if iB is None else iB.ravel().tolist()
    okA, okB = (0 in idsA), (0 in idsB)
    seenA += okA; seenB += okB
    if k % 25 == 0 or (okB and not okA):
        szB = (np.mean([np.linalg.norm(cB[0][0][i]-cB[0][0][(i+1)%4]) for i in range(4)])
               if idsB else 0)
        print(f"f{k:3d}: bright {bright:3.0f} sharp {sharp:5.0f}  "
              f"teleop:{'YES' if okA else 'no '} relaxed:{'YES' if okB else 'no '}  "
              f"ids={idsB}  tag_px~{szB:.0f}")
    if (okA or okB) and saved < 3:
        vis = bgr.copy(); cv2.aruco.drawDetectedMarkers(vis, cB if okB else cA, iB if okB else iA)
        cv2.imwrite(f"{OUT}/tagseen_{saved}.png", vis); saved += 1
    if k == N-1: cv2.imwrite(f"{OUT}/tagdiag_lastframe.png", bgr)
src.close()
print(f"\nSUMMARY {N} frames: teleop-config {seenA}, relaxed {seenB}")
if seenA==0 and seenB==0: print(">> tag NEVER seen - check tagdiag_lastframe.png: is the tag in view, sharp, lit?")
elif seenB>seenA+5: print(">> relaxed finds it much more - teleop params too strict; will loosen them")
else: print(">> detection works; if teleop still says LOST, it's tag distance/visibility during motion")
