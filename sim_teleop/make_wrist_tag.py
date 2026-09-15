#!/usr/bin/env python3
"""Generate a TRUE-TO-SCALE printable AprilTag for the glove-wrist tracker
(``live_teleop.py --wrist-tag``).

Output is a **PDF** whose marker is drawn at an exact physical size, because a PNG carries
no reliable physical-size metadata (printers fit-to-page it, so it comes out the wrong
size - the usual cause of "this isn't 30 mm"). Print the PDF at **100% / Actual size**
(NOT "fit to page") and the black square measures exactly ``--mm`` millimetres. A little
ruler is printed beside it so you can confirm the scale before trusting it.

    conda run -n cam python sim_teleop/make_wrist_tag.py                 # id 0, 30 mm PDF
    conda run -n cam python sim_teleop/make_wrist_tag.py --mm 30 --id 0

Then measure the black square (should equal --mm) and run teleop with that value in metres:
    --wrist-tag --tag-size 0.030 --tag-id 0
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", type=int, default=0, help="AprilTag id to encode")
    ap.add_argument("--mm", type=float, default=30.0,
                    help="printed marker (black square) side length in mm (default 30)")
    ap.add_argument("--frame-mm", type=float, default=None,
                    help="if set, draw a black square CUT FRAME at this outer size (mm) "
                    "around the tag; the white gap (frame-mm - mm)/2 is the quiet zone "
                    "that lets the tag detect against a dark background/glove")
    ap.add_argument("--dict", default="DICT_APRILTAG_36h11", help="cv2.aruco dictionary")
    ap.add_argument("--out", default="wrist_tag.pdf",
                    help="output path; .pdf (exact size, recommended) or .png")
    ap.add_argument("--page", default="letter", choices=["letter", "a4"],
                    help="PDF page size")
    args = ap.parse_args()

    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    marker = cv2.aruco.generateImageMarker(dic, args.id, 1000)   # outer edge = black border

    # self-check: the tag must re-detect (catches a bad dict/id)
    det = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
    _, ids, _ = det.detectMarkers(np.pad(marker, 250, constant_values=255))
    ok = ids is not None and args.id in ids.ravel()

    mm = float(args.mm)
    if args.out.lower().endswith(".pdf"):
        _write_pdf(marker, mm, args)
    else:
        _write_png(marker, mm, args)

    print(f"[make-wrist-tag] wrote {args.out}  ({args.dict} id {args.id}, {mm:.0f} mm)")
    print(f"[make-wrist-tag] self-detect: {'OK' if ok else 'FAILED'}")
    print(f"[make-wrist-tag] PRINT AT 100% / ACTUAL SIZE (not fit-to-page), then measure "
          f"the black square = {mm:.0f} mm and the ruler; then run:")
    print(f"[make-wrist-tag]   ... --wrist-tag --tag-size {mm / 1000:.3f} "
          f"--tag-id {args.id} --tag-dict {args.dict}")


def _write_pdf(marker: np.ndarray, mm: float, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    IN = 25.4
    page_w, page_h = (215.9, 279.4) if args.page == "letter" else (210.0, 297.0)  # mm
    fig = plt.figure(figsize=(page_w / IN, page_h / IN))

    def frac(x_mm, y_mm, w_mm, h_mm):
        return [x_mm / page_w, y_mm / page_h, w_mm / page_w, h_mm / page_h]

    # marker at EXACT physical size, centred horizontally, near the top
    mx = (page_w - mm) / 2.0
    my = page_h - 45.0 - mm
    ax = fig.add_axes(frac(mx, my, mm, mm))
    ax.imshow(marker, cmap="gray", vmin=0, vmax=255, aspect="auto",
              interpolation="nearest")
    ax.axis("off")

    frame_mm = args.frame_mm
    if frame_mm:
        # black cut-frame centred on the tag; the (frame-mm - mm)/2 gap is white quiet zone
        cxm, cym = mx + mm / 2.0, my + mm / 2.0
        fx0, fy0 = cxm - frame_mm / 2.0, cym - frame_mm / 2.0
        fax = fig.add_axes(frac(fx0 - 5, fy0 - 5, frame_mm + 10, frame_mm + 10))
        fax.axis("off"); fax.set_xlim(0, frame_mm + 10); fax.set_ylim(0, frame_mm + 10)
        fax.set_zorder(-1)                       # behind the marker axes
        fax.add_patch(Rectangle((5, 5), frame_mm, frame_mm, fill=False,
                                edgecolor="black", linewidth=1.2))
    else:
        # corner ticks marking the exact black-square extent (no frame requested)
        for xx, yy, dx, dy in [(mx, my, 6, 0), (mx, my, 0, 6),
                               (mx + mm, my + mm, -6, 0), (mx + mm, my + mm, 0, -6)]:
            fig.add_artist(plt.Line2D([xx / page_w, (xx + dx) / page_w],
                                      [yy / page_h, (yy + dy) / page_h],
                                      color="black", lw=0.8))

    # a real ruler below the tag (0..50 mm) to verify print scale with a physical ruler
    ry = my - 22.0
    rx0 = (page_w - 50.0) / 2.0
    ruler = fig.add_axes(frac(rx0, ry, 50.0, 12.0)); ruler.axis("off")
    ruler.set_xlim(0, 50); ruler.set_ylim(0, 12)
    ruler.add_patch(Rectangle((0, 8), 50, 0.4, color="black"))
    for t in range(0, 51, 5):
        h = 3.2 if t % 10 == 0 else 2.0
        ruler.add_line(plt.Line2D([t, t], [8, 8 - h], color="black", lw=0.8))
        if t % 10 == 0:
            ruler.text(t, 3.0, f"{t}", ha="center", va="top", fontsize=6)
    ruler.text(25, 0.5, "ruler: these should read 0-50 mm on a real ruler",
               ha="center", va="top", fontsize=6, style="italic")

    cut = (f"   cut ON/just OUTSIDE the {frame_mm:.0f} mm frame (keep the white ring)"
           if frame_mm else "")
    fig.text(0.5, (page_h - 22) / page_h,
             f"wrist AprilTag   {args.dict}   id {args.id}   "
             f"black square = {mm:.0f} mm   (print at 100% / actual size){cut}",
             ha="center", va="center", fontsize=8.5)
    fig.savefig(args.out)
    plt.close(fig)


def _write_png(marker: np.ndarray, mm: float, args) -> None:
    from PIL import Image

    dpi = 600
    px = int(round(mm / 25.4 * dpi))
    m = cv2.resize(marker, (px, px), interpolation=cv2.INTER_NEAREST)
    quiet = px // 4
    label = int(0.4 * dpi)
    W = px + 2 * quiet
    canvas = np.full((W + label, W), 255, np.uint8)
    canvas[quiet:quiet + px, quiet:quiet + px] = m
    cv2.putText(canvas, f"{args.dict} id={args.id} {mm:.0f}mm (print 100%)",
                (quiet, W + label // 2), cv2.FONT_HERSHEY_SIMPLEX,
                dpi / 900.0, 0, max(1, dpi // 200), cv2.LINE_AA)
    # embed the DPI so at least DPI-aware print paths get the size right
    Image.fromarray(canvas).save(args.out, dpi=(dpi, dpi))


if __name__ == "__main__":
    main()
