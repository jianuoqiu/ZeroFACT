#!/usr/bin/env python3
"""Wuji-glove -> UDP bridge for live teleop.

The Wuji SDK is a compiled cp311 extension living in the ``wuji-sdk`` conda env,
while ``live_teleop.py`` runs in ``cam`` (py3.12) - so the glove is read in its
own small process and streamed as JSON over UDP (loopback), exactly like the
tracker->viewer link. Two sources, one wire format:

  live (default; run in the wuji-sdk env, glove on Ethernet):
      conda run -n wuji-sdk python sim_teleop/wuji_bridge.py
  mcap replay (debug without hardware; any python with the ``mcap`` package):
      python3 sim_teleop/wuji_bridge.py --mcap <recording.mcap> [--loop]

Wire format, one JSON datagram per skeleton frame (~120 Hz), to 127.0.0.1:5557:
    {"seq": n, "t_us": <skeleton timestamp>, "skel": [[x,y,z]*21],
     "quat": [x,y,z,w]}          # palm IMU orientation, world frame (z-up, ENU)
    {"bye": 1}                    on exit

Conventions (measured on the recordings in ~/tactile_glove):
  * skel: 21 MediaPipe-ordered joints, WRIST-LOCAL - joint 0 pinned at (0,0,0)
    and the palm frame constant; pure finger articulation, real hand scale.
  * quat: fused world orientation; R(quat) @ linear_acceleration = [0,0,+9.8]
    at rest, i.e. a gravity-aligned z-up world - same "up" as the robot base.
"""

from __future__ import annotations

import argparse
import functools
import json
import socket
import sys
import time

print = functools.partial(print, flush=True)   # the launcher pipes our output

WIRE_ADDR = ("127.0.0.1", 5557)


def _send(sock, obj):
    sock.sendto(json.dumps(obj).encode(), WIRE_ADDR)


# --------------------------------------------------------------------------------------
# mcap replay source (debug without hardware)
# --------------------------------------------------------------------------------------
def stream_mcap(path: str, loop: bool, rate_scale: float) -> None:
    from mcap.reader import make_reader

    skel, imu = [], []          # (t_us, payload)
    with open(path, "rb") as f:
        for _, ch, msg in make_reader(f).iter_messages(
                topics=["/right_glove/hand_skeleton", "/right_glove/imu_data/palm"]):
            d = json.loads(msg.data)
            t = d["header"]["timestamp_us"]
            if ch.topic.endswith("hand_skeleton"):
                skel.append((t, [[j["pose"]["position"][a] for a in range(3)]
                                 for j in d["joints"]]))
            else:
                o = d["orientation"]
                imu.append((t, [o["x"], o["y"], o["z"], o["w"]]))
    if not skel or not imu:
        print(f"[wuji-bridge][ERROR] no skeleton/imu messages in {path}")
        sys.exit(1)
    skel.sort(); imu.sort()
    print(f"[wuji-bridge] mcap replay: {len(skel)} skeleton frames, {len(imu)} imu "
          f"({(skel[-1][0]-skel[0][0])/1e6:.1f} s) -> udp {WIRE_ADDR[0]}:{WIRE_ADDR[1]}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = 0
    try:
        while True:
            t_wall0, t_rec0 = time.perf_counter(), skel[0][0]
            ii = 0
            for t_us, pts in skel:
                # pace on the recording's own clock
                dt = (t_us - t_rec0) / 1e6 / rate_scale - (time.perf_counter() - t_wall0)
                if dt > 0:
                    time.sleep(dt)
                while ii + 1 < len(imu) and imu[ii + 1][0] <= t_us:
                    ii += 1                      # latest imu sample at/before this frame
                _send(sock, {"seq": seq, "t_us": t_us, "skel": pts, "quat": imu[ii][1]})
                seq += 1
            if not loop:
                break
            print("[wuji-bridge] looping.")
    except KeyboardInterrupt:
        pass
    finally:
        _send(sock, {"bye": 1})
        print("[wuji-bridge] done.")


# --------------------------------------------------------------------------------------
# live SDK source (run inside the wuji-sdk conda env)
# --------------------------------------------------------------------------------------
def stream_live(sn: str | None, address: str | None) -> None:
    try:
        from wuji_sdk import SdkManager
    except ImportError:
        print("[wuji-bridge][ERROR] wuji_sdk not importable - run this in the "
              "'wuji-sdk' conda env:  conda run -n wuji-sdk python sim_teleop/wuji_bridge.py")
        sys.exit(1)

    m = SdkManager.instance()

    def _try_connect():
        if sn:
            return m.connect(sn=sn, device_name="teleop_glove")
        if address:
            return m.connect(address=address, device_name="teleop_glove")
        devs = m.scan()
        if not devs:
            raise RuntimeError("no Wuji devices found (Ethernet up? glove powered?)")
        print(f"[wuji-bridge] found: " + ", ".join(f"{d.sn}@{d.address}" for d in devs))
        return m.auto_connect("teleop_glove")

    # The glove accepts ONE session. If Wuji Studio (or another SDK client) holds it,
    # connecting fails with "Session already exists" - do not die: tell the operator and
    # retry until the session frees (Studio closed / glove disconnected there; the
    # device also needs its heartbeat timeout to lapse before accepting a new session).
    glove = None
    warned = False
    deadline = time.perf_counter() + 600.0
    while glove is None:
        try:
            glove = _try_connect()
        except Exception as e:
            msg = str(e)
            if "Session already exists" in msg:
                if not warned:
                    print("[wuji-bridge] the glove is HELD BY ANOTHER APP (Wuji Studio?).")
                    print("[wuji-bridge] -> close Wuji Studio or disconnect the glove "
                          "there; retrying every 3 s ...")
                    warned = True
            else:
                print(f"[wuji-bridge] connect failed: {msg} - retrying in 3 s ...")
            if time.perf_counter() > deadline:
                print("[wuji-bridge][ERROR] gave up after 10 min.")
                sys.exit(1)
            time.sleep(3.0)
    print(f"[wuji-bridge] connected: {glove.serial_number}")

    sub_skel = glove.hand_skeleton().subscribe()
    sub_imu = glove.imu_data_palm().subscribe()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq, last_quat, last_report, n_report = 0, None, time.perf_counter(), 0
    last_data = time.perf_counter()
    print(f"[wuji-bridge] streaming -> udp {WIRE_ADDR[0]}:{WIRE_ADDR[1]}  (ctrl-c to stop)")
    try:
        while True:
            imu = sub_imu.recv()                 # drain to the newest orientation
            while imu is not None:
                o = imu.orientation
                last_quat = [o.x, o.y, o.z, o.w]
                imu = sub_imu.recv()
            skel = sub_skel.recv()
            sent = False
            while skel is not None:              # forward every skeleton frame
                if last_quat is not None:
                    pts = [list(j.pose.position) for j in skel.joints]
                    _send(sock, {"seq": seq, "t_us": skel.header.timestamp_us,
                                 "skel": pts, "quat": last_quat})
                    seq += 1
                    n_report += 1
                    sent = True
                skel = sub_skel.recv()
            now = time.perf_counter()
            if sent:
                last_data = now
            else:
                time.sleep(0.002)
                # the SDK disconnects the glove on heartbeat timeout (e.g. the USB-Ethernet
                # link dropped) and recv() then just returns None forever - don't stream a
                # silent 0 Hz, detect it and stop loudly so the operator fixes the link
                if now - last_data > 3.0:
                    connected = getattr(glove, "is_connected", True)
                    if not connected or now - last_data > 8.0:
                        print("[wuji-bridge][ERROR] glove stopped sending"
                              + ("" if connected else " (SDK reports DISCONNECTED)")
                              + " - check the USB-Ethernet link (see README: nmcli "
                              "connection up wuji-glove) and the glove power, then rerun.")
                        break
            if now - last_report >= 5.0:
                hz = n_report / (now - last_report)
                print(f"[wuji-bridge] {hz:.0f} Hz"
                      + ("   (NO DATA - glove link down?)" if hz < 1 else ""))
                last_report, n_report = now, 0
    except KeyboardInterrupt:
        pass
    finally:
        _send(sock, {"bye": 1})
        print("[wuji-bridge] done.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mcap", default=None, help="replay a Wuji Studio .mcap instead of live SDK")
    ap.add_argument("--loop", action="store_true", help="mcap: loop forever")
    ap.add_argument("--rate-scale", type=float, default=1.0, help="mcap: playback speed factor")
    ap.add_argument("--sn", default=None, help="live: connect by serial number")
    ap.add_argument("--address", default=None, help="live: connect by IP:port")
    args = ap.parse_args()

    if args.mcap:
        stream_mcap(args.mcap, args.loop, args.rate_scale)
    else:
        stream_live(args.sn, args.address)


if __name__ == "__main__":
    main()
