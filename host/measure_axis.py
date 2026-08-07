"""Work out which sensor axis runs along the arm, by measuring instead of
guessing. Re-run whenever the enclosure or mounting changes.

Hold three poses; this reports the LIMB_AXIS / LIMB_SIGN to put in config.py.
"""
import asyncio, sys, time
import numpy as np
import websockets

from . import protocol

POSES = [
    ("HANGING  — arm straight down at your side, hand flat", 7),
    ("HORIZONTAL — arm straight out in front, palm down", 7),
    ("OVERHEAD — arm straight up, hand flat above your head", 7),
]

samples = {}


def rotate(q, v):
    """Rotate v from sensor frame into world frame. q is (w,x,y,z)."""
    w, xyz = q[0], q[1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


async def handler(ws):
    print("\ndevice connected\n")
    current = {"frames": []}

    async def collect():
        async for msg in ws:
            kind, payload = protocol.unpack(msg)
            if kind == protocol.MOTION:
                for f in payload:
                    current["frames"].append([f.qw, f.qx, f.qy, f.qz])

    task = asyncio.create_task(collect())

    for name, secs in POSES:
        for n in (3, 2, 1):
            print(f"  next: {name}   ...{n}", end="\r", flush=True)
            await asyncio.sleep(1)
        print(f"  HOLD: {name}          ")
        current["frames"].clear()
        await asyncio.sleep(secs)
        frames = list(current["frames"])
        # middle half only, so entering and leaving the pose don't count
        mid = frames[len(frames) // 4: 3 * len(frames) // 4]
        samples[name.split()[0]] = np.median(np.array(mid), axis=0)
        print(f"        captured {len(mid)} frames\n")

    task.cancel()
    await ws.close()


async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765, max_size=None):
        print("waiting for the device (make sure host.server is stopped)...")
        while len(samples) < len(POSES):
            await asyncio.sleep(0.2)

    print("=" * 58)
    hang = samples["HANGING"] / np.linalg.norm(samples["HANGING"])
    horiz = samples["HORIZONTAL"] / np.linalg.norm(samples["HORIZONTAL"])
    over = samples["OVERHEAD"] / np.linalg.norm(samples["OVERHEAD"])

    print("world-Z component of each sensor axis, per pose")
    print("  axis   hanging  horizontal  overhead   swing")
    best, best_swing = None, 0.0
    for i, label in enumerate("xyz"):
        e = np.zeros(3); e[i] = 1.0
        zs = [float(rotate(q, e)[2]) for q in (hang, horiz, over)]
        swing = zs[2] - zs[0]
        print("   %s    %+7.2f    %+7.2f   %+7.2f   %+6.2f"
              % (label, zs[0], zs[1], zs[2], swing))
        if abs(swing) > abs(best_swing):
            best, best_swing = i, swing

    print()
    print("  LIMB_AXIS = %d      # %s" % (best, "xyz"[best]))
    print("  LIMB_SIGN = %d" % (1 if best_swing > 0 else -1))
    print()

    # Persist it, so the measurement survives the terminal it was printed in.
    import json
    with open("axis_result.json", "w") as fh:
        json.dump({
            "limb_axis": int(best),
            "limb_sign": 1 if best_swing > 0 else -1,
            "swing": float(best_swing),
            "quaternions": {k: [float(x) for x in v] for k, v in samples.items()},
        }, fh, indent=2)
    print("  written to axis_result.json")
    if abs(best_swing) < 1.0:
        print("  WARNING: swing is small (%.2f). Poses may not have been held"
              % abs(best_swing))
        print("  distinctly enough — expect close to 2.0 for a clean axis.")



if __name__ == "__main__":
    asyncio.run(main())
