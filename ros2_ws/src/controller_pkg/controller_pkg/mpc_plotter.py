#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def lemniscate_ref(t, x0, y0, a=1.0):
    """
    Reconstruct lemniscate reference used in your controller.
    """
    b = 0.5 * np.tanh(0.1 * t)
    px = x0 + a * np.sin(b * t)
    py = y0 + a * np.sin(b * t) * np.cos(b * t)
    return px, py


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file", required=True,
                        help="npy file storing ground truth (T x 3)")
    parser.add_argument("--label", default="nom",
                        help="Label prefix for saving")
    args = parser.parse_args()

    save_dir = "mpc_logs"
    os.makedirs(save_dir, exist_ok=True)

    ### Load ground truth
    gt = np.load(args.gt_file)    # shape (T, 3)
    if gt.shape[1] < 2:
        raise ValueError("Ground truth must have at least x,y columns")

    T = gt.shape[0]

    # Ground truth XY
    gt_xy = gt[:, :2]

    # Use first GT point as origin
    x0, y0 = gt_xy[0]

    # time index → assume 50 Hz, can adjust
    dt_assumed = 1.0 / 50.0
    times = np.arange(T) * dt_assumed

    # Create reference curve
    ref_xy = np.zeros_like(gt_xy)
    for i, t in enumerate(times):
        ref_xy[i,0], ref_xy[i,1] = lemniscate_ref(t, x0, y0)

    # Plot
    plt.figure(figsize=(6,6))

    plt.plot(gt_xy[:,0], gt_xy[:,1], "-",  label="Ground Truth")
    plt.plot(ref_xy[:,0], ref_xy[:,1], "--", label="Lemniscate Reference")

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"{args.label} — Lemniscate Reference Tracking")
    plt.grid(True)
    plt.legend()

    # Equal axis scale
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    ts = os.path.basename(args.gt_file).replace(".npy", "")
    fname = f"{save_dir}/{args.label}_compare_{ts}.png"
    plt.savefig(fname, dpi=200)
    plt.close()

    print(f":white_check_mark: Saved plot: {fname}")


if __name__ == "__main__":
    main()