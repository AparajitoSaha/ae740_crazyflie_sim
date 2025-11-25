#!/usr/bin/env python3
"""Compare two CF trajectories (CF1 target, CF2 pursuer).

Saves:
- {label}_cf_compare_3d_{ts}.png  (3D trajectories)
- {label}_cf_error_{ts}.png       (aligned per-sample Euclidean error)

Usage:
 - Edit the CF1_NPY and CF2_NPY at the top to point to your files, or pass --cf1/--cf2 on CLI.
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime


def make_save_dir(base_path=None):
    if base_path is None:
        base_path = os.path.join(os.path.dirname(__file__), "mpc_logs")
    os.makedirs(base_path, exist_ok=True)
    return base_path


def load_gt(path):
    gt = np.load(path)
    if gt.ndim == 1:
        gt = gt.reshape(-1, gt.shape[0])
    if gt.shape[1] < 2:
        raise ValueError("Ground truth must have at least x,y columns")
    return gt


def find_best_shift(a, b, max_shift_samples=200):
    """Search integer shifts between a and b that minimize XY MSE.
    Returns (best_shift, aligned_a, aligned_b)
    Positive shift means a is delayed relative to b (a shifted forward).
    """
    n1 = a.shape[0]
    n2 = b.shape[0]
    max_shift = min(max_shift_samples, n1//2, n2//2)

    def xy_mse(x, y):
        return np.mean(np.sum((x[:, :2] - y[:, :2])**2, axis=1))

    best_shift = 0
    best_mse = None
    for shift in range(-max_shift, max_shift+1):
        if shift >= 0:
            L = min(n1-shift, n2)
            if L <= 5:
                continue
            x = a[shift:shift+L, :]
            y = b[:L, :]
        else:
            s = -shift
            L = min(n1, n2-s)
            if L <= 5:
                continue
            x = a[:L, :]
            y = b[s:s+L, :]

        mse = xy_mse(x, y)
        if best_mse is None or mse < best_mse:
            best_mse = mse
            best_shift = shift

    # Build aligned arrays
    if best_shift >= 0:
        L = min(n1-best_shift, n2)
        a_al = a[best_shift:best_shift+L, :]
        b_al = b[:L, :]
    else:
        s = -best_shift
        L = min(n1, n2-s)
        a_al = a[:L, :]
        b_al = b[s:s+L, :]

    return best_shift, a_al, b_al


def plot_3d_compare(a, b, save_path, title='CF Trajectories', center_on_origin=True):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    a_plot = a.copy()
    b_plot = b.copy()
    if center_on_origin and a_plot.shape[0] > 0:
        offset = a_plot[0, :2].copy()
        a_plot[:, 0] -= offset[0]
        a_plot[:, 1] -= offset[1]
        b_plot[:, 0] -= offset[0]
        b_plot[:, 1] -= offset[1]

    a_z = a_plot[:, 2] if a_plot.shape[1] > 2 else np.zeros(a_plot.shape[0])
    b_z = b_plot[:, 2] if b_plot.shape[1] > 2 else np.zeros(b_plot.shape[0])

    ax.plot(a_plot[:, 0], a_plot[:, 1], a_z, '-C0', label='Target (CF1)')
    ax.plot(b_plot[:, 0], b_plot[:, 1], b_z, '-C1', label='Pursuer (CF2)')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    ax.set_zlim(0.0, 1.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_error(err, save_path, title='Position Error'):
    plt.figure(figsize=(8, 3))
    plt.plot(np.arange(err.shape[0]), err, '-k')
    plt.xlabel('Sample index')
    plt.ylabel('Position error (m)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    # --- Edit these variables to point to your numpy files ---
    # Set to None if you don't have that file
    CF1_NPY = '/home/robin/ae740_crazyflie_sim/ros2_ws/src/controller_pkg/controller_pkg/mpc_logs/nom_gt_20251125_005957.npy'
    CF2_NPY = '/home/robin/ae740_crazyflie_sim/ros2_ws/src/controller_pkg/controller_pkg/mpc_logs/ssi_gt_20251125_013434.npy'

    parser = argparse.ArgumentParser(description='Compare two CF trajectories (CF1 target, CF2 pursuer)')
    parser.add_argument('--cf1', help='Path to CF1 numpy file', default=CF1_NPY)
    parser.add_argument('--cf2', help='Path to CF2 numpy file', default=CF2_NPY)
    parser.add_argument('--label', help='Label prefix for outputs', default='cf_compare')
    parser.add_argument('--max-shift-samples', type=int, default=200, help='Max integer shift samples to search')
    args = parser.parse_args()

    if args.cf1 is None or args.cf2 is None:
        raise RuntimeError('Please set CF1 and CF2 numpy file paths either by editing the script or passing --cf1/--cf2')

    a = load_gt(args.cf1)
    b = load_gt(args.cf2)

    save_dir = make_save_dir()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Find best alignment and compute error
    best_shift, a_al, b_al = find_best_shift(a, b, max_shift_samples=args.max_shift_samples)
    err = np.linalg.norm(a_al - b_al, axis=1)

    fname3d = os.path.join(save_dir, f"{args.label}_3d_{ts}.png")
    # do not set titles as requested
    plot_3d_compare(a_al, b_al, fname3d, title=None, center_on_origin=True)

    fname_err = os.path.join(save_dir, f"{args.label}_error_{ts}.png")
    plot_error(err, fname_err, title=None)

    print(f":white_check_mark: Saved trajectories: {fname3d}")
    print(f":white_check_mark: Saved error plot: {fname_err}")


if __name__ == '__main__':
    main()
