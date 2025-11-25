#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime


def lemniscate_ref(t, x0, y0, a=1.0):
    """
    Reconstruct lemniscate reference used in your controller.
    Vectorized over t (array or scalar).
    """
    b = 0.5 * np.tanh(0.1 * t)
    px = x0 + a * np.sin(b * t)
    py = y0 + a * np.sin(b * t) * np.cos(b * t)
    return px, py


def circle_ref(t, x0, y0, a=1.0):
    """
    Simple circular reference. Vectorized over t (array or scalar).
    Uses a slowly varying frequency term similar to the lemniscate for smoother start.
    """
    b = 0.5 * np.tanh(0.1 * t)
    px = x0 + a * np.cos(b * t)
    py = y0 + a * np.sin(b * t)
    return px, py


def make_save_dir(base_path=None):
    """Create package-local mpc_logs by default."""
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


def build_reference_for_gt(gt, dt=1.0/50.0, a=1.0, shape=None):
    """Build a reference curve aligned to the GT first point.

    By default this returns the lemniscate reference. Pass `shape='circle'` to build a
    circular reference instead. Returned array has shape (T,3) where z is set to 1.0.
    """
    # precedence: explicit arg -> attribute hook -> default
    if shape is None:
        shape = getattr(build_reference_for_gt, "shape", "lemniscate")

    T = gt.shape[0]
    times = np.arange(T) * dt
    x0, y0 = gt[0, 0], gt[0, 1]

    if shape == "circle":
        px, py = circle_ref(times, x0, y0, a=a)
    else:
        px, py = lemniscate_ref(times, x0, y0, a=a)

    # Per request: set reference Z to constant 1.0
    pz = np.ones(T) * 1.0
    ref = np.stack([px, py, pz], axis=1)
    return ref


def plot_3d(gt, ref, title, fname, ref_label="Lemniscate Reference", center_on_origin=False):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Optionally recenter both GT and reference so that GT starts at origin.
    gt_plot = gt.copy()
    ref_plot = ref.copy()
    if center_on_origin and gt_plot.shape[0] > 0 and ref_plot.shape[0] > 0:
        # shift by GT first sample in XY
        offset = gt_plot[0, :2].copy()
        gt_plot[:, 0] -= offset[0]
        gt_plot[:, 1] -= offset[1]
        ref_plot[:, 0] -= offset[0]
        ref_plot[:, 1] -= offset[1]

    gt_z = gt_plot[:, 2] if gt_plot.shape[1] > 2 else np.zeros(gt_plot.shape[0])
    ax.plot(gt_plot[:, 0], gt_plot[:, 1], gt_z, '-k', label=title)
    ax.plot(ref_plot[:, 0], ref_plot[:, 1], ref_plot[:, 2], '--r', label=ref_label)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    # Fix z axis as requested
    ax.set_zlim(0.0, 1.5)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def plot_error(gt, ref, title, fname):
    # compute per-sample Euclidean error
    # ensure both are same shape
    # If ref has fewer columns, pad to match gt
    if ref.shape[1] < gt.shape[1]:
        ref = np.hstack([ref, np.zeros((ref.shape[0], gt.shape[1] - ref.shape[1]))])
    # Align ref and gt in time to compensate for logging delay.
    # Search for a small integer offset in both directions that minimizes XY MSE.
    # Positive shift means gt is delayed relative to ref (gt shifted forward).
    max_shift = 200
    max_shift = min(max_shift, gt.shape[0]//2, ref.shape[0]//2)

    def xy_mse(a, b):
        return np.mean(np.sum((a[:, :2] - b[:, :2])**2, axis=1))

    best_shift = 0
    best_mse = None
    # search negative->positive to allow ref leading or lagging
    for shift in range(-max_shift, max_shift+1):
        if shift >= 0:
            L = min(gt.shape[0]-shift, ref.shape[0])
            if L <= 5:
                continue
            g = gt[shift:shift+L, :]
            r = ref[:L, :]
        else:
            s = -shift
            L = min(gt.shape[0], ref.shape[0]-s)
            if L <= 5:
                continue
            g = gt[:L, :]
            r = ref[s:s+L, :]

        mse = xy_mse(g, r)
        if best_mse is None or mse < best_mse:
            best_mse = mse
            best_shift = shift

    # Apply best shift to produce aligned windows for error plotting
    if best_shift >= 0:
        L = min(gt.shape[0]-best_shift, ref.shape[0])
        gt_a = gt[best_shift:best_shift+L, :ref.shape[1]]
        ref_a = ref[:L, :ref.shape[1]]
    else:
        s = -best_shift
        L = min(gt.shape[0], ref.shape[0]-s)
        gt_a = gt[:L, :ref.shape[1]]
        ref_a = ref[s:s+L, :ref.shape[1]]
    err = np.linalg.norm(gt_a - ref_a, axis=1)
    times = np.arange(L)

    # Include the found alignment shift in the title to make debugging easier
    title = f"{title} (aligned shift={best_shift})"

    plt.figure(figsize=(8, 3))
    plt.plot(times, err, '-b')
    plt.xlabel('Sample index')
    plt.ylabel('Position error (m)')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def main():
    # --- Edit these variables to point to your numpy files ---
    # Set to None if you don't have that file
    GT_NOM = '/home/robin/ae740_crazyflie_sim/ros2_ws/src/controller_pkg/controller_pkg/mpc_logs/nom_gt_20251125_005957.npy'
    GT_SSI = '/home/robin/ae740_crazyflie_sim/ros2_ws/src/controller_pkg/controller_pkg/mpc_logs/ssi_gt_20251125_013434.npy'

    # Example:
    # GT_NOM = '/home/robin/ae740_crazyflie_sim/ros2_ws/src/controller_pkg/controller_pkg/mpc_logs/nom_gt_20250101_120000.npy'
    # GT_SSI = '/home/robin/ae740_crazyflie_sim/ros2_ws/src/controller_pkg/controller_pkg/mpc_logs/ssi_gt_20250101_120000.npy'

    LABEL = 'mpc'
    DT = 1.0 / 50.0
    A = 1.0

    # Allow optional CLI to switch reference shape or change save directory.
    parser = argparse.ArgumentParser(description='Plot MPC ground truth vs reference')
    parser.add_argument('--shape', choices=['lemniscate', 'circle'], default='lemniscate',
                        help='Reference shape to rebuild (default: lemniscate)')
    parser.add_argument('--scan-dir', action='store_true',
                        help='If set and one of the GT paths is None, scan package mpc_logs for files')
    args = parser.parse_args()

    # attach chosen shape to builder (simple global-ish hook)
    setattr(build_reference_for_gt, 'shape', args.shape)

    save_dir = make_save_dir()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    any_saved = False

    if GT_NOM:
        gt_nom = load_gt(GT_NOM)
        if gt_nom.shape[1] == 2:
            gt_nom = np.hstack([gt_nom, np.zeros((gt_nom.shape[0], 1))])
        ref_nom = build_reference_for_gt(gt_nom, dt=DT, a=A, shape=args.shape)

        fname3d = os.path.join(save_dir, f"{LABEL}_3d_nom_{ts}.png")
        ref_label = "Circle Reference" if args.shape == 'circle' else 'Lemniscate Reference'
        plot_3d(gt_nom, ref_nom, f"Nominal MPC — 3D Trajectory", fname3d, ref_label=ref_label, center_on_origin=True)

        fname_err = os.path.join(save_dir, f"{LABEL}_error_nom_{ts}.png")
        plot_error(gt_nom, ref_nom, f"Nominal MPC — Position Error", fname_err)

        print(f":white_check_mark: Saved nominal plots: {fname3d}, {fname_err}")
        any_saved = True

    if GT_SSI:
        gt_ssi = load_gt(GT_SSI)
        if gt_ssi.shape[1] == 2:
            gt_ssi = np.hstack([gt_ssi, np.zeros((gt_ssi.shape[0], 1))])
        ref_ssi = build_reference_for_gt(gt_ssi, dt=DT, a=A, shape=args.shape)

        fname3d = os.path.join(save_dir, f"{LABEL}_3d_ssi_{ts}.png")
        ref_label = "Circle Reference" if args.shape == 'circle' else 'Lemniscate Reference'
        plot_3d(gt_ssi, ref_ssi, f"SA MPC — 3D Trajectory", fname3d, ref_label=ref_label, center_on_origin=True)

        fname_err = os.path.join(save_dir, f"{LABEL}_error_ssi_{ts}.png")
        plot_error(gt_ssi, ref_ssi, f"SA MPC — Position Error", fname_err)

        print(f":white_check_mark: Saved SA plots: {fname3d}, {fname_err}")
        any_saved = True

    if not any_saved:
        # If user enabled scanning, try to find gt files in the package mpc_logs
        if args.scan_dir:
            pkg_dir = os.path.dirname(__file__)
            search_dir = os.path.join(pkg_dir, 'mpc_logs')
            if os.path.isdir(search_dir):
                import glob
                matches = sorted(glob.glob(os.path.join(search_dir, '*_gt_*.npy')))
                if len(matches) > 0:
                    print(f"Found {len(matches)} ground-truth files in {search_dir}. Try setting GT_NOM/GT_SSI manually or rerun with paths.")
                    return
        raise RuntimeError('No GT files configured. Edit GT_NOM and/or GT_SSI at the top of main().')


if __name__ == '__main__':
    main()