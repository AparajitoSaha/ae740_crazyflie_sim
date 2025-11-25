#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os


class TrajectoryCompareLogger(Node):

    def __init__(self):
        super().__init__("trajectory_compare_logger")

        self.declare_parameter("cf_name", "cf_1")
        self.declare_parameter("label", "ssi")

        self.cf_name = self.get_parameter("cf_name").value
        self.label   = self.get_parameter("label").value

        # Buffers
        self.ground_truth_cf1 = []  # /cf_1/pose
        self.ground_truth_cf2 = []  # /cf_2/pose
        self.ref_traj     = []      # /all/mpc_trajectory

        # SUBSCRIBE
        # subscribe to both cf_1 and cf_2 poses
        self.create_subscription(PoseStamped, f"/cf_1/pose", self.cb_pose_cf1, 10)
        self.create_subscription(PoseStamped, f"/cf_2/pose", self.cb_pose_cf2, 10)

        self.create_subscription(
            Path,
            "/all/mpc_trajectory",
            self.cb_ref_traj,
            10
        )

        self.get_logger().info(f"[{self.label}] logger active: logging /cf_1/pose, /cf_2/pose and /all/mpc_trajectory")


    # --- Callbacks ---
    def cb_pose_cf1(self, msg: PoseStamped):
        p = msg.pose.position
        self.ground_truth_cf1.append([p.x, p.y, p.z])

    def cb_pose_cf2(self, msg: PoseStamped):
        p = msg.pose.position
        self.ground_truth_cf2.append([p.x, p.y, p.z])


    def cb_ref_traj(self, msg: Path):
        """Store full trajectory published on /all/mpc_trajectory"""
        pts = []
        for ps in msg.poses:
            pts.append([
                ps.pose.position.x,
                ps.pose.position.y,
                ps.pose.position.z
            ])

        if len(pts) > 0:
            # store entire track horizon
            self.ref_traj.append(pts)


    # --- Saving + plotting ---
    def save_and_plot(self):
        # Save logs inside the ROS2 package directory so they live together with the node
        save_dir = os.path.join(os.path.dirname(__file__), "mpc_logs")
        os.makedirs(save_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        gt1 = np.array(self.ground_truth_cf1) if len(self.ground_truth_cf1) > 0 else np.zeros((0,3))
        gt2 = np.array(self.ground_truth_cf2) if len(self.ground_truth_cf2) > 0 else np.zeros((0,3))
        ref = np.array(self.ref_traj)         # shape (T, N, 3) or object

        # Save ground truth for both CFs
        np.save(f"{save_dir}/{self.label}_cf1_gt_{ts}.npy", gt1)
        np.save(f"{save_dir}/{self.label}_cf2_gt_{ts}.npy", gt2)

        # Save full reference trajectory array if available. The ref array may be ragged
        # (list of lists) so save using object dtype via numpy.save which preserves structure.
        if len(ref) > 0:
            np.save(f"{save_dir}/{self.label}_ref_full_{ts}.npy", ref, allow_pickle=True)

            # Also save a compact summary: first pose in each path
            try:
                r_first = np.array([r[0] for r in ref])   # shape (T, 3)
                np.save(f"{save_dir}/{self.label}_ref_firstpose_{ts}.npy", r_first)
            except Exception:
                pass

        # Create plots: combined trajectories + error between cf1 and cf2
        self.plot_compare_and_error(gt1, gt2, ref, ts, save_dir)


    def plot(self, gt, ref, ts, save_dir):
        # legacy single-GT plot retained for backward compatibility
        plt.figure(figsize=(6,6))

        # plot ground truth
        if len(gt) > 0:
            plt.plot(gt[:,0], gt[:,1], '-', label="Ground Truth")

        # reference trajectory — take first pose per published message
        if len(ref) > 0:
            try:
                r = np.array([r[0] for r in ref])   # shape (T, 3)
                plt.plot(r[:,0], r[:,1], '--', label="Reference Trajectory")
            except Exception:
                pass

        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title(self.label.upper() + " Trajectory Comparison")
        plt.grid(True)
        plt.legend()

        # --- force equal aspect scaling ---
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")

        fname = f"{save_dir}/{self.label}_compare_{ts}.png"
        plt.savefig(fname, dpi=200)
        plt.close()

        self.get_logger().info(f":white_check_mark: Saved PNG: {fname}")


    def plot_compare_and_error(self, gt1, gt2, ref, ts, save_dir):
        """Plot CF1 and CF2 trajectories, reference (if present), and error between CF1 and CF2.
        Saves two PNGs: trajectories and error-over-time.
        """
        plt.figure(figsize=(6,6))
        if gt1.shape[0] > 0:
            plt.plot(gt1[:,0], gt1[:,1], '-C0', label='CF1 Ground Truth')
        if gt2.shape[0] > 0:
            plt.plot(gt2[:,0], gt2[:,1], '-C1', label='CF2 Ground Truth')

        # plot reference first-pose summary if available
        if len(ref) > 0:
            try:
                r = np.array([r[0] for r in ref])
                plt.plot(r[:,0], r[:,1], '--r', label='Reference Trajectory')
            except Exception:
                pass

        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(self.label.upper() + ' — CF1 vs CF2 Trajectories')
        plt.grid(True)
        plt.legend()
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        fname = f"{save_dir}/{self.label}_cf1_cf2_traj_{ts}.png"
        plt.savefig(fname, dpi=200)
        plt.close()
        self.get_logger().info(f":white_check_mark: Saved trajectories PNG: {fname}")

        # --- compute and plot error between CF1 and CF2 ---
        if gt1.shape[0] == 0 or gt2.shape[0] == 0:
            return

        err, best_shift = self.compute_shifted_error(gt1, gt2)
        times = np.arange(err.shape[0])
        plt.figure(figsize=(8,3))
        plt.plot(times, err, '-k')
        plt.xlabel('Sample index')
        plt.ylabel('CF1-CF2 position error (m)')
        plt.title(f"CF1 vs CF2 Error (aligned shift={best_shift})")
        plt.grid(True)
        fname2 = f"{save_dir}/{self.label}_cf1_cf2_error_{ts}.png"
        plt.savefig(fname2, dpi=200)
        plt.close()
        self.get_logger().info(f":white_check_mark: Saved error PNG: {fname2}")


    def compute_shifted_error(self, a, b, max_shift=200):
        """Search for integer shift between a and b that minimizes XY MSE.
        Returns (err_array, best_shift) where err_array is per-sample Euclidean distance
        after alignment.
        """
        n1 = a.shape[0]
        n2 = b.shape[0]
        max_shift = min(max_shift, n1//2, n2//2)

        def xy_mse(x, y):
            return np.mean(np.sum((x[:,:2]-y[:,:2])**2, axis=1))

        best_shift = 0
        best_mse = None
        for shift in range(-max_shift, max_shift+1):
            if shift >= 0:
                L = min(n1-shift, n2)
                if L <= 5:
                    continue
                x = a[shift:shift+L,:]
                y = b[:L,:]
            else:
                s = -shift
                L = min(n1, n2-s)
                if L <= 5:
                    continue
                x = a[:L,:]
                y = b[s:s+L,:]

            mse = xy_mse(x, y)
            if best_mse is None or mse < best_mse:
                best_mse = mse
                best_shift = shift

        # construct error array with best shift
        if best_shift >= 0:
            L = min(n1-best_shift, n2)
            x = a[best_shift:best_shift+L,:]
            y = b[:L,:]
        else:
            s = -best_shift
            L = min(n1, n2-s)
            x = a[:L,:]
            y = b[s:s+L,:]

        err = np.linalg.norm(x - y, axis=1)
        return err, best_shift


def main():
    rclpy.init()
    node = TrajectoryCompareLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_and_plot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()