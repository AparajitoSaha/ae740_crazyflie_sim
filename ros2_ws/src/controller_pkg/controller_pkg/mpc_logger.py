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
        self.ground_truth = []      # /cf_1/pose
        self.ref_traj     = []      # /all/mpc_trajectory

        # SUBSCRIBE
        self.create_subscription(
            PoseStamped,
            f"/{self.cf_name}/pose",
            self.cb_pose,
            10
        )

        self.create_subscription(
            Path,
            "/all/mpc_trajectory",
            self.cb_ref_traj,
            10
        )

        self.get_logger().info(f"[{self.label}] logger active: comparing /{self.cf_name}/pose vs /all/mpc_trajectory")


    # --- Callbacks ---
    def cb_pose(self, msg: PoseStamped):
        p = msg.pose.position
        self.ground_truth.append([p.x, p.y, p.z])


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
        save_dir = "mpc_logs"
        os.makedirs(save_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        gt  = np.array(self.ground_truth)     # shape (T, 3)
        ref = np.array(self.ref_traj)         # shape (T, N, 3)

        np.save(f"{save_dir}/{self.label}_gt_{ts}.npy",  gt)
        np.save(f"{save_dir}/{self.label}_ref_{ts}.npy", ref)

        self.plot(gt, ref, ts, save_dir)


    def plot(self, gt, ref, ts, save_dir):
        plt.figure(figsize=(6,6))

        # plot ground truth
        if len(gt) > 0:
            plt.plot(gt[:,0], gt[:,1], '-', label="Ground Truth")

        # reference trajectory — take first pose per published message
        if len(ref) > 0:
            # extract first pose in each path
            r = np.array([r[0] for r in ref])   # shape (T, 3)
            plt.plot(r[:,0], r[:,1], '--', label="Reference Trajectory")

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