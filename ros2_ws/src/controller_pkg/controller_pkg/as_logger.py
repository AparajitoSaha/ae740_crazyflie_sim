#!/usr/bin/env python3
"""ROS2 node to log AS-module data for as_analysis_plotter.

Saves:
 - {label}_gt_{ts}.npy              -> ground truth positions (T x 3)
 - {label}_expert_preds_{ts}.npy    -> raw expert predictions (list or array)
 - {label}_expert_errors_{ts}.npy   -> per-expert errors (T x P)
 - {label}_expert_probs_{ts}.npy    -> expert probability trace (T x P)
 - {label}_expert_selected_{ts}.npy -> selected expert indices (T,)

By default the node subscribes to:
 - /cf_1/pose (geometry_msgs/PoseStamped)
 - /as/expert_preds (std_msgs/Float32MultiArray)
 - /as/expert_errors (std_msgs/Float32MultiArray)
 - /as/expert_probs (std_msgs/Float32MultiArray)
 - /as/expert_selected (std_msgs/Int32MultiArray)

All topics are configurable via ROS2 parameters.
"""
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, Int32MultiArray

import numpy as np
import os
from datetime import datetime


def _array_from_multiarray(msg):
    """Convert ROS MultiArray message into numpy array using layout dims if available."""
    data = np.asarray(msg.data, dtype=float)
    # Try to reshape according to layout dims if present
    try:
        dims = msg.layout.dim
        if dims:
            shape = tuple(int(d.size) for d in dims)
            if np.prod(shape) == data.size:
                return data.reshape(shape)
    except Exception:
        pass
    return data


def _coerce_preds_to_P_T_3(preds_list):
    """Coerce various preds_list shapes into (P, T, 3) if possible.

    preds_list: list of numpy arrays or a single array
    Returns: np.ndarray shape (P, T, 3) or raises ValueError
    """
    arr = np.asarray(preds_list, dtype=object)
    if arr.size == 0:
        raise ValueError('empty')

    # If it's already a numeric array
    try:
        numeric = np.asarray(preds_list)
    except Exception:
        numeric = None

    if isinstance(numeric, np.ndarray) and numeric.dtype != object:
        if numeric.ndim == 3 and numeric.shape[2] == 3:
            p0, p1, p2 = numeric.shape
            # decide orientation
            if p0 <= 10 and p1 > p0:
                return numeric  # (P, T, 3)
            if p1 <= 10 and p0 > p1:
                return numeric.transpose(1, 0, 2)  # (T, P, 3) -> (P, T, 3)
            # ambiguous: choose smaller dim as P
            if p0 <= p1:
                return numeric
            return numeric.transpose(1, 0, 2)

    # If object list: try stack
    try:
        stacked = np.stack([np.asarray(x) for x in arr])
    except Exception:
        # last chance: try converting to list and analyze
        lst = list(arr)
        # case: list of length P, each (T,3)
        if all(getattr(x, 'ndim', 0) == 2 and x.shape[-1] == 3 for x in lst):
            # stack into (P, T, 3)
            return np.stack([np.asarray(x) for x in lst])
        # case: list of length T, each (P,3)
        if all(getattr(x, 'ndim', 0) == 2 and x.shape[-1] == 3 for x in lst):
            a = np.stack([np.asarray(x) for x in lst])
            return a.transpose(1, 0, 2)
        raise ValueError('cannot coerce preds_list to (P,T,3)')

    # stacked is numeric
    if stacked.ndim == 3 and stacked.shape[2] == 3:
        s0, s1, s2 = stacked.shape
        if s0 <= 10 and s1 > s0:
            return stacked
        if s1 <= 10 and s0 > s1:
            return stacked.transpose(1, 0, 2)
        if s0 <= s1:
            return stacked
        return stacked.transpose(1, 0, 2)

    raise ValueError('unsupported stacked preds shape')


def _coerce_matrix_to_T_P(mat_list):
    """Coerce list/array to a (T, P) numeric 2D array if possible.

    Accepts:
    - list of length T where each element is (P,) -> stack -> (T,P)
    - list of length P where each element is (T,) -> stack -> (P,T) -> transpose
    - already (T,P) or (P,T) numeric arrays
    """
    a = np.asarray(mat_list, dtype=object)
    if a.size == 0:
        raise ValueError('empty')

    try:
        numeric = np.asarray(mat_list)
    except Exception:
        numeric = None

    if isinstance(numeric, np.ndarray) and numeric.dtype != object:
        if numeric.ndim == 2:
            t0, t1 = numeric.shape
            # heuristics: larger dim is time
            if t0 >= t1:
                return numeric
            return numeric.T
        if numeric.ndim == 1:
            return numeric.reshape(-1, 1)

    # object list
    lst = list(a)
    if all(np.asarray(x).ndim == 1 for x in lst):
        # either list length T of (P,) or list length P of (T,)
        shapes = [np.asarray(x).shape[0] for x in lst]
        # if lengths equal -> ambiguous; choose to stack as (len(lst), n) and transpose if needed
        stacked = np.stack([np.asarray(x) for x in lst])
        if stacked.ndim == 2:
            s0, s1 = stacked.shape
            if s0 >= s1:
                return stacked
            return stacked.T

    raise ValueError('cannot coerce to (T,P)')


class ASLogger(Node):
    def __init__(self):
        super().__init__('as_logger')

        # parameters for topics and label
        self.declare_parameter('gt_topic', '/cf_1/pose')
        self.declare_parameter('preds_topic', '/as/expert_preds')
        self.declare_parameter('errors_topic', '/as/expert_errors')
        self.declare_parameter('probs_topic', '/as/expert_probs')
        self.declare_parameter('selected_topic', '/as/expert_selected')
        self.declare_parameter('label', 'as')

        self.gt_topic = self.get_parameter('gt_topic').value
        self.preds_topic = self.get_parameter('preds_topic').value
        self.errors_topic = self.get_parameter('errors_topic').value
        self.probs_topic = self.get_parameter('probs_topic').value
        self.selected_topic = self.get_parameter('selected_topic').value
        self.label = self.get_parameter('label').value

        # buffers
        self.gt_positions = []           # list of [x,y,z]
        self.expert_preds = []           # list of incoming preds (each a numpy array)
        self.expert_errors_list = []     # list of arrays (could be appended or single publish)
        self.expert_probs_list = []
        self.expert_selected_list = []

        # Subscribers
        self.create_subscription(PoseStamped, self.gt_topic, self.cb_gt_pose, 10)
        self.create_subscription(Float32MultiArray, self.preds_topic, self.cb_preds, 10)
        self.create_subscription(Float32MultiArray, self.errors_topic, self.cb_errors, 10)
        self.create_subscription(Float32MultiArray, self.probs_topic, self.cb_probs, 10)
        self.create_subscription(Int32MultiArray, self.selected_topic, self.cb_selected, 10)

        self.get_logger().info(f"AS logger listening: gt={self.gt_topic}, preds={self.preds_topic}, errors={self.errors_topic}, probs={self.probs_topic}, selected={self.selected_topic}")

    # Callbacks
    def cb_gt_pose(self, msg: PoseStamped):
        p = msg.pose.position
        self.gt_positions.append([p.x, p.y, p.z])

    def cb_preds(self, msg: Float32MultiArray):
        arr = _array_from_multiarray(msg)
        # store each incoming preds message as-is; later we will save a list/stack
        self.expert_preds.append(arr)

    def cb_errors(self, msg: Float32MultiArray):
        arr = _array_from_multiarray(msg)
        self.expert_errors_list.append(arr)

    def cb_probs(self, msg: Float32MultiArray):
        arr = _array_from_multiarray(msg)
        self.expert_probs_list.append(arr)

    def cb_selected(self, msg: Int32MultiArray):
        arr = np.asarray(msg.data, dtype=int)
        # If it's a single value, append; if vector, extend/append depending on length
        if arr.size == 1:
            self.expert_selected_list.append(int(arr[0]))
        else:
            # add as sequence (e.g., selected vector over time)
            self.expert_selected_list.extend(arr.tolist())

    # Saving
    def save_all(self):
        save_dir = os.path.join(os.path.dirname(__file__), 'mpc_logs')
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        # save GT (always)
        gt = np.array(self.gt_positions) if len(self.gt_positions) > 0 else np.zeros((0, 3))
        np.save(f"{save_dir}/{self.label}_gt_{ts}.npy", gt)

        # helper to save an object placeholder
        def _save_placeholder(fname):
            np.save(fname, np.array([], dtype=object), allow_pickle=True)

        # save expert preds: could be list of arrays (ragged) or stackable
        preds_fname = f"{save_dir}/{self.label}_expert_preds_{ts}.npy"
        if len(self.expert_preds) > 0:
            try:
                # try to coerce into (P, T, 3)
                preds_out = _coerce_preds_to_P_T_3(self.expert_preds)
                np.save(preds_fname, preds_out, allow_pickle=False)
            except Exception:
                # fallback: save as object array
                np.save(preds_fname, np.array(self.expert_preds, dtype=object), allow_pickle=True)
        else:
            # no preds received — save empty placeholder so downstream tools find a file
            _save_placeholder(preds_fname)

        # save expert errors: prefer provided errors, otherwise try to compute from preds + gt
        errors_fname = f"{save_dir}/{self.label}_expert_errors_{ts}.npy"
        if len(self.expert_errors_list) > 0:
            try:
                errs = np.stack(self.expert_errors_list)
                np.save(errors_fname, errs, allow_pickle=False)
            except Exception:
                np.save(errors_fname, np.array(self.expert_errors_list, dtype=object), allow_pickle=True)
        else:
            # try to compute from preds and gt if possible
            # Prefer computing from preds if preds exist
            computed_saved = False
            if len(self.expert_preds) > 0 and gt.shape[0] > 0:
                try:
                    # try to coerce preds -> (P, T, 3) then compute errors
                    preds_coerced = _coerce_preds_to_P_T_3(self.expert_preds)
                    # preds_coerced is (P, T, 3); compute errors (T, P)
                    P, T, _ = preds_coerced.shape
                    errs = np.zeros((T, P))
                    for i in range(P):
                        errs[:, i] = np.linalg.norm(preds_coerced[i] - gt, axis=1)
                    np.save(errors_fname, errs, allow_pickle=False)
                    computed_saved = True
                except Exception:
                    computed_saved = False

            if not computed_saved:
                _save_placeholder(errors_fname)

        # save expert probs
        probs_fname = f"{save_dir}/{self.label}_expert_probs_{ts}.npy"
        if len(self.expert_probs_list) > 0:
            try:
                # coerce to (T, P) if possible
                try:
                    probs_arr = _coerce_matrix_to_T_P(self.expert_probs_list)
                    np.save(probs_fname, probs_arr, allow_pickle=False)
                except Exception:
                    probs = np.stack(self.expert_probs_list)
                    np.save(probs_fname, probs, allow_pickle=False)
            except Exception:
                np.save(probs_fname, np.array(self.expert_probs_list, dtype=object), allow_pickle=True)
        else:
            _save_placeholder(probs_fname)

        # save selected trace
        sel_fname = f"{save_dir}/{self.label}_expert_selected_{ts}.npy"
        if len(self.expert_selected_list) > 0:
            sel = np.asarray(self.expert_selected_list, dtype=int)
            np.save(sel_fname, sel)
        else:
            # save empty int array
            np.save(sel_fname, np.asarray([], dtype=int))

        self.get_logger().info(f"Saved AS logs to {save_dir} with label {self.label} (ts={ts})")

    def _compute_errors_from_preds(self, preds_list, gt):
        """Compute errors (T x P) from preds_list and gt.

        preds_list can be:
        - list of arrays where each array is (T,3) for each expert (P entries)
        - list of arrays over time where each entry is (P,3)
        - stackable to (P,T,3) or (T,P,3)
        Returns: errors (T, P)
        """
        # reuse logic similar to as_analysis_plotter
        arr = np.asarray(preds_list, dtype=object)
        if arr.dtype == object:
            try:
                stacked = np.stack([p for p in arr])
                preds = stacked
            except Exception:
                preds = np.asarray(arr.tolist())
        else:
            preds = arr

        if preds.ndim == 3:
            p0, p1, p2 = preds.shape
            if p0 <= 10 and p1 == gt.shape[0] and p2 == 3:
                # (P, T, 3)
                P = p0
                T = p1
                errors = np.zeros((T, P))
                for i in range(P):
                    errors[:, i] = np.linalg.norm(preds[i] - gt, axis=1)
                return errors
            elif p0 == gt.shape[0] and p2 == 3:
                # (T, P, 3)
                T = p0
                P = p1
                errors = np.zeros((T, P))
                for i in range(P):
                    errors[:, i] = np.linalg.norm(preds[:, i, :] - gt, axis=1)
                return errors

        # unsupported shapes -> raise and fallback to placeholder
        raise ValueError('Unsupported preds shape for computing errors')


def main():
    rclpy.init()
    node = ASLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_all()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
