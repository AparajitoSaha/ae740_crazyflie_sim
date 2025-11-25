#!/usr/bin/env python3
"""Simple ROS2 test publisher to emit AS diagnostic messages so the AS logger records non-empty arrays.

Publishes:
 - /as/expert_errors : Float32MultiArray (P,) per timestep
 - /as/expert_probs  : Float32MultiArray (P,) per timestep
 - /as/expert_selected: Int32MultiArray (1,) per timestep

Run while as_logger is running to generate real log files for testing the plotter.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32MultiArray
import numpy as np
import time
import argparse

class ASTestPublisher(Node):
    def __init__(self, num_experts=2, rate=10.0):
        super().__init__('as_test_publisher')
        self.num_experts = int(num_experts)
        self.rate = float(rate)
        self.err_pub = self.create_publisher(Float32MultiArray, '/as/expert_errors', 10)
        self.prob_pub = self.create_publisher(Float32MultiArray, '/as/expert_probs', 10)
        self.sel_pub = self.create_publisher(Int32MultiArray, '/as/expert_selected', 10)
        self._t0 = self.get_clock().now().nanoseconds

    def publish_once(self, t):
        # create a drifting error signal per expert so the logger sees non-trivial data
        base = 0.1 + 0.01 * t
        errors = base + 0.05 * np.arange(self.num_experts)
        probs = np.exp(-errors)
        probs = probs / np.sum(probs)
        selected = int(t) % self.num_experts

        e_msg = Float32MultiArray()
        e_msg.data = [float(x) for x in errors]
        self.err_pub.publish(e_msg)

        p_msg = Float32MultiArray()
        p_msg.data = [float(x) for x in probs]
        self.prob_pub.publish(p_msg)

        s_msg = Int32MultiArray()
        s_msg.data = [int(selected)]
        self.sel_pub.publish(s_msg)

    def run(self, duration_s=5.0):
        end_t = time.time() + duration_s
        rate_hz = max(1.0, float(self.rate))
        period = 1.0 / rate_hz
        t = 0
        while time.time() < end_t:
            self.publish_once(t)
            t += 1
            time.sleep(period)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experts', type=int, default=2, help='Number of experts (P)')
    parser.add_argument('--rate', type=float, default=10.0, help='Publish rate (Hz)')
    parser.add_argument('--duration', type=float, default=6.0, help='Duration (s) to publish')
    args = parser.parse_args()

    rclpy.init()
    node = ASTestPublisher(num_experts=args.experts, rate=args.rate)
    try:
        node.get_logger().info(f'Publishing AS test messages: P={args.experts}, rate={args.rate}Hz, duration={args.duration}s')
        node.run(duration_s=args.duration)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
