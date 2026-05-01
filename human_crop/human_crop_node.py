# coding: utf-8
import os
import datetime
from pathlib import Path
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from bboxes_ex_msgs.msg import BoundingBoxes
import message_filters
from cv_bridge import CvBridge

class HumanCropNode(Node):
    def __init__(self):
        super().__init__('human_crop_node')
        self.bridge = CvBridge()

        default_base_dir = Path(__file__).resolve().parents[1] / 'human_crops'
        base_dir = Path(os.environ.get(
            'HUMAN_CROP_SAVE_DIR',
            str(default_base_dir),
        ))
        session_name = datetime.datetime.now().strftime('session_%Y%m%d_%H%M%S')
        self.save_dir = base_dir / session_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_count = 0

        shigure_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        color_subscriber = message_filters.Subscriber(
            self, CompressedImage, '/rs/color/compressed',
            qos_profile=shigure_qos)

        bbox_subscriber = message_filters.Subscriber(
            self, BoundingBoxes, '/bounding_boxes',
            qos_profile=shigure_qos)

        self.time_synchronizer = message_filters.TimeSynchronizer(
            [bbox_subscriber, color_subscriber], 1000)
        self.time_synchronizer.registerCallback(self.callback)

        self.get_logger().info('human_crop_node started')
        self.get_logger().info(f'saving to: {self.save_dir}')

    def callback(self, bbox_src: BoundingBoxes, color_img_src: CompressedImage):
        color_img = self.bridge.compressed_imgmsg_to_cv2(color_img_src)
        h, w = color_img.shape[:2]

        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        for bbox in bbox_src.bounding_boxes:
            if bbox.class_id != 'person':
                continue

            x1 = int(bbox.xmin)
            y1 = int(bbox.ymin)
            x2 = int(bbox.xmax)
            y2 = int(bbox.ymax)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            cropped = color_img[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            filename = str(self.save_dir / f'person_{stamp}_{self.save_count:05d}.jpg')
            cv2.imwrite(filename, cropped)
            self.get_logger().info(f'Saved: {filename}')
            self.save_count += 1

def main(args=None):
    rclpy.init(args=args)
    node = HumanCropNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()