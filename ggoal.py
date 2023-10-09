#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2

# Publisher for the projected image
pub = None
bridge = CvBridge()

# Filter and project function
def process_point_cloud(cloud_msg):

    # Convert PointCloud2 to array
    gen = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
    points = np.array(list(gen))

    # Filter points based on X distance (adjust A and B as needed)
    A = 0.25  # example value
    B = 0.85  # example value
    filtered_points = points[np.logical_and(points[:, 0] > A, points[:, 0] < B)]

    # Assuming the points are in meters, a scaling factor will convert them into pixel coordinates
    scaling_factor = 100  # for example, 100 pixels per meter
    image_width = 500  # adjust as needed
    image_height = 500  # adjust as needed
    image = np.zeros((image_height, image_width), dtype=np.uint8)
    for point in filtered_points:
        y = int(point[1] * scaling_factor) + image_height // 2
        z = int(point[2] * scaling_factor) + image_width // 2
        if 0 <= y < image_height and 0 <= z < image_width:
            image[y, z] = 255

    # Convert the numpy image to a ROS Image message and publish
    image_msg = bridge.cv2_to_imgmsg(image, "mono8")
    image_msg.header = cloud_msg.header
    pub.publish(image_msg)
    
if __name__ == '__main__':
    rospy.init_node('pointcloud_to_image')
    rospy.Subscriber("/scan_3D", PointCloud2, process_point_cloud)
    pub = rospy.Publisher("/lidar_convert_image", Image, queue_size=10)
    rospy.spin()









