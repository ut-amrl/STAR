#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
import roslib; roslib.load_manifest('amrl_msgs')
from amrl_msgs.srv import SemanticObjectDetectionSrv, SemanticObjectDetectionSrvRequest

def numpy_to_ros_image(np_image, encoding="rgb8"):
    """ Convert a NumPy image (OpenCV) to a ROS Image message. """
    ros_image = Image()
    ros_image.height = np_image.shape[0]
    ros_image.width = np_image.shape[1]
    ros_image.encoding = encoding  # "bgr8" for OpenCV images, "rgb8" for PIL images
    ros_image.is_bigendian = 0
    ros_image.step = np_image.shape[1] * np_image.shape[2]  # width * channels
    ros_image.data = np_image.tobytes()
    return ros_image

def load_image_as_ros_msg(image_path: str):
    cv_image = cv2.imread(image_path)
    if cv_image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Convert BGR to RGB (OWLv2 expects RGB)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    ros_image = numpy_to_ros_image(cv_image, encoding="rgb8")
    return ros_image

def main():
    rospy.init_node("owlv2_client_test")

    # ---- Hardcoded inputs ----
    image_path = "debug/cup.png"  # Replace with a valid image path
    query_text = "cup"

    # Load and prepare request
    ros_image = load_image_as_ros_msg(image_path)
    req = SemanticObjectDetectionSrvRequest(query_image=ros_image, query_text=query_text)

    # Wait for service
    rospy.loginfo("Waiting for OWLv2 service...")
    rospy.wait_for_service('/owlv2/semantic_object_detection')

    try:
        detect = rospy.ServiceProxy('/owlv2/semantic_object_detection', SemanticObjectDetectionSrv)
        res = detect(req)
        rospy.loginfo(f"Received {len(res.bounding_boxes.bboxes)} bounding boxes.")
        for i, box in enumerate(res.bounding_boxes.bboxes):
            print(f"[{i}] {box.label} ({box.conf:.2f}) at {box.xyxy}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

if __name__ == "__main__":
    main()
