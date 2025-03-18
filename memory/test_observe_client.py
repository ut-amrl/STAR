import os
from PIL import Image as PILImage
from sensor_msgs.msg import Image
import rospy
import numpy as np
import roslib; roslib.load_manifest('amrl_msgs')
from amrl_msgs.msg import *
from amrl_msgs.srv import RememberSrv, RememberSrvRequest

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

def load_images(image_paths):
        """Load images from file paths as PIL images."""
        images = []
        for path in image_paths:
            if os.path.exists(path):
                try:
                    img = PILImage.open(path)
                    images.append(img)
                    print(f"Loaded image: {path}")
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            else:
                print(f"File not found: {path}")
        return images
    
def call_observe_service(images):
    rospy.wait_for_service('/memory/observe')
    try:
        observe = rospy.ServiceProxy('/memory/observe', RememberSrv)
        video = []
        for image in images:
            if image.mode == "RGBA":
                image = image.convert("RGB")
            ros_image = numpy_to_ros_image(np.array(image))
            video.append(ros_image)
        request = RememberSrvRequest()
        import time
        request.timestamp = time.time()
        request.x = 1.
        request.y = 2.
        request.theta = 3.14
        request.video = video
        response = observe(request)
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
    
if __name__ == "__main__":
    rospy.init_node("observe_client")
    
    image_files = ["3dparty/VILA/demo_images/av.png"]
    images = load_images(image_files)
    call_observe_service(images)