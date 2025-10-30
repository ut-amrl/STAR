import os
from typing import List
import cv2

from agent.utils.utils import opencv_to_ros_image

# ROS1 Service Calls
# TODO need to support ROS2
import rospy
import roslib; roslib.load_manifest('amrl_msgs')
from amrl_msgs.srv import (
    GetImageAtPoseSrv,
    GetImageAtPoseSrvResponse, 
    PickObjectSrv, 
    PickObjectSrvResponse,
    DetectVirtualHomeObjectSrv,
    DetectVirtualHomeObjectSrvResponse,
)

current_pose = "shelf"

# def dummy_navigate(pos: List[float], theta: float) -> GetImageAtPoseSrvResponse:
def dummy_navigate(req) -> GetImageAtPoseSrvResponse:
    """
    Dummy navigate to a specific position and orientation.
    """
    pos = [req.x, req.y, 0]
    theta = req.theta
    response = GetImageAtPoseSrvResponse()
    def dummy_image(pos: List[float], theta: float):
        image_dir = os.path.join("example", "toy_data_1")
        global current_pose
        
        import numpy as np
        transl_tol, rot_tol = 0.1, 0.1
        if np.linalg.norm(pos - np.array([0.5, 1.0, 0.0])) < transl_tol and np.fabs(theta - 0.0) < rot_tol:
            imgpath = "000012.png"
            current_pose = "shelf"
        elif np.linalg.norm(pos - np.array([1.0, -0.5, 0.0])) < transl_tol and np.fabs(theta - 1.57) < rot_tol:
            imgpath = "000013.png"
            current_pose = "coffeetable"
        elif np.linalg.norm(pos - np.array([-3.0, 2.0, 0.0])) < transl_tol and np.fabs(theta + 1.57) < rot_tol:
            imgpath = "000014.png" 
            current_pose = "studydesk"
        else:
            return False, None
            
        imgpath = os.path.join(image_dir, imgpath)
        cv_img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        return True, opencv_to_ros_image(cv_img)
    
    success, ros_img = dummy_image(pos, theta)
    if not success:
        response.success = False
        return response
    
    response.success = True
    response.pano_images = [ros_img]
    return response

def dummy_detect(req) -> DetectVirtualHomeObjectSrvResponse:
    query_txt = req.query_text
    response = DetectVirtualHomeObjectSrvResponse()
    query_txt = query_txt.lower()
    if query_txt != "book":
        response.success = False
        return response
    
    def dummy_image():
        global current_pose
        
        image_dir = os.path.join("example", "toy_data_1", "detections")
        if current_pose == "shelf":
            imgpath = "000012.png"
        elif current_pose == "coffeetable":
            imgpath = "000013.png"
        elif current_pose == "studydesk":
            imgpath = "000014.png"
        else:
            raise ValueError("Unknown current pose")
        
        imgpath = os.path.join(image_dir, imgpath)
        cv_img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        return opencv_to_ros_image(cv_img)
    
    global current_pose
    if current_pose == "shelf":
        ids = [1]
    elif current_pose == "coffeetable":
        ids = []
    elif current_pose == "studydesk":
        ids = [2]
    else:
        raise ValueError("Unknown current pose")
    
    response.success = True
    response.ids = ids
    response.images = [dummy_image()]
    return response

def dummy_pick(req) -> PickObjectSrvResponse:
    object_id = req.instance_id
    
    response = PickObjectSrvResponse()
    response.success = False
    if current_pose == "shelf" and object_id == 1:
        response.success = True
        response.instance_uid = "book_1"
    elif current_pose == "studydesk" and object_id == 2:
        response.success = True
        response.instance_uid = "book_2"
    return response

if __name__ == "__main__":
    rospy.init_node("quick_start_server_node")
    rospy.Service('/moma/navigate', GetImageAtPoseSrv, dummy_navigate)
    rospy.loginfo("Service /moma/navigate ready")
    rospy.Service('/moma/detect_virtual_home_object', DetectVirtualHomeObjectSrv, dummy_detect)
    rospy.loginfo("Service /moma/detect_virtual_home_object ready")
    rospy.Service('/moma/pick_object', PickObjectSrv, dummy_pick)
    rospy.loginfo("Service /moma/pick_object ready")
    rospy.spin()