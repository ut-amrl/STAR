from typing import List

## ROS Service Calls
import rospy
import roslib; roslib.load_manifest('amrl_msgs')
from amrl_msgs.srv import (
    GetImageSrv,
    GetImageSrvResponse,
    GetImageAtPoseSrv, 
    GetImageAtPoseSrvRequest, 
    GetImageAtPoseSrvResponse, 
    PickObjectSrv, 
    PickObjectSrvResponse,
    GetVisibleObjectsSrv,
    GetVisibleObjectsSrvResponse,
    SemanticObjectDetectionSrv,
    SemanticObjectDetectionSrvResponse,
    FindObjectSrv,
    FindObjectSrvRequest,
    FindObjectSrvResponse
)
from agent.utils.utils import *

def get_image_path_for_simulation(viddir: str, frame: int) -> str:
    return os.path.join(viddir, f"Action_{frame:04d}_0_normal.png")

def navigate(pos: List[float], theta: float) -> GetImageAtPoseSrvResponse:
    """
    Navigate to a specific position and orientation.
    """
    rospy.wait_for_service("/moma/navigate")
    try:
        navigate_service = rospy.ServiceProxy("/moma/navigate", GetImageAtPoseSrv)
        request = GetImageAtPoseSrvRequest()
        request.x = pos[0]
        request.y = pos[1]
        if len(pos) == 3:
            request.z = pos[2]
        request.theta = theta
        response = navigate_service(request)
        return response
    except rospy.ServiceException as e:
        print("Service call failed:", e)
        
def observe() -> GetImageSrvResponse:
    """
    Observe the current environment.
    """
    rospy.wait_for_service("/moma/observe")
    try:
        observe_service = rospy.ServiceProxy("/moma/observe", GetImageSrv)
        response = observe_service()
        return response
    except rospy.ServiceException as e:
        print("Service call failed:", e)
        
def pick(query_text: str) -> PickObjectSrvResponse: # TODO need to change function signature later
    """
    Pick an object.
    """
    rospy.wait_for_service("/moma/pick_object")
    try:
        pick_service = rospy.ServiceProxy("/moma/pick_object", PickObjectSrv)
        response = pick_service(query_text=query_text)
        return response
    except rospy.ServiceException as e:
        print("Service call failed:", e)
        

def get_visible_objects() -> GetVisibleObjectsSrvResponse:
    """
    Get a list of visible objects.
    """
    rospy.wait_for_service("/moma/visible_objects")
    try:
        get_visible_objects_service = rospy.ServiceProxy("/moma/visible_objects", GetVisibleObjectsSrv)
        response = get_visible_objects_service()
        return response
    except rospy.ServiceException as e:
        print("Service call failed:", e)    

def detect_objects_owlv2(query_image: Image, query_cls: str) -> SemanticObjectDetectionSrvResponse:
    """
    Detect objects by class.
    """
    rospy.wait_for_service("/owlv2/semantic_object_detection")
    try:
        detect_service = rospy.ServiceProxy("/owlv2/semantic_object_detection", SemanticObjectDetectionSrv)
        req = SemanticObjectDetectionSrvRequest()
        req.query_image = query_image
        req.query_text = query_cls
        response = detect_service(req)
        return response
    except rospy.ServiceException as e:
        print("Service call failed:", e)
    
def find_object(query_text: str) -> FindObjectSrvResponse:
    """
    Find an object by class.
    """
    rospy.wait_for_service("/moma/find_object")
    try:
        find_object_service = rospy.ServiceProxy("/moma/find_object", FindObjectSrv)
        req = FindObjectSrvRequest()
        req.query_text = query_text
        response = find_object_service(req)
        return response
    except rospy.ServiceException as e:
        print("Service call failed:", e)
    
# # NOTE This is just a placeholder function for the simpliest case.    
# # This current implemntation is not correct
# def find_object(query_cls: str) -> List[List[int]]:
#     obs_response = observe()
    
#     detections = []
#     if obs_response.pano_images:
#         for pano_image in obs_response.pano_images:
#             response = detect_objects_owlv2(pano_image, query_cls)
#             if response.bounding_boxes.bboxes:
#                 detections += response.bounding_boxes.bboxes

#     detections.sort(key=lambda x: x.conf, reverse=True)
#     xyxys = []
#     for detection in detections:
#         xyxy = [int(x) for x in detection.xyxy]
#         xyxys.append(xyxy)
#     return xyxys