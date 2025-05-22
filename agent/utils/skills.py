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
    PickObjectSrvResponse
)
from agent.utils.utils import *

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
        
def pick() -> PickObjectSrvResponse:
    """
    Pick an object.
    """
    rospy.wait_for_service("/moma/pick")
    try:
        pick_service = rospy.ServiceProxy("/moma/pick", PickObjectSrv)
        response = pick_service()
        return response
    except rospy.ServiceException as e:
        print("Service call failed:", e)
        

    
    