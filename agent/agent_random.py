from agent.utils.tools import *
from agent.agent import Agent
import random

class RandomAgent(Agent):
    def __init__(self, 
                 navigate_fn: Callable,
                 detect_fn: Callable,
                 pick_fn: Callable,
                 verbose: bool = False,
                 logdir: str = None,
                 logger_prefix: str = "",
                 is_interactive: bool = False,
                 robot_model: str = ""
                ):
        super().__init__(verbose, logdir, logger_prefix, is_interactive, robot_model)

        self.navigate_fn = navigate_fn
        self.detect_fn = detect_fn
        self.pick_fn = pick_fn

    def set_task(self, task_desc: str):
        return super().set_task(task_desc)
    
    def before_run(self, obj_cls: str, poses: List):
        self.obj_cls = obj_cls
        self.poses = poses
    
    def random_search(self):
        pos, theta = random.choice(self.poses)
        self.task.search_proposal = SearchProposal(
                summary="",
                instance_description="",
                position=pos,
                theta=theta,
                records=[]
            )
        if self.robot_model == "tiago":
            return
        response_nav = self.navigate_fn(pos, theta)
        if not response_nav.success:
            import pdb; pdb.set_trace()
            return
        response_detect = self.detect_fn(self.obj_cls)
        if not response_detect.success:
            return
        detected_ids = response_detect.ids
        if len(detected_ids) == 1:
            instance_id = detected_ids[0]
        else:
            instance_id = random.choice(detected_ids)
        self.task.search_proposal.visible_instances = response_detect.visible_instances
        response_pick = self.pick_fn(instance_id)
        self.task.search_proposal.has_picked = response_pick.success
        self.task.search_proposal.instance_name = response_pick.instance_uid
    
    def run(self, question: str):
        if self.obj_cls is None or self.poses is None:
            raise ValueError("Object class and poses must be set before running the agent (check `before_run`).")
        
        if self.logger:
            self.logger.info("=============== START ===============")
            self.logger.info(f"User question: {question}.")
            
        self.set_task(question)
        self.random_search()
        
        self.obj_cls, self.poses = None, None
        return {
            "task_result": self.task.search_proposal,
            "toolcalls": [],
        }