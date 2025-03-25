import os
import json
from typing import Annotated, Sequence, TypedDict
from math import radians

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from utils.function_wrapper import FunctionsWrapper
from utils.utils import *
from utils.tools import (
    create_recall_any_tool, 
    create_recall_last_tool, 
    create_find_any_at_tool, 
    is_instance_observed
)

from memory.memory import MilvusMemory

import rospy
import roslib; roslib.load_manifest('amrl_msgs')
from amrl_msgs.srv import GetImageAtPoseSrv, GetImageAtPoseSrvRequest


def from_find_at_to(state):
    if state["current_goal"].found:
        return "next"
    return "try_again"


class ObjectRetrievalPlan:
    def __init__(self):
        self.found = False
        self.plan = None # a description of the plan
        self.object = None # a semantic class label for object detection
        self.object_obs = None # a past observation of the instance
        self.records = []
        
    def curr_target(self):
        if len(self.records) == 0:
            return None
        return copy.copy(self.records[-1])
    

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_goal: Annotated[Sequence, replace_messages]
    task: Annotated[Sequence, replace_messages]
    

class Agent:
    def __init__(self, llm_type: str = "gpt-4", vlm_type: str = "gpt-4o", verbose: bool = False):
        self.verbose = verbose
        
        self.llm_type, self.vlm_type = llm_type, vlm_type
        self.llm = self._llm_selector(self.llm_type)
        self.vlm, self.vlm_processor = self._vlm_selector(self.vlm_type)
    
        if "qwen" in self.vlm_type:
            self.local_vlm = self.vlm
            self.local_vlm_processor = self.vlm_processor
        else:
            self.local_vlm, self.local_vlm_processor =  self._vlm_selector("qwen")
        
    def set_memory(self, memory: MilvusMemory):
        self.memory = memory
        
        recall_any_tool = create_recall_any_tool(self.memory, self.llm, self.vlm)
        self.recall_tools = [recall_any_tool]
        self.recall_tool_definitions = [convert_to_openai_function(t) for t in self.recall_tools]
        
        prompt_dir = os.path.join(str(os.path.dirname(__file__)), "prompts", "agent")
        self.recall_any_prompt = file_to_string(os.path.join(prompt_dir, 'recall_any_prompt.txt'))
        # Recall last seen prompts
        self.get_param_from_txt_prompt = file_to_string(os.path.join(prompt_dir, 'get_param_from_txt_prompt.txt'))
        self.find_instance_from_txt_prompt = file_to_string(os.path.join(prompt_dir, 'find_instance_from_txt_prompt.txt'))
        
        self._build_graph()
    
    def _llm_selector(self, llm_type):
        if 'gpt-4' in llm_type:
            import os
            llm = ChatOpenAI(model='gpt-4o', api_key=os.environ.get("OPENAI_API_KEY"))
            return FunctionsWrapper(llm)
        else:
            raise ValueError("Unsupported LLM type!")
        
    def _vlm_selector(self, vlm_type):
        if 'gpt-4' in vlm_type:
            import os
            model = ChatOpenAI(model='gpt-4o', api_key=os.environ.get("OPENAI_API_KEY"))
            model = FunctionsWrapper(model)
            processor = None
        elif 'qwen' in vlm_type:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype="auto", device_map={"": 1}
            )
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        else:
            raise ValueError("Unsupported VLM type!")
        return model, processor
    
    def initialize(self, state):
        messages = state["messages"]
        task = messages[0].content
        
        # TODO ask LLM to fill it in
        current_goal = ObjectRetrievalPlan()
        current_goal.plan = "bring me a cup from a table"
        current_goal.object = "cup"
        
        return {"task": task, "current_goal": current_goal}
    
    def recall_any(self, state):
        model = self.llm
        model = model.bind_tools(tools=self.recall_tool_definitions)
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("ai", self.recall_any_prompt),
                ("human", "{question}"),
            ]
        )
        model = prompt | model
        question = f"User Task: {state['task']}"
        response = model.invoke({"question": question})
        return {"messages": [response]}
    
    def _recall_last_seen_from_txt(self, current_goal):
        question = current_goal.plan
        
        model = self.llm
        prompt = ChatPromptTemplate.from_messages(
            [
                ("ai", self.get_param_from_txt_prompt),
                ("human", "{question}"),
            ]
        )
        model = prompt | model
        question = f"User Task: {question}"
        response = model.invoke({"question": question})
        keywords = eval(response.content)
        
        query = ', or '.join(keywords)
        record_found = None
        for _ in range(5):
            docs = self.memory.search_last_k_by_text(is_first_time=True, query=query)
            if docs == '':
                break
            
            parsed_docs = parse_db_records_for_llm(eval(docs))
            
            model = self.llm
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "{docs}"),
                    ("ai", self.find_instance_from_txt_prompt),
                    ("human", "{question}"),
                ]
            )
            model = prompt | model
            question = f"User Task: {question}. Have you seen the instance user needs in your recalled moments?"
            response = model.invoke({"question": question, "docs": parsed_docs})
            
            record_id = response.content
            if len(record_id) == 0:
                continue
            record_id = int(eval(record_id))
            docs = self.memory.get_by_id(record_id)
            record_found = eval(docs)
            
        return record_found
    
    def _recall_last_seen_from_obs(self, obs):
        pass
    
    def recall_last_seen(self, state):
        current_goal = state["current_goal"]
        # TODO need to handle the case where no record is retrieved
        record = self._recall_last_seen_from_txt(current_goal)
        current_goal.records.append(record[0])
        return {"current_goal": current_goal}
    
    def _send_getImageAtPose_request(self, goal_x:float, goal_y:float, goal_theta:float):
        rospy.wait_for_service("/Cobot/GetImageAtPose")
        img_message = None
        try: 
            get_image_at_pose = rospy.ServiceProxy("/Cobot/GetImageAtPose", GetImageAtPoseSrv)
            request = GetImageAtPoseSrvRequest()
            request.x = goal_x
            request.y = goal_y
            request.theta = goal_theta
            response = get_image_at_pose(request)
            
            pil_img = ros_image_to_pil(response.image)
            img_data = pil_to_utf8(pil_img)
            img_message = get_vlm_img_message(img_data)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
        return img_message
        
    def find_at(self, state):
        current_goal = state["current_goal"]
        current_goal.found = False
        target = current_goal.curr_target()
        if type(target["position"]) == str:
            target["position"] = eval(target["position"])
        object_desc = "a cup" # TODO retrieve this from current_goal
            
        goal_x, goal_y, goal_theta = target["position"][0], target["position"][1], target["position"][2]
        img_message = self._send_getImageAtPose_request(goal_x, goal_y, goal_theta)
        if is_instance_observed(self.local_vlm, self.local_vlm_processor, img_message, object_desc):
            current_goal.found = True
        if not current_goal.found:
            img_message = self._send_getImageAtPose_request(goal_x, goal_y, goal_theta-radians(15))
            if is_instance_observed(self.local_vlm, self.local_vlm_processor, img_message, object_desc):
                current_goal.found = True
        if not current_goal.found:
            img_message = self._send_getImageAtPose_request(goal_x, goal_y, goal_theta+radians(15))
            if is_instance_observed(self.local_vlm, self.local_vlm_processor, img_message, object_desc):
                current_goal.found = True
                
        if current_goal.found:
            debug_vid(current_goal.curr_target(), "debug")
        
        return {"current_goal": current_goal}
    
    def pick(self, state):
        curr_target = state["current_goal"].curr_target()
        
        import pdb; pdb.set_trace()
        return
    
    def terminate(self, state):
        curr_target = state["current_goal"].curr_target()
        debug_vid(curr_target, "debug")
        print(curr_target)
        pass
    
    def _build_graph(self):
        from langgraph.graph import END, StateGraph
        from langgraph.prebuilt import ToolNode
        
        workflow = StateGraph(AgentState)
        
        workflow.add_node("initialize", lambda state: self.initialize(state))
        workflow.add_node("terminate", lambda state: self.terminate(state))
        
        # workflow.add_node("recall_any_node", lambda state: try_except_continue(state, self.recall_any))
        # workflow.add_node("recall_any_action_node", ToolNode(self.recall_tools))
        # workflow.add_edge("initialize", "recall_any_node")
        # workflow.add_edge("recall_any_node", "recall_any_action_node")
        # workflow.add_edge("recall_any_action_node", "terminate")
        
        workflow.add_node("recall_last_seen", lambda state: try_except_continue(state, self.recall_last_seen))
        workflow.add_node("find_at", lambda state: self.find_at(state))
        workflow.add_node("pick", lambda state: self.pick(state))
        workflow.add_edge("initialize", "recall_last_seen")
        workflow.add_edge("recall_last_seen", "find_at")
        workflow.add_conditional_edges(
            "find_at",
            from_find_at_to,
            {
                "next": "pick",
                "try_again": "recall_last_seen",
            },
        )
        workflow.add_edge("pick", "terminate")
        workflow.add_edge("terminate", END)
        
        workflow.set_entry_point("initialize")
        self.graph = workflow.compile()
        
    def run(self, question: str):
        inputs = { 
            "messages": [
                (("user", question)),
            ],
        }
        state = self.graph.invoke(inputs)
        
        
if __name__ == "__main__":
    from utils.memloader import remember_from_paths
    memory = MilvusMemory("test1", obs_savepth="data/cobot/cobot_test_1", db_ip='127.0.0.1')
    memory.reset()
    inpaths = [
        "/robodata/taijing/RobotMem/data/captions/cobot/2025-03-10-17-01-55_VILA1.5-8b_3_secs.json",
        "/robodata/taijing/RobotMem/data/captions/cobot/2025-03-10-17-00-15_VILA1.5-8b_3_secs.json",
    ]
    t_offset = 1738952666.5530548-len(inpaths)*86400 + 86400
    remember_from_paths(memory, inpaths, t_offset, viddir="/robodata/taijing/RobotMem/data/images")
    
    agent = Agent()
    agent.set_memory(memory)
    # agent.run(question="Today is 2025-02-07. Where is the coffee that was on a table yesterday?")
    agent.run(question="Bring me a cup from a table.")
    