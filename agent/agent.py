import os
import json
from typing import Annotated, Sequence, TypedDict
from math import radians

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from utils.debug import get_logger
from utils.function_wrapper import FunctionsWrapper
from utils.utils import *
from utils.tools import (
    create_recall_any_tool, 
    create_recall_last_tool, 
    create_find_any_at_tool, 
)

from memory.memory import MilvusMemory

import rospy
import roslib; roslib.load_manifest('amrl_msgs')
from amrl_msgs.srv import (
    GetImageSrv,
    GetImageSrvRequest,
    GetImageAtPoseSrv, 
    GetImageAtPoseSrvRequest, 
    SemanticObjectDetectionSrv, 
    SemanticObjectDetectionSrvRequest,
    PickObjectSrv,
    PickObjectSrvRequest,
)


def from_find_at_to(state):
    if state["current_goal"].found:
        return "next"
    return "try_again"


class ObjectRetrievalPlan:
    def __init__(self):
        self.found = False
        self.task = None # a description of the plan
        self.task_type = None
        self.query_obj_desc = None
        self.query_obj_cls = None
        self.query_img = None # a past observation of the instance
        self.records = []
        
    def __str__(self):
        return str(self.__dict__)
        
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
        
        self.logger = get_logger()
        
        self.verbose = verbose
        
        self.llm_type, self.vlm_type = llm_type, vlm_type
        self.llm = self._llm_selector(self.llm_type)
        self.vlm, self.vlm_processor = self._vlm_selector(self.vlm_type)
    
        # if "qwen" in self.vlm_type:
        #     self.local_vlm = self.vlm
        #     self.local_vlm_processor = self.vlm_processor
        # else:
        #     self.local_vlm, self.local_vlm_processor =  self._vlm_selector("qwen")
        
    def set_memory(self, memory: MilvusMemory):
        self.memory = memory
        
        recall_any_tool = create_recall_any_tool(self.memory, self.llm, self.vlm)
        self.recall_tools = [recall_any_tool]
        self.recall_tool_definitions = [convert_to_openai_function(t) for t in self.recall_tools]
        
        prompt_dir = os.path.join(str(os.path.dirname(__file__)), "prompts", "agent")
        self.object_search_prompt = file_to_string(os.path.join(prompt_dir, 'object_search_prompt.txt'))
        self.recall_any_prompt = file_to_string(os.path.join(prompt_dir, 'recall_any_prompt.txt'))
        # Recall last seen prompts
        self.get_param_from_txt_prompt = file_to_string(os.path.join(prompt_dir, 'get_param_from_txt_prompt.txt'))
        self.find_instance_from_txt_prompt = file_to_string(os.path.join(prompt_dir, 'find_instance_from_txt_prompt.txt'))
        
        self._build_graph()
    
    def _llm_selector(self, llm_type):
        if 'gpt-4' in llm_type:
            import os
            llm = ChatOpenAI(model=llm_type, api_key=os.environ.get("OPENAI_API_KEY"))
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
    
    def initialize_object_search(self, state):
        messages = state["messages"]
        task = messages[0].content
        
        model = self.llm
        prompt = ChatPromptTemplate.from_messages(
            [
                ("ai", self.object_search_prompt),
                ("human", "{question}"),
            ]
        )
        model = prompt | model
        question = f"User Task: {task}"
        response = model.invoke({"question": question})
        task_info = eval(response.content)
        
        # TODO ask LLM to fill it in
        current_goal = ObjectRetrievalPlan()
        current_goal.task = f"Find {task_info['object_desc']}"
        current_goal.task_type = task_info['task_type']
        current_goal.query_obj_desc = task_info['object_desc']
        current_goal.query_obj_cls = task_info['object_class']
        
        self.logger.info(current_goal.__str__())
        
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
        question = current_goal.task
        
        model = self.llm
        prompt = ChatPromptTemplate.from_messages(
            [
                ("ai", self.get_param_from_txt_prompt),
                ("human", "{question}"),
            ]
        )
        model = prompt | model
        # question = f"User Task: Find {current_goal.query_obj_desc}"
        question = f"User Task: Find {current_goal.query_obj_cls}" # TODO
        response = model.invoke({"question": question})
        keywords = eval(response.content)
        
        self.logger.info(f"Searching vector db for keywords: {keywords}")
        
        query = ', or '.join(keywords)
        record_found = None
        for i in range(5):
            docs = self.memory.search_last_k_by_text(is_first_time=(i==0), query=query, k=10)
            if docs == '' or docs == None: # End of search
                break
            
            seen_records = set([record["id"] for record in current_goal.records])
            seen_positions = [eval(record["position"]) for record in current_goal.records]
            
            filtered_records = []
            for record in eval(docs):
                if record["id"] not in seen_records:
                    filtered_records.append(record)
            filtered_records2 = []
            for record in filtered_records:
                target_pos = eval(record["position"])
                discard = False
                for seen_pos in seen_positions:
                    if np.fabs(target_pos[0]-seen_pos[0]) < 0.4 and np.fabs(target_pos[1]-seen_pos[1]) < 0.4 and np.fabs(target_pos[2]-seen_pos[2]) < radians(45):
                        discard = True; break
                if not discard:
                    filtered_records2.append(record)
            filtered_records = filtered_records2
            if len(filtered_records) == 0:
                continue
            
            parsed_docs = parse_db_records_for_llm(filtered_records)
            
            # parsed_docs = parse_db_records_for_llm(eval(docs))
            
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
            
            self.logger.info(f"Retrived docs: {parsed_docs}")
            
            record_id = response.content
            self.logger.info(f"LLM response: {record_id}")
            
            if len(record_id) != 0:
                record_id = int(eval(record_id))
                docs = self.memory.get_by_id(record_id)
                
                self.logger.info(f"Record found: {docs}")
                
                record_found = eval(docs)
                break
            
        return record_found
    
    def _recall_last_seen_from_obs(self, obs):
        pass
    
    def recall_last_seen(self, state):
        current_goal = state["current_goal"]
        # TODO need to handle the case where no record is retrieved
        record = self._recall_last_seen_from_txt(current_goal)
        current_goal.records += record
        return {"current_goal": current_goal}
    
    def _send_getImage_request(self):
        rospy.wait_for_service("/Cobot/GetImage")
        try: 
            get_image = rospy.ServiceProxy("/Cobot/GetImage", GetImageSrv)
            request = GetImageSrvRequest()
            response = get_image(request)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
        return response
        
    def _find_at_by_txt(self, goal_x: float, goal_y: float, goal_theta: float, query_txt):
        self.logger.info(f"Finding object {query_txt} at ({goal_x:.2f}, {goal_y:.2f}, {goal_theta:.2f})")
        response = request_get_image_at_pose_service(goal_x, goal_y, goal_theta, logger=self.logger)
        depth = np.array(response.depth.data).reshape((response.depth.height, response.depth.width))
        rospy.loginfo(f"Checking instance at {goal_x}, {goal_y}, {goal_theta}")
        return is_txt_instance_observed(response.image, query_txt, depth, logger=self.logger)
        
    def find_at(self, state):
        current_goal = state["current_goal"]
        current_goal.found = False
        target = current_goal.curr_target()
        
        rospy.loginfo(f"current target: \n{target}")
        
        if type(target["position"]) == str:
            target["position"] = eval(target["position"])
        query_txt = current_goal.query_obj_cls
        
        import pdb; pdb.set_trace()
        
        goal_x, goal_y, goal_theta = target["position"][0], target["position"][1], target["position"][2]
        
        candidate_goals = [
            [goal_x, goal_y, goal_theta],
            [goal_x, goal_y, goal_theta-radians(60)],
            [goal_x, goal_y, goal_theta+radians(60)],
        ]
        for candidate_goal in candidate_goals:
            self.logger.info(f"Finding {query_txt} at ({candidate_goal[0]:.2f}, {candidate_goal[1]:.2f}, {candidate_goal[2]:.2f})")
            if self._find_at_by_txt(candidate_goal[0], candidate_goal[1], candidate_goal[2], query_txt):
                current_goal.found = True
                break
            
        # rospy.loginfo(f"Checking instance at {goal_x}, {goal_y}, {goal_theta}")
        # if is_viz_instance_observed(self.local_vlm, self.local_vlm_processor, img_message, object_desc):
        #     current_goal.found = True
        # if not current_goal.found:
        #     rospy.loginfo(f"Checking instance at {goal_x}, {goal_y}, {goal_theta-radians(30)}")
        #     img_message = self._send_getImageAtPose_request(goal_x, goal_y, goal_theta-radians(30))
        #     if is_viz_instance_observed(self.local_vlm, self.local_vlm_processor, img_message, object_desc):
        #         current_goal.found = True
        # if not current_goal.found:
        #     rospy.loginfo(f"Checking instance at {goal_x}, {goal_y}, {goal_theta+radians(30)}")
        #     img_message = self._send_getImageAtPose_request(goal_x, goal_y, goal_theta+radians(30))
        #     if is_viz_instance_observed(self.local_vlm, self.local_vlm_processor, img_message, object_desc):
        #         current_goal.found = True
                
        if current_goal.found:
            self.logger.info(f"Found {query_txt} at ({candidate_goal[0]:.2f}, {candidate_goal[1]:.2f}, {candidate_goal[2]:.2f})!")
            debug_vid(current_goal.curr_target(), "debug")
            
        return {"current_goal": current_goal}
    
    def pick(self, state):
        object_text = state["current_goal"].query_obj_cls
        curr_target = state["current_goal"].curr_target()
        import pdb; pdb.set_trace()
        response = request_pick_service(query_txt=object_text)
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
        
        workflow.add_node("initialize_object_search", lambda state: try_except_continue(state, self.initialize_object_search))
        workflow.add_node("terminate", lambda state: self.terminate(state))
        
        # workflow.add_node("recall_any_node", lambda state: try_except_continue(state, self.recall_any))
        # workflow.add_node("recall_any_action_node", ToolNode(self.recall_tools))
        # workflow.add_edge("initialize", "recall_any_node")
        # workflow.add_edge("recall_any_node", "recall_any_action_node")
        # workflow.add_edge("recall_any_action_node", "terminate")
        
        workflow.add_node("recall_last_seen", lambda state: try_except_continue(state, self.recall_last_seen))
        workflow.add_node("find_at", lambda state: self.find_at(state))
        workflow.add_node("pick", lambda state: self.pick(state))
        workflow.add_edge("initialize_object_search", "recall_last_seen")
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
        
        workflow.set_entry_point("initialize_object_search")
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
    