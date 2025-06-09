import os
import json
from typing import Annotated, Sequence, TypedDict
from math import radians
from collections import defaultdict
from typing import Callable, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from agent.utils.debug import get_logger
from agent.utils.function_wrapper import FunctionsWrapper
from agent.utils.utils import *
from agent.utils.tools import (
    create_find_specific_past_instance_tool, 
    create_best_guess_tool
)

from memory.memory import MilvusMemory

import rospy
from sensor_msgs.msg import Image
import roslib; roslib.load_manifest('amrl_msgs')
from amrl_msgs.srv import (
    GetImageSrvResponse,
    GetImageAtPoseSrvResponse, 
    PickObjectSrvResponse,
    GetVisibleObjectsSrvResponse,
    SemanticObjectDetectionSrvResponse
)

def from_initialize_object_search_to(state):
    return state["current_goal"].task_type 

def from_find_specific_past_instance_to(state):
    if state["current_goal"].found_in_mem:
        return "next"
    elif type(state["messages"][-1]) == ToolMessage:
        return "common_sense"
    return "retry" # TODO Fix the naming; We should default to common sense instead

def from_find_by_description_to(state):
    if state["current_goal"].found_in_mem:
        return "next"
    return "find_by_best_guess"

def from_find_by_frequency_to(state):
    if state["current_goal"].found_in_mem:
        return "find_specifici_past_instance"
    return "find_by_description"

def from_find_at_to(state):
    if state["current_goal"].found_in_world:
        return "next"
    return "retry"

class ObjectRetrievalPlan:
    def __init__(self):
        self.found_in_world: bool = False
        self.found_in_mem: bool = False
        self.task = None # a description of the plan
        self.task_type = None
        self.query_obj_desc = None
        self.query_obj_cls = None
        self.query_img = None # a past observation of the instance
        self.annotated_query_img = None
        self.has_picked: bool = False
        self.instance_uid: str = None # a unique identifier for the instance, 

        self.candidate_records_in_mem = []
        self.explored_records_in_mem = []
        self.candidate_records_in_world = []
        self.explored_records_in_world = []
        
    def __str__(self):
        return f"Find {self.query_obj_desc}; Task Type: {self.task_type}."
        
    def curr_target(self):
        if len(self.candidate_records_in_world) == 0:
            return None
        return copy.copy(self.candidate_records_in_world[-1])
    
    def next_target(self, target_type: str):
        if target_type == "world":
            if len(self.candidate_records_in_world) == 0:
                return None
            return self.candidate_records_in_world[-1]
        elif target_type == "mem":
            if len(self.candidate_records_in_mem) == 0:
                return None
            return self.candidate_records_in_mem[-1]
        else:
            raise ValueError(f"Invalid target type: {target_type}")

class Agent:
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        current_goal: Annotated[Sequence, replace_messages]
        task: Annotated[Sequence, replace_messages]
        output: Annotated[Sequence, replace_messages]
    
    def __init__(self,
        allow_recaption: bool = False,
        navigate_fn: Callable[[List[float], float], GetImageAtPoseSrvResponse] = None,
        find_object_fn: Callable[[str], List[List[int]]] = None, 
        observe_fn: Callable[[], GetImageSrvResponse] = None,
        pick_fn: Callable[[str], PickObjectSrvResponse] = None,
        visible_objects_fn: Callable[[], GetVisibleObjectsSrvResponse] = None,
        image_path_fn: Callable[[str], str] = None,
        llm_type: str = "gpt-4", 
        vlm_type: str = "gpt-4o", 
        verbose: bool = False
    ):
        self.allow_recaption = allow_recaption
        
        # TODO raise error if the functions are not callable in non-memory-only mode
        self.navigate_fn = navigate_fn
        self.observe_fn = observe_fn
        self.find_object_fn = find_object_fn
        self.pick_fn = pick_fn
        self.visible_objects_fn = visible_objects_fn
        self.image_path_fn = image_path_fn
        
        self.logger = get_logger()
        
        self.verbose = verbose
        
        self.llm_type, self.vlm_type = llm_type, vlm_type
        self.llm = self._llm_selector(self.llm_type)
        self.vlm, self.vlm_processor = self._vlm_selector(self.vlm_type)
        
        # NOTE: When we only want the model to make free-from response, 
        # we can directly call llm_raw/vlm_raw (without function wrapper) to avoid LLM formatting errors
        # We can clean up this code later
        self.llm_raw = ChatOpenAI(model=self.llm_type, api_key=os.environ.get("OPENAI_API_KEY"))
        self.vlm_raw = ChatOpenAI(model=self.vlm_type, api_key=os.environ.get("OPENAI_API_KEY"))
    
    def set_memory(self, memory: MilvusMemory):
        self.memory = memory
        
        recall_tool = create_find_specific_past_instance_tool(self.memory, self.llm, self.vlm, self.vlm_raw, self.allow_recaption, self.logger)
        self.recall_tools = [recall_tool]
        self.recall_tool_definitions = [convert_to_openai_function(t) for t in self.recall_tools]
        
        best_guess_tool = create_best_guess_tool(self.memory, self.llm, self.vlm, self.logger)
        self.best_guess_tools = [best_guess_tool]
        self.best_guess_tool_definitions = [convert_to_openai_function(t) for t in self.best_guess_tools]
        
        prompt_dir = os.path.join(str(os.path.dirname(__file__)), "prompts", "agent")
        self.object_search_prompt = file_to_string(os.path.join(prompt_dir, 'object_search_prompt.txt'))

        # Find specific past instance prompt        
        self.find_specific_past_instance_prompt = file_to_string(os.path.join(prompt_dir, 'find_specific_past_instance_prompt.txt'))
        self.prepare_find_from_specific_instance_prompt = file_to_string(os.path.join(prompt_dir, 'prepare_find_from_specific_instance_prompt.txt'))

        # Recall last seen prompts
        self.get_param_from_txt_prompt = file_to_string(os.path.join(prompt_dir, 'get_param_from_txt_prompt.txt'))
        self.find_instance_from_txt_prompt = file_to_string(os.path.join(prompt_dir, 'find_instance_from_txt_prompt.txt'))
        self.find_instance_from_obs_prompt = file_to_string(os.path.join(prompt_dir, 'find_instance_from_obs_prompt.txt'))
        self.same_instance_prompt = file_to_string(os.path.join(prompt_dir, 'same_instance_prompt.txt'))
        
        # Frequency
        self.recall_all_prompt = file_to_string(os.path.join(prompt_dir, 'recall_all_prompt.txt'))
        self.find_by_frequency_prompt = file_to_string(os.path.join(prompt_dir, 'find_by_frequency_prompt.txt'))
        
        self.contain_instance_prompt = file_to_string(os.path.join(prompt_dir, 'contain_instance_prompt.txt'))
        
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
    
    ##############################
    # Task-Level Search Initialization
    ##############################
    
    def initialize_object_search(self, state):
        messages = state["messages"]
        task = messages[0].content
        today = messages[1].content
        
        model = self.llm
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.object_search_prompt),
                ("human", "{question}"),
            ]
        )
        model = prompt | model
        question = (
            f"Fact: {today}\n"
            f"User Task: {task}"
        )
        response = model.invoke({"question": question})
        task_info = eval(response.content)
        
        # TODO ask LLM to fill it in
        current_goal = ObjectRetrievalPlan()
        current_goal.task = f"Find {task_info['object_desc']}."
        current_goal.task_type = task_info['task_type']
        if current_goal.task_type not in ["find_by_description" , "find_specific_past_instance", "find_by_frequency"]:
            raise ValueError(f"LLM failed to respond valid task type. LLM response: {current_goal.task_type}")
        current_goal.query_obj_desc = task_info['object_desc']
        current_goal.query_obj_cls = task_info['object_class']
        
        self.logger.info(current_goal.__str__())
        
        return {"task": task, "current_goal": current_goal}
    
    ##############################
    # Recall Last Seen (find_by_description)
    ##############################
    
    def _recall_last_seen_retriever(self, current_goal: ObjectRetrievalPlan, keyword_prompt: str, identification_prompt: str):
        model = self.llm
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", keyword_prompt),
                ("human", "{question}"),
            ]
        )
        model = prompt | model
        question = f"User Task: Find {current_goal.query_obj_desc}"
        # question = f"User Task: Find {current_goal.query_obj_cls}" 
        response = model.invoke({"question": question})
        keywords = eval(response.content)
        
        self.logger.info(f"Searching vector db for keywords: {keywords}")
        
        query = ', or '.join(keywords)
        
        record_found = []
        for i in range(5):
            docs = self.memory.search_last_k_by_text(is_first_time=(i==0), query=query, k=15)
            if docs == '' or docs == None: # End of search
                break
            
            # TODO verify this logic
            explored_record_ids = set([record["id"] for record in current_goal.explored_records_in_world])
            explored_positions = [eval(record["position"]) for record in current_goal.explored_records_in_world]
            
            filtered_records = []
            for record in eval(docs):
                if record["id"] not in explored_record_ids:
                    filtered_records.append(record)
            filtered_records2 = []
            for record in filtered_records:
                target_pos = eval(record["position"])
                discard = False
                for attempted_pos in explored_positions:
                    if np.fabs(target_pos[0]-attempted_pos[0]) < 0.4 and np.fabs(target_pos[1]-attempted_pos[1]) < 0.4 and np.fabs(target_pos[2]-attempted_pos[2]) < radians(45):
                        discard = True; break
                if not discard:
                    filtered_records2.append(record)
            filtered_records = filtered_records2
            if len(filtered_records) == 0:
                continue
            
            parsed_docs = parse_db_records_for_llm(filtered_records)
            
            model = self.llm
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "{docs}"),
                    ("system", identification_prompt),
                    ("human", "{question}"),
                ]
            )
            model = prompt | model
            question = f"User Task: {question}. Have you seen the instance user needs in your recalled moments?"
            response = model.invoke({"question": question, "docs": parsed_docs})
            self.logger.info(f"Retrived docs: {parsed_docs}")
            
            if len(response.content) == 0:
                continue
            if len(response.content) < 5: # TODO Fix me
                record_ids = response.content
                record_ids = [int(record_ids)]
            else:
                parsed_response = eval(response.content)
                record_ids = parsed_response["ids"]
                if type(record_ids) == str:
                    record_ids = eval(record_ids)
                record_ids = [int(i) for i in record_ids]
            
            self.logger.info(f"LLM response: {record_ids}")
            
            if len(record_ids) == 0:
                continue
            for record_id in record_ids:
                docs = self.memory.get_by_id(record_id)
                record_found += eval(docs)
            break
        return record_found
    
    def _recall_last_seen_from_txt(self, current_goal: ObjectRetrievalPlan):
        records = self._recall_last_seen_retriever(current_goal, self.get_param_from_txt_prompt, self.find_instance_from_txt_prompt)
        if len(records) == 0:
            return None
        return records[:1]
    
    def _recall_last_seen_from_obs(self, current_goal: ObjectRetrievalPlan):
        records = self._recall_last_seen_retriever(current_goal, self.get_param_from_txt_prompt, self.find_instance_from_obs_prompt)
        if len(records) == 0:
            return None
        self.logger.info(f"Verifying visual charateristics of between the query instance and candidate instances")
        image_messages = [get_vlm_img_message(current_goal.query_img, type=self.vlm_type)]
        for record in records:
            image = get_image_from_record(record, type="utf-8", resize=True)
            image_message = get_vlm_img_message(image, type=self.vlm_type)
            image_messages.append(image_message)
        
            question = f"I am looking for an instance that is likely matching the following description: {current_goal.query_obj_desc}. Did you see this instance on both images I sent you?"
            response = ask_chatgpt(self.vlm, self.same_instance_prompt, image_messages, question)
            if 'yes' in eval(response.content)["same_instance"]:
                self.logger.info(f"Found instance in record: {record}")
                return [record]
        self.logger.info(f"Failed to find this instance in memory!")
        return None
        
    def _recall_last_seen(self, current_goal: ObjectRetrievalPlan):
        current_goal.found_in_mem = False
        
        if current_goal.query_img:
            record = self._recall_last_seen_from_obs(current_goal)
        else:
            record = self._recall_last_seen_from_txt(current_goal)
        if record:
            current_goal.candidate_records_in_mem += record
            current_goal.candidate_records_in_world += record
            current_goal.found_in_mem = True
        else:
            current_goal.found_in_mem = False
                
        return {"current_goal": current_goal}
    
    def find_by_description(self, state):
        current_goal = state["current_goal"]
        return self._recall_last_seen(current_goal)
    
    ##############################
    # Recall Specific Episode (find_specific_past_instance)
    ##############################
    
    def find_specific_past_instance(self, state):
        last_message = state["messages"][-1]
        
        if type(last_message) == ToolMessage:
            current_goal = state["current_goal"]
            if len(last_message.content) != 0:
                try:
                    records = eval(last_message.content)
                except:
                    import pdb; pdb.set_trace()
                current_goal.found_in_mem = True
                current_goal.candidate_records_in_mem += records
                self.logger.info(f"Find {len(current_goal.candidate_records_in_mem)} record(s): {current_goal.candidate_records_in_mem}")
                return self._prepare_find_from_specific_instance(current_goal)
            else:
                return {"current_goal": current_goal}
        else:
            state["current_goal"].found_in_mem = False
            
            model = self.llm
            model = model.bind_tools(tools=self.recall_tool_definitions)
            
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.find_specific_past_instance_prompt),
                    ("human", "{question}"),
                ]
            )
            model = prompt | model
            question = f"User Task: {state['current_goal'].task}"
            response = model.invoke({"question": question})
            
            self.logger.info(f"Tool Call: {response.tool_calls}")
            return {"messages": response}
        
    def _prepare_find_from_specific_instance(self, current_goal: ObjectRetrievalPlan):
        for record in copy.copy(current_goal.candidate_records_in_mem):
            current_goal.candidate_records_in_mem = current_goal.candidate_records_in_mem[1:]
            current_goal.explored_records_in_mem.append(record)
            
            image = get_image_from_record(record, type="utf-8")
            image_message = [get_vlm_img_message(image, type="gpt-4o")]
            
            model = self.vlm
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.prepare_find_from_specific_instance_prompt),
                    HumanMessage(content=image_message),
                    ("human", "{question}")
                ]
            )
            
            model = prompt | model
            question = f"User task: {current_goal.task}. Are you seeing this instance in the image? If so, please describe the instance user is looking for."
            response = model.invoke({"question": question})
            parsed_response = eval(response.content)
            if "yes" in parsed_response["is_instance_observed"].lower():
                current_goal.query_obj_desc = parsed_response["instance_desc"]
                current_goal.query_img = image
                break
        # TODO need to handle the case where there's no record available
        self.logger.info(f"Based on past observation {current_goal.explored_records_in_mem[-1]['id']} - New goal: Find {current_goal.query_obj_desc}")
        return {"messages": response, "current_goal": current_goal}
    
    ##############################
    # Frequency Reasoning
    ##############################
    
    def _recall_all(self, current_goal: ObjectRetrievalPlan):
        docs = self.memory.search_all(current_goal.query_obj_desc)
        records = eval(docs)
        records = sorted(records, key=lambda x: x["id"])
        
        batch_size = 25 # Note: LLM can only takes in about 30-40 records
        record_ids = []
        for i in range(0, len(records), batch_size):
            parsed_llm_records = parse_db_records_for_llm(records[i:i+batch_size])
            
            model = self.llm
            prompt = ChatPromptTemplate.from_messages(
                [
                    MessagesPlaceholder("chat_history"),
                    ("system", self.recall_all_prompt),
                    ("human", "{question}"),
                ]
            )
            model = prompt | model
            question = f"User wants you to help find {current_goal.query_obj_desc}. To identify the instance user is referring to, you need to first recall all momented where you saw objects matching this description. Can you list ALL record ids for me?"
            
            response = model.invoke({"question": question, "chat_history": parsed_llm_records})
            
            parsed_response = eval(response.content)
            record_ids += [int(record_id) for record_id in parsed_response["selected_ids"]]
        record_ids = downsample_consecutive_ids(record_ids, rate=2)
        
        records_found = []
        for record in records:
            if record["id"] in record_ids:
                records_found.append(record)
                
        verified_records_found = []
        for record in records_found:
            image = get_image_from_record(record)
            ros_image = opencv_to_ros_image(image)
            response = request_bbox_detection_service(ros_image, current_goal.query_obj_cls)
            if len(response.bounding_boxes.bboxes) > 0:
                verified_records_found.append(record)
        import pdb; pdb.set_trace()
        records_found = verified_records_found
        
        # debug
        debugdir = "debug/recall_all"
        os.makedirs(debugdir, exist_ok=True)
        from pathlib import Path
        for file in Path(debugdir).glob("*.png"):
            file.unlink()
        for record in records_found:
            img = get_image_from_record(record)
            imgpath = os.path.join(debugdir, f"{record['id']}.png")
            cv2.imwrite(imgpath, img)
            
        # import pdb; pdb.set_trace()
            
        # verified_records_found = []
        # for record in records_found:
        #     image = get_image_from_record(record, type="utf-8")
        #     image_messages = [get_vlm_img_message(image, self.vlm_type)]
            
        #     question = f"User is looking for item: {current_goal.query_obj_desc}. Does this object appear on this image?"
        #     response = ask_chatgpt(self.vlm, self.contain_instance_prompt, image_messages, question)
        #     if "yes" in response.content:
        #         verified_records_found.append(record)
                
        self.logger.info(f"Found {len(verified_records_found)} record(s) about '{current_goal.query_obj_desc}': {verified_records_found}")
            
        return verified_records_found
    
    def find_by_frequency(self, state):
        current_goal = state["current_goal"]
        records = self._recall_all(current_goal)
        record_ids = [record["id"] for record in records]
        selected_record_ids = record_ids # downsample_consecutive_ids(record_ids, rate=3)
        
        records = []
        for record_id in selected_record_ids:
            records += eval(self.memory.get_by_id(record_id))
        
        selected_records = {}
        for record in records:
            if record["id"] in selected_record_ids:
                selected_records[record["id"]] = record
                
        question = f"I am looking for an instance that is likely matching the following description: {current_goal.query_obj_desc}. Did you see this instance on both images I sent you?"
        
        n = len(selected_record_ids)
        uf = UnionFind(n)
        for i in range(n):
            for j in range(i+1, n):
                if uf.connected(i, j):
                    continue
                record_id_i, record_id_j = selected_record_ids[i], selected_record_ids[j]
                record_i, record_j = selected_records[record_id_i], selected_records[record_id_j]
                image_i, image_j = get_image_from_record(record_i, type="utf-8"), get_image_from_record(record_j, type="utf-8")
                image_message_i, image_message_j = get_vlm_img_message(image_i, self.vlm_type), get_vlm_img_message(image_j, self.vlm_type)
                image_messages = [image_message_i, image_message_j]
                response = ask_chatgpt(self.vlm, self.same_instance_prompt, image_messages, question)
                if 'yes' in eval(response.content)["same_instance"]:
                    uf.union(i, j)
        
        groups = uf.get_groups()
        grouped_record_ids = [[selected_record_ids[i] for i in group] for group in groups]
        grouped_record_ids.sort(key=len, reverse=True)
        self.logger.info(f"All moments when I observed {current_goal.query_obj_cls}: {grouped_record_ids}")
        
        last_idx = last_multi_group_index(grouped_record_ids)
        
        if last_idx == -1:
            self.logger.info(f"I cannot meaningfullly answer this question as I only saw each {current_goal.query_obj_cls} once. Therefore, I can only help you find an arbitrary {current_goal.query_obj_cls}.")
            current_goal.find_in_mem = False
            return {"current_goal": current_goal}
        
        if last_idx == 0:
            current_goal.find_in_mem = True
            for record_id in grouped_record_ids[0]:
                current_goal.candidate_records_in_mem.append(selected_records[record_id])
            self.logger.info(f"Found instance in memory: {current_goal.candidate_records_in_mem}")
            return self._prepare_find_from_specific_instance(current_goal)
        
        grouped_record_ids[:last_idx]
        grouped_records = defaultdict(str)
        for i, record_ids in enumerate(grouped_record_ids):
            for record_id in record_ids:
                grouped_records[i] += ("- " + selected_records[record_id]["text"] + "\n")
        
        model = self.llm
        prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{chat_history}"),
                ("system", self.find_by_frequency_prompt),
                ("human", "{question}"),
            ]
        )
        model = prompt | model
        question = f"User wants you to help find {current_goal.query_obj_desc}. Your job now is to identify which instance is user referring to based on the past observations of each instance. Do you know the answer?"
        chat_history = "\n\n\n\n".join(f"id: {k}\n{v}" for k, v in grouped_records.items()) # json.dumps(grouped_records) 
        
        response = model.invoke({"question": question, "chat_history": chat_history})
        import pdb; pdb.set_trace()
        
        parsed_response = eval(response.content)
        
        object_id = int(parsed_response["id"])
        record_ids = grouped_record_ids[object_id]
        current_goal.find_in_mem = True
        for record_id in record_ids:
            current_goal.candidate_records_in_mem.append(selected_records[record_id]) 
        return self._prepare_find_from_specific_instance(current_goal)
        
    ##############################
    # Common Sense Reasoning
    ##############################
    def find_by_best_guess(self, state):
        current_goal = state["current_goal"]
        response = self.best_guess_tools[0].func(instance_description=current_goal.query_obj_desc)
        parsed_response = eval(response.content)
        reason = parsed_response["reasoning"]
        pos = parsed_response["position"]
        self.logger.info(f"LLM thinks we can find {current_goal.query_obj_desc} at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}). Reason: {reason}")
        
        dummy_record = {
            "id": -1,
            "position": pos
        }
        
        current_goal.candidate_records_in_world.append(dummy_record)
        return {"messages": response, "current_goal": current_goal}
        
        # return self._prepare_find_from_specific_instance(current_goal)
        
    ##############################
    # Robot Tools
    ##############################
    def _find_at_by_txt(self, goal_x: float, goal_y: float, goal_theta: float, query_txt):
        self.logger.info(f"Finding object {query_txt} at ({goal_x:.2f}, {goal_y:.2f}, {goal_theta:.2f})")
        response = request_get_image_at_pose_service(goal_x, goal_y, goal_theta, logger=self.logger)
        depth = np.array(response.depth.data).reshape((response.depth.height, response.depth.width))
        rospy.loginfo(f"Checking instance at {goal_x}, {goal_y}, {goal_theta}")
        return is_txt_instance_observed(response.image, query_txt, depth, logger=self.logger)
        
    def find_at(self, state):
        current_goal = state["current_goal"]
        current_goal.found_in_world = False
        target = current_goal.curr_target()
        
        self.logger.info(f"current target: {target}")
        
        if type(target["position"]) == str:
            target["position"] = eval(target["position"])
        query_txt = current_goal.query_obj_cls
        
        # NOTE currently, cobot takes (x,y,theta), while simulator takes (x.y.z)
        nav_response = self.navigate_fn(target["position"], target["position"][2]) # TODO: need to use theta instead of position in the future
        if nav_response.success:
            find_response = self.find_object_fn(query_txt) 
            current_goal.found_in_world = find_response.success
        
        if current_goal.found_in_world:
            position = target['position']
            self.logger.info(f"Found {query_txt} at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})!")
        
        return {"current_goal": current_goal}
        
        goal_x, goal_y, goal_theta = target["position"][0], target["position"][1], target["position"][2]
        
        candidate_goals = [
            [goal_x, goal_y, goal_theta],
            [goal_x, goal_y, goal_theta-radians(60)],
            [goal_x, goal_y, goal_theta+radians(60)],
        ]
        for candidate_goal in candidate_goals:
            self.logger.info(f"Finding {query_txt} at ({candidate_goal[0]:.2f}, {candidate_goal[1]:.2f}, {candidate_goal[2]:.2f})")
            if self._find_at_by_txt(candidate_goal[0], candidate_goal[1], candidate_goal[2], query_txt):
                current_goal.found_in_world = True
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
                
        if current_goal.found_in_world:
            self.logger.info(f"Found {query_txt} at ({candidate_goal[0]:.2f}, {candidate_goal[1]:.2f}, {candidate_goal[2]:.2f})!")
            # debug_vid(current_goal.curr_target(), "debug")
            
        return {"current_goal": current_goal}
    
    def pick(self, state):
        current_goal = state["current_goal"]
        query_text = current_goal.query_obj_cls
        self.logger.info(f"Attempting to pick up object: {query_text}")
        
        pick_response = self.pick_fn(query_text)
        
        current_goal.has_picked = pick_response.success
        current_goal.instance_uid = pick_response.instance_uid
        return {"current_goal": current_goal}
    
    def terminate(self, state):
        curr_target = state["current_goal"].curr_target()
        # debug_vid(curr_target, "debug")
        # print(curr_target)
        current_goal = state["current_goal"]
        return {"output": current_goal}
    
    def retrieval_terminate(self, state):
        next_target = state["current_goal"].next_target("mem")
        self.logger.info(f"Retrieval terminated. Output: {next_target}")
        if next_target is not None:
            next_target["output_type"] = "episode"
        return {"output": next_target}
    
    def _build_retrieval_graph(self):
        from langgraph.graph import END, StateGraph
        from langgraph.prebuilt import ToolNode
        
        workflow = StateGraph(Agent.AgentState)
        workflow.add_node("initialize_object_search", lambda state: try_except_continue(state, self.initialize_object_search))
        workflow.add_node("retrieval_terminate", lambda state: self.retrieval_terminate(state))
        workflow.add_edge("retrieval_terminate", END)
        
        workflow.add_node("find_by_description", lambda state: try_except_continue(state, self.find_by_description))
        workflow.add_node("find_specific_past_instance", lambda state: try_except_continue(state, self.find_specific_past_instance))
        
        # Memory tool nodes
        workflow.add_node("find_specific_past_instance_action_node", ToolNode(self.recall_tools))
        
        workflow.add_conditional_edges(
            "initialize_object_search",
            from_initialize_object_search_to,
            {
                "find_by_description": "find_by_description",
                "find_specific_past_instance": "find_specific_past_instance",
            }
        )
        workflow.add_edge("find_by_description", "retrieval_terminate")
        workflow.add_edge("find_specific_past_instance_action_node", "find_specific_past_instance")
        workflow.add_conditional_edges( # TODO this condition is incorrect
            "find_specific_past_instance",
            from_find_specific_past_instance_to,
            {
                "next": "retrieval_terminate",
                "common_sense": "retrieval_terminate",
                "retry": "find_specific_past_instance_action_node",
            }
        )
        
        workflow.set_entry_point("initialize_object_search")
        self.graph = workflow.compile()
        
    
    def _build_graph(self):
        from langgraph.graph import END, StateGraph
        from langgraph.prebuilt import ToolNode
        
        workflow = StateGraph(Agent.AgentState)
        
        workflow.add_node("initialize_object_search", lambda state: try_except_continue(state, self.initialize_object_search))
        workflow.add_node("terminate", lambda state: self.terminate(state))
        
        # Task and the corresponding action nodes
        workflow.add_node("find_by_description", lambda state: try_except_continue(state, self.find_by_description))
        workflow.add_node("find_specific_past_instance", lambda state: try_except_continue(state, self.find_specific_past_instance))
        workflow.add_node("find_by_frequency", lambda state: try_except_continue(state, self.find_by_frequency))
        workflow.add_node("find_by_best_guess", lambda state: try_except_continue(state, self.find_by_best_guess))
        
        # Memory tool nodes
        workflow.add_node("find_specific_past_instance_action_node", ToolNode(self.recall_tools))
        
        
        # Robot tool nodes
        workflow.add_node("find_at", lambda state: self.find_at(state))
        workflow.add_node("pick", lambda state: self.pick(state))
        
        workflow.add_conditional_edges(
            "initialize_object_search",
            from_initialize_object_search_to,
            {
                "find_by_description": "find_by_description",
                "find_specific_past_instance": "find_specific_past_instance",
                "find_by_frequency": "find_by_frequency",
            }
        )
        workflow.add_conditional_edges(
            "find_by_description",
            from_find_by_description_to,
            {
                "next": "find_at",
                "find_by_best_guess": "find_by_best_guess"
            }
        )
        workflow.add_edge("find_by_best_guess", "find_at")
        
        workflow.add_edge("find_specific_past_instance_action_node", "find_specific_past_instance")
        workflow.add_conditional_edges( # TODO this condition is incorrect
            "find_specific_past_instance",
            from_find_specific_past_instance_to,
            {
                "next": "find_by_description",
                "retry": "find_specific_past_instance_action_node", # TODO need to handle common sense
            }
        )
        
        workflow.add_edge("find_by_frequency", "find_by_description")
        workflow.add_conditional_edges(
            "find_by_frequency",
            from_find_by_frequency_to,
            {
                "find_specific_past_instance": "find_specific_past_instance",
                "find_by_description": "find_by_description",
            }
        )
        
        workflow.add_conditional_edges(
            "find_at",
            from_find_at_to,
            {
                "next": "pick",
                "retry": "find_by_description", # TODO
            },
        )
        workflow.add_edge("pick", "terminate")
        workflow.add_edge("terminate", END)
        
        workflow.set_entry_point("initialize_object_search")
        self.graph = workflow.compile()
        
    def build_graph(self, graph_type: str):
        if graph_type == "retrieval":
            self._build_retrieval_graph()
        elif graph_type == "full":
            self._build_graph()
        else:
            raise ValueError(f"Unsupported graph type: {graph_type}")
        
    def run(self, question: str, today: str, graph_type: str = "full"):
        self.build_graph(graph_type)
        
        inputs = { 
            "messages": [
                (("user", question)),
                (("system", today)),
            ],
        }
        state = self.graph.invoke(inputs)
        return state["output"]
        
        
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
    