import sys
import os
import re
from typing import List, Dict, Optional
from typing import Annotated, Sequence, TypedDict
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from langchain.tools import StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from memory.memory import MilvusMemory
from agent.utils.utils import *

class SearchInstance:
    def __init__(self):
        self.inst_desc: str = ""
        self.inst_viz_path: str = None
        self.annotated_inst_viz = None
        self.annotated_bbox = None
        
        self.found_in_world: bool = False
        self.found_in_memory: bool = False
        self.past_observations: List[Dict] = []  # TODO likely needs to be refactored
        
    def __str__(self):
        return f"SearchInstance(inst_desc={self.inst_desc}, inst_viz_path={self.inst_viz_path}, found_in_memory={self.found_in_memory})"
    
    def to_message(self, type: str = "mem"):
        message = []
        
        if type == "mem":
            txt_msg = [{"type": "text", "text": f"Current Memory Search Instance and its most update-to-update status: {self.__str__()}"}]
        else:
            txt_msg = [{"type": "text", "text": f"Current World Search Instance and its most update-to-update status: {self.__str__()}"}]
        
        message += txt_msg
        
        if self.inst_viz_path:
            try:
                img = PILImage.open(self.inst_viz_path).convert("RGB")  # RGBA to preserve color + alpha
            except Exception:
                # If the image cannot be opened, skip adding the image message
                return message
            img = img.resize((512, 512), PILImage.BILINEAR)
            # Draw the record ID with background
            draw = ImageDraw.Draw(img)
            text = f"Current Search Instance: {self.inst_desc}"
            font = ImageFont.load_default()
            text_size = draw.textbbox((0, 0), text, font=font)  # (left, top, right, bottom)
            padding = 4
            bg_rect = (
                text_size[0] - padding,
                text_size[1] - padding,
                text_size[2] + padding,
                text_size[3] + padding
            )
            draw.rectangle(bg_rect, fill=(0, 0, 0))  # Black background
            draw.text((text_size[0], text_size[1]), text, fill=(255, 255, 255), font=font)
            
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img = base64.b64encode(buffer.getvalue()).decode("utf-8")
            img_msg = [get_vlm_img_message(img, type="gpt")]
            
            message += img_msg
        
        return message
        

def create_db_txt_search_tool(memory: MilvusMemory):
    class TextRetrieverInput(BaseModel):
            x: str = Field(description="The query that will be searched by the vector similarity-based retriever.\
                                Text embeddings of this description are used. There should always be text in here as a response! \
                                Based on the question and your context, decide what text to search for in the database. \
                                This query argument should be a phrase such as 'a crowd gathering' or 'a green car driving down the road'.\
                                The query will then search your memories for you.")

    txt_retriever_tool = StructuredTool.from_function(
        func=lambda x: memory.search_by_text(x),
        name="retrieve_from_text",
        description="Search and return information from your video memory in the form of captions",
        args_schema=TextRetrieverInput
        # coroutine= ... <- you can specify an async method if desired as well
    )
    return [txt_retriever_tool]
    
def create_db_txt_search_with_time_tool(memory: MilvusMemory):

    class TextRetrieverInputWithTime(BaseModel):
        x: str = Field(
            description="The query that will be searched by the vector similarity-based retriever. "
                        "Text embeddings of this description are used. There should always be text in here as a response! "
                        "Based on the question and your context, decide what text to search for in the database. "
                        "This query argument should be a phrase such as 'a crowd gathering' or 'a green car driving down the road'."
        )
        start_time: str = Field(
            description="Start search time in YYYY-MM-DD HH:MM:SS format. Only search for observations made after this timestamp. Following unix standard, this can only be a time after 1970-01-01 00:00:00."
        )
        end_time: str = Field(
            description="End search time in YYYY-MM-DD HH:MM:SS format. Only search for observations made before this timestamp. Following unix standard, this can only be a time larger than the start time, and also before current time 2025-03-01."
        )

    txt_time_retriever_tool = StructuredTool.from_function(
        func=lambda x, start_time, end_time: memory.search_by_txt_and_time(x, start_time, end_time),
        name="retrieve_from_text_with_time",
        description="Search and return information from your video memory in the form of captions, filtered by time range.",
        args_schema=TextRetrieverInputWithTime
    )
    return [txt_time_retriever_tool]

def create_db_txt_backward_search_tool(memory: MilvusMemory):
    pass 

def create_recall_best_match_tool(
    memory: MilvusMemory,
    llm,
    llm_raw,
    vlm,
    vlm_raw,
    logger=None
) -> StructuredTool:
    
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        output: Annotated[Sequence, replace_messages] = None
        
    def from_agent_to(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "generate"
        else:
            return "action"
    
    class BestMatchAgent:
        def __init__(self, memory, llm, llm_raw, vlm, vlm_raw, logger=None):
            self.memory = memory
            self.llm = llm
            self.llm_raw = llm_raw
            self.vlm = vlm
            self.vlm_raw = vlm_raw
            self.logger = logger
            
            self.db_retriever_tools = create_db_txt_search_tool(memory)
            self.recall_tool_definitions = [convert_to_openai_function(t) for t in self.db_retriever_tools]
            self.db_retriever_with_time_tools = create_db_txt_search_with_time_tool(memory)
            self.recall_tool_with_time_definitions = [convert_to_openai_function(t) for t in self.db_retriever_with_time_tools]
            
            self.tools = None
            self.tool_definitions = None
            
            prompt_dir = str(os.path.dirname(__file__)) + '/../prompts/best_match_tool/'
            self.agent_prompt = file_to_string(prompt_dir+'agent_prompt.txt')
            self.agent_gen_only_prompt = file_to_string(prompt_dir+'agent_gen_only_prompt.txt')
            self.generate_prompt = file_to_string(prompt_dir+'generate_prompt.txt')
            
            self.agent_call_count = 0
            self.previous_tool_requests = "I have already used the following retrieval tools and the results are included below. Do not repeat them:\n"
            
            self.system_prompt = "You are a memory retrieval agent. Your task is to help the user recall all memory records that are relevant to their current goal. You have access to tools that let you search memory by description, with or without time constraints. You should first determine what information to retrieve, then use the available tools to search the memory database. After retrieving results, analyze them carefully and select only those that are relevant to the user’s task. You may call memory tools multiple times if needed, but avoid repeating previous requests. Your final output should be a list of memory records that best match the user’s intent."    
        
        def agent(self, state: AgentState):
            messages = state["messages"]
            
            model = self.llm
            
            if self.agent_call_count > 2:
                prompt = self.agent_gen_only_prompt
            else:
                prompt= self.agent_prompt
                model = model.bind_tools(self.tool_definitions)
                
            chat_prompt = ChatPromptTemplate.from_messages([
                MessagesPlaceholder(variable_name="chat_history"),
                (("human"), self.previous_tool_requests),
                ("system", self.system_prompt),
                ("user", prompt),
                ("human", "{question}"),
            ])
            chained_model = chat_prompt | model
            question  = f"Task Context: {self.context}\n" \
                        f"Instance user is looking for: {self.instance_description}\n" \
                        "Could you please help me recall the best matching memory records based on this information using tools you have?"
            
            response = chained_model.invoke({"question": question, "chat_history": messages[1:]})
        
            if self.logger:
                self.logger.info(f"[BEST_MATCH] Received agent response. Tool calls: {bool(response.tool_calls)}")
        
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call['name'] != "__conversational_response":
                        args = re.sub(r'^\{(.*)\}$', r'(\1)', str(tool_call['args'])) # remove curly braces
                        self.previous_tool_requests += f" {tool_call['name']} tool with the arguments: {args}.\n"
                        
            self.agent_call_count += 1
            return {"messages": [response]}
        
        def generate(self, state: AgentState):
            messages = state["messages"]
            
            model = self.llm_raw
            prompt = self.generate_prompt
            chat_prompt = ChatPromptTemplate.from_messages(
                [
                    MessagesPlaceholder("chat_history"),
                    (("human"), self.previous_tool_requests),
                    ("system", self.system_prompt),
                    ("system", prompt),
                    ("human", "{question}"),
                ]
            )
            chained_model = chat_prompt | model
            question = f"Task Context: {self.context}\n" \
                       f"Instance user is looking for: {self.instance_description}\n" \
                        "Based on the information retrieved from your previous tool calls, could you list out all memory record IDs that best match the user intent?"
                        
            response = chained_model.invoke({"question": question, "chat_history": messages[1:]})
            record_ids = [int(i.strip()) for i in response.content.split(",") if i.strip().isdigit()]
            
            db_messages = filter_retrieved_record(messages[:])
            retrieved_messages = {r["id"]: r for r in db_messages}
            
            records = []
            for id in record_ids:
                if id in retrieved_messages:
                    records.append(retrieved_messages[id])
                    
            if self.logger:
                self.logger.info(f"[BEST_MATCH] Generated response with {len(records)} records based on records: {records}")
            
            return {"messages": [response], "output": records}
        
        
        def build_graph(self):
            workflow = StateGraph(AgentState)
            
            workflow.add_node("agent", lambda state: try_except_continue(state, self.agent))
            workflow.add_node("action", ToolNode(self.tools))
            workflow.add_node("generate", lambda state: try_except_continue(state, self.generate))
            
            workflow.add_conditional_edges(
                "agent",
                from_agent_to,
                {
                    "action": "action",
                    "generate": "generate",
                },
            )
            workflow.add_edge('action', 'agent')
            workflow.add_edge("generate", END)
            
            workflow.set_entry_point("agent")
            self.graph = workflow.compile()

        def run(self, context: str, instance_description: str, search_start_time: Optional[str] = None, search_end_time: Optional[str] = None) -> List[Dict]:
            if self.logger:
                self.logger.info(
                    f"[BEST_MATCH] Running tool with context: {context}, "
                    f"instance_description: {instance_description}, "
                    f"search_start_time: {search_start_time}, search_end_time: {search_end_time}"
                )
            
            if (search_start_time is None) != (search_end_time is None):
                raise ValueError("Both search_start_time and search_end_time must be provided together, or both must be None.")
            with_Time = (search_start_time is not None and search_end_time is not None)
            if with_Time:
                self.tools = self.db_retriever_with_time_tools
                self.tool_definitions = self.recall_tool_with_time_definitions
            else:
                self.tools = self.db_retriever_tools
                self.tool_definitions = self.recall_tool_definitions
            
            self.build_graph()

            self.agent_call_count = 0
            self.previous_tool_requests = "I have already used the following retrieval tools and the results are included below. Do not repeat them:\n"

            if search_start_time and search_end_time:
                query = f"Between {search_start_time} and {search_end_time}, have you observed {instance_description}? If so, when?"
            else:
                query = f"Have you observed {instance_description}?"
            inputs = { "messages": [
                    (("user", query))
                ]
            }
            
            self.context = context
            self.instance_description = instance_description
            self.search_start_time = search_start_time
            self.search_end_time = search_end_time
            
            state = self.graph.invoke(inputs)
            return state.get("output", [])

    tool_runner = BestMatchAgent(memory, llm, llm_raw, vlm, vlm_raw, logger)
    
    class BestMatchInput(BaseModel):
        context: str = Field(
            description="High-level context or purpose for calling this tool. \
                        For example, 'The user is about to interact with the object', or 'This is for confirming past observations before navigation', or 'This is to figure out which instance user is referring to' \
                        This helps the agent reason about what the goal is beyond just finding the object."
        )
        instance_description: str = Field(
            description="Describe the object you're trying to recall from memory. \
                         Example: 'a red suitcase near the kitchen entrance'"
        )
        search_start_time: Optional[str] = Field(
            default=None,
            description="(Optional) Start time in 'YYYY-MM-DD HH:MM:SS'. Results after this time will be considered."
        )
        search_end_time: Optional[str] = Field(
            default=None,
            description="(Optional) End time in 'YYYY-MM-DD HH:MM:SS'. Results before this time will be considered."
        )
        
    return [StructuredTool.from_function(
        func=lambda context, instance_description, search_start_time=None, search_end_time=None: 
            tool_runner.run(context, instance_description, search_start_time, search_end_time),
        name="recall_best_match",
        description="Recalls the single best-matching memory observation based on description and optional time range.",
        args_schema=BestMatchInput
    )]

def create_recall_last_seen_tool(
    memory: MilvusMemory,
    llm,
    vlm,
    logger=None
) -> StructuredTool:
    
    pass

def create_recall_all_tool(
    memory: MilvusMemory,
    llm,
    vlm,
    logger=None
) -> StructuredTool:
    pass

def create_memory_inspection_tool(memory: MilvusMemory) -> StructuredTool:

    class MemoryInspectionInput(BaseModel):
        record_id: int = Field(
            description="The ID of the memory record you want to inspect. The image associated with this record will be returned in base64 (utf-8) format."
        )

    def _inspect_memory_record(record_id: int) -> str:
        # Should return base64-encoded (utf-8) image string for the record
        docs = memory.get_by_id(record_id)
        if docs is None or len(docs) == 0:
            return "" # "No record found with the given ID."
        record = eval(docs)[0]

        image_path_fn = lambda vidpath, frame: os.path.join(vidpath, f"{frame:06d}.png")
        vidpath = record["vidpath"]
        start_frame = record["start_frame"]
        end_frame = record["end_frame"]
        start_frame, end_frame = int(start_frame), int(end_frame)
        frame = (start_frame + end_frame) // 2
        imgpath = image_path_fn(vidpath, frame)
        return {record_id : imgpath}

        # img = get_image_from_record(record, type="utf-8", resize=True)
        # img_msg = get_vlm_img_message(img, type="gpt")
        # return [img_msg]

    inspection_tool = StructuredTool.from_function(
        func=_inspect_memory_record,
        name="inspect_memory_record",
        description="Given a memory record ID, return its associated visual observation as a base64-encoded image string.",
        args_schema=MemoryInspectionInput
    )

    return [inspection_tool]

def create_determine_search_instance_tool(
    memory: MilvusMemory,
    llm,
    llm_raw,
    vlm,
    vlm_raw,
    logger=None
) -> StructuredTool:
    
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        output: Annotated[Sequence, replace_messages] = None
        
    def from_decide_to(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        else:
            return "action"
        
    class DetermineSearchInstanceAgent:
        def __init__(self, memory, llm, llm_raw, vlm, vlm_raw, logger=None):
            self.memory = memory
            self.llm = llm
            self.llm_raw = llm_raw
            self.vlm = vlm
            self.vlm_raw = vlm_raw
            self.logger = logger
            
            self.setup_tools(memory)
            
            prompt_dir = str(os.path.dirname(__file__)) + '/../prompts/determine_search_instance_tool/'
            self.decide_prompt = file_to_string(prompt_dir+'decide_prompt.txt')
            self.decide_gen_only_prompt = file_to_string(prompt_dir+'decide_gen_only_prompt.txt')
            
            self.decide_call_count = 0
            self.previous_tool_requests = "I have already used the following retrieval tools and the results are included below. Do not repeat them:\n"
            
        def setup_tools(self, memory: MilvusMemory):
            self.tools = create_memory_inspection_tool(memory)
            self.tool_definitions = [convert_to_openai_function(t) for t in self.tools]
            
        def _parse_memory_records(self, messages: Sequence[BaseMessage]) -> List[Dict]:
            image_paths = {}
            for msg in filter(lambda x: isinstance(x, ToolMessage), messages):
                if not msg.content:
                    continue
                try:
                    val = eval(msg.content)
                    for k,v in val.items():
                        image_paths[int(k)] = v
                except Exception:
                    continue
            memory_messages = []
            for record in self.memory_records:
                msg = [{"type": "text", "text": parse_db_records_for_llm(record)}]
                memory_messages += msg
                
                if int(record["id"]) in image_paths:
                    image_path = image_paths[int(record["id"])]
                    img = PILImage.open(image_path).convert("RGB")  # RGBA to preserve color + alpha
                    img = img.resize((512, 512), PILImage.BILINEAR)
                    
                    # Draw the record ID with background
                    draw = ImageDraw.Draw(img)
                    text = f"record_id: {record['id']}"
                    font = ImageFont.load_default()
                    text_size = draw.textbbox((0, 0), text, font=font)  # (left, top, right, bottom)
                    padding = 4
                    bg_rect = (
                        text_size[0] - padding,
                        text_size[1] - padding,
                        text_size[2] + padding,
                        text_size[3] + padding
                    )
                    draw.rectangle(bg_rect, fill=(0, 0, 0))  # Black background
                    draw.text((text_size[0], text_size[1]), text, fill=(255, 255, 255), font=font)
                    
                    buffer = BytesIO()
                    img.save(buffer, format="PNG")
                    img = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    memory_messages += [get_vlm_img_message(img, type="gpt")]
            return memory_messages
            
        def decide(self, state: AgentState):
            messages = state["messages"]
            
            model = self.vlm
            if self.decide_call_count > 2:
                prompt = self.decide_gen_only_prompt
            else:
                prompt = self.decide_prompt
                model = model.bind_tools(self.tool_definitions)
                
            chat_prompt = ChatPromptTemplate.from_messages([
                ("human", "{memory_records}"),
                # ("human", self.previous_tool_requests),
                ("user", prompt),
                ("human", "{question}"),
            ])
            chained_model = chat_prompt | model
            question  = f"User Task: {self.user_task}\n" \
                        f"History Summary: {self.history_summary}\n" \
                        f"Current Task: {self.current_task}\n"
                        
            memory_messages = self._parse_memory_records(messages)
            response = chained_model.invoke({"question": question, "memory_records": memory_messages})
        
            if self.logger:
                self.logger.info(f"[DETERMINE_SEARCH_INSTANCE] decide() - Tool calls present: {bool(getattr(response, 'tool_calls', None))}")
        
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call['name'] != "__conversational_response":
                        args = re.sub(r'^\{(.*)\}$', r'(\1)', str(tool_call['args'])) # remove curly braces
                        self.previous_tool_requests += f" {tool_call['name']} tool with the arguments: {args}.\n"
                        
            self.decide_call_count += 1
            return {"messages": [response]}
        
        def generate(self, state: AgentState):
            messages = state["messages"]
            keys_to_check_for = ["found_in_memory", "instance_desc", "record_id"]
            
            prompt = self.decide_gen_only_prompt
            chat_prompt = ChatPromptTemplate.from_messages([
                ("human", "{memory_records}"),
                ("user", prompt),
                ("human", "{question}"),
            ])
            model = self.vlm
            chained_model = chat_prompt | model
            question  = f"User Task: {self.user_task}\n" \
                        f"History Summary: {self.history_summary}\n" \
                        f"Current Task: {self.current_task}\n"
                        
            memory_messages = self._parse_memory_records(messages)
            response = chained_model.invoke({"question": question, "memory_records": memory_messages})

            parsed = eval(response.content)
            for key in keys_to_check_for:
                if key not in parsed:
                    raise ValueError("Missing required keys during generate. Retrying...")
            
            output = {
                "found_in_memory": parsed["found_in_memory"],
                "instance_desc": parsed["instance_desc"],
            }
            target_record = None
            if not parsed["found_in_memory"]:
                instance_viz_path = None
            else:
                for record in self.memory_records:
                    if int(record["id"]) == int(parsed["record_id"]):
                        target_record = record
                        break
                if target_record is None:
                    raise ValueError(f"Could not find record with ID {parsed['record_id']} in memory records. Retrying...")
                image_path_fn = lambda vidpath, frame: os.path.join(vidpath, f"{frame:06d}.png")
                vidpath = record["vidpath"]
                start_frame = record["start_frame"]
                end_frame = record["end_frame"]
                start_frame, end_frame = int(start_frame), int(end_frame)
                frame = (start_frame + end_frame) // 2
                instance_viz_path = image_path_fn(vidpath, frame)
            output["instance_viz_path"] = instance_viz_path
            output["past_observations"] = [target_record] if target_record else []
                
            if self.logger:
                self.logger.info(f"[DETERMINE_SEARCH_INSTANCE] generate() - Parsed output: {parsed}")
                self.logger.info(f"[DETERMINE_SEARCH_INSTANCE] generate() - Output keys: found_in_memory={output['found_in_memory']}, instance_desc={output['instance_desc']}, instance_viz_path={output['instance_viz_path']}")
                
            return {"messages": [response], "output": parsed}
        
        def build_graph(self):
            workflow = StateGraph(AgentState)
            
            workflow.add_node("decide", lambda state: try_except_continue(state, self.decide))
            workflow.add_node("action", ToolNode(self.tools))
            workflow.add_node("generate", lambda state: try_except_continue(state, self.generate))
            
            workflow.add_conditional_edges(
                "decide",
                from_decide_to,
                {
                    "action": "action",
                    "end": "generate",
                },
            )
            workflow.add_edge('action', 'decide')
            workflow.add_edge("generate", END)
            workflow.set_entry_point("decide")
            self.graph = workflow.compile()
            
            
        def run(self, user_task: str, history_summary: str, current_task: str, memory_records: Optional[List[Dict]] = None) -> SearchInstance:
            if self.logger:
                self.logger.info(
                    f"[DETERMINE_SEARCH_INSTANCE] Running tool with user_task: {user_task}, "
                    f"current_task: {current_task}, memory_records: {memory_records}"
                )
            
            self.decide_call_count = 0
            self.previous_tool_requests = "I have already used the following retrieval tools and the results are included below. Do not repeat them:\n"
            
            self.user_task = user_task
            self.history_summary = history_summary
            self.current_task = current_task
            self.memory_records = memory_records
            
            self.build_graph()
            
            question = f"In your context, you have the following information, and you will need to address some user request -" \
                       f"User Task: {user_task}\n" \
                       f"History Summary: {history_summary}\n" \
                       f"Current Task: {current_task}\n" \
                       f"Memory Records: {memory_records if memory_records else 'None'}\n"
            inputs = { "messages": [
                    (("user", question))
                ]
            }
            state = self.graph.invoke(inputs)
            
            output = state.get("output", [])
            return output
            
    class DetermineSearchInstanceInput(BaseModel):
        user_task: str = Field(
            description="The high-level task the user wants to perform. For example, 'bring me the book I was reading yesterday'."
        )
        history_summary: str = Field(
            description="A summary of the conversation history leading up to this point. This can help the agent understand the context better."
        )
        current_task: str = Field(
            description="The current goal or subtask the system is working on. For example, 'resolve which book the user is referring to', 'identify where (or how) to retrieve ths instance."
        )
        memory_records: Optional[List[Dict]] = Field(
            default=None,
            description="(Optional) A list of memory records retrieved earlier. Each record includes a caption, time, position, and possibly an image path."
        )
        
    memory_instance_tool_runner = DetermineSearchInstanceAgent(memory, llm, llm_raw, vlm, vlm_raw, logger)
    memory_instance_tool = StructuredTool.from_function(
        func=lambda user_task, history_summary, current_task, memory_records: memory_instance_tool_runner.run(user_task, history_summary, current_task, memory_records),
        name="create_or_update_target_search_instance",
        description="Create or Update the memory search instance based on the high-level user task, current reasoning step, and optionally retrieved memory records. This would be the next memory search target for the agent.",
        args_schema=DetermineSearchInstanceInput
    )
    
    real_world_instance_tool_runner = DetermineSearchInstanceAgent(memory, llm, llm_raw, vlm, vlm_raw, logger)
    real_world_instance_tool = StructuredTool.from_function(
        func=lambda user_task, history_summary, current_task, memory_records: real_world_instance_tool_runner.run(user_task, history_summary, current_task, memory_records),
        name="create_or_update_target_search_instance",
        description="Create or Update the real world search instance based on the high-level user task, current reasoning step, and optionally retrieved memory records.",
        args_schema=DetermineSearchInstanceInput
    )
    
    return [memory_instance_tool, real_world_instance_tool]