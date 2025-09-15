import sys
import os
from typing import List, Dict, Optional
from typing import Annotated, Sequence, TypedDict
from pydantic import conint

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from memory.memory import MilvusMemory
from agent.utils.utils import *
from agent.utils.skills import *

def create_physical_termination_skill() -> List[StructuredTool]:
    class TerminateInput(BaseModel):
        summary: str = Field(
            description="A short explanation of what has been done and why the task is considered complete"
        )
    
    def _terminate_fn(summary: str) -> bool:
        # Dummy implementation: always return True
        return True

    terminate_tool = StructuredTool.from_function(
        func=_terminate_fn,
        name="terminate_task",
        description=(
            "Finalize the search task once it is complete. "
            "You should call this tool when you have successfully completed the task and no further actions are needed (i.e., you saw the object required by task).\n\n"
            "- Constraint: Must be called ALONE (no other tools in the same step)."
        ),
        args_schema=TerminateInput,
    )

    return [terminate_tool]

def create_tiago_physical_skills(store: TempJsonStore) -> List[StructuredTool]:
    class NavigateInput(BaseModel):
        tool_rationale: str = Field(
            description=TOOL_RATIONALE_DESC
        )
        pos: List[float] = Field(
            description="Target position in 3D space as [x, y, z]. (Remember: you should select from the provided landmark positions.)"
        )
        theta: float = Field(
            description="Orientation angle in radians. (Remember: you should select from the provided landmark orientations.)"
        )
        
    def _navigate(
        store: TempJsonStore,
        tool_rationale: str,
        pos: List[float],
        theta: float
    ) -> dict:
        response = navigate(pos, theta)
        images = []
        for img_msg in response.pano_images:
            images += ros_image_to_vlm_message(img_msg)
        payload = {
            "type": "navigate",
            "success": response.success,
            "images": images,
        }
        file_id = store.save(payload)
        return {
            "success": response.success,
            "file_id": file_id,
        }
        
    navigate_tool = StructuredTool.from_function(
        func=lambda tool_rationale, pos, theta: _navigate(store, tool_rationale, pos, theta),
        name="robot_navigate",
        description=(
            "Navigate the robot to a specific position and orientation in the environment. "
            "This tool allows the agent to move to a desired location and capture images of the surroundings.\n\n"
        ),
        args_schema=NavigateInput,  
    )
    return [navigate_tool]

def create_physical_skills(store: TempJsonStore) -> List[StructuredTool]:
    class NavigateInput(BaseModel):
        tool_rationale: str = Field(
            description=TOOL_RATIONALE_DESC
        )
        pos: List[float] = Field(
            description="Target position in 3D space as [x, y, z]."
        )
        theta: float = Field(
            description="Orientation angle in radians. If theta is not known, it can be set to 0. "
        )
        
    def _navigate(
        store: TempJsonStore,
        tool_rationale: str,
        pos: List[float],
        theta: float
    ) -> dict:
        response = navigate(pos, theta)
        images = []
        for img_msg in response.pano_images:
            images += ros_image_to_vlm_message(img_msg)
        payload = {
            "type": "navigate",
            "success": response.success,
            "images": images,
        }
        file_id = store.save(payload)
        return {
            "success": response.success,
            "file_id": file_id,
        }
        
    navigate_tool = StructuredTool.from_function(
        func=lambda tool_rationale, pos, theta: _navigate(store, tool_rationale, pos, theta),
        name="robot_navigate",
        description=(
            "Navigate the robot to a specific position and orientation in the environment. "
            "This tool allows the agent to move to a desired location and capture images of the surroundings.\n\n"
        ),
        args_schema=NavigateInput,  
    )
    
    class DetectInput(BaseModel):
        tool_rationale: str = Field(
            description=TOOL_RATIONALE_DESC
        )
        query_text: str = Field(
            description="The class label of the object to detect. It must be one of 'book', 'magazine', 'toy', 'folder', 'cabinet', 'bananas', 'apple', 'cupcake', 'cereal', 'mincedmeat', 'creamybuns'"
        )
        
    def _detect(store: TempJsonStore, tool_rationale: str, query_text: str) -> dict:
        response = detect_virtual_home_object(query_text)
        if not response.success:
            return {"success": False, "message": "Detection failed."}
        
        images = []
        for img_msg in response.images:
            images += ros_image_to_vlm_message(img_msg)
        
        payload = {
            "type": "detect",
            "success": response.success,
            "images": images,
            "instance_ids": response.ids,
            "visible_instances": response.visible_instances,
        }
        file_id = store.save(payload)
        
        return {
            "success": response.success,
            "file_id": file_id,
            "instance_ids": response.ids,
            "visible_instances": response.visible_instances,
        }
    
    detect_tool = StructuredTool.from_function(
        func=lambda tool_rationale, query_text: _detect(store, tool_rationale, query_text),
        name="robot_detect",
        description=(
            "Detect all objects in the environment based on the given class label. "
            "It will return all detected bounding boxes as well as the instance IDs (required by other skills such as robot_pick and robot_open)"
        ),
        args_schema=DetectInput,
    )
    
    class PickInput(BaseModel):
        tool_rationale: str = Field(
            description=TOOL_RATIONALE_DESC
        )
        instance_id: int = Field(
            description="The instance ID of the object to pick. If you don't know the instance ID, you can use robot_detect to get all detected objects and their IDs."
        )
    
    def _pick(store: TempJsonStore, tool_rationale: str, instance_id: int) -> dict:
        response = pick_by_instance_id(instance_id)
        return {
            "success": response.success,
            "instance_uid": response.instance_uid,
        }
    
    pick_tool = StructuredTool.from_function(
        func=lambda tool_rationale, instance_id: _pick(store, tool_rationale, instance_id),
        name="robot_pick",
        description=(
            "Pick an object by its instance ID. "
            "You can use robot_detect to get the instance ID of the object you want to pick."
        ),
        args_schema=PickInput,
    )
    
    class OpenInput(BaseModel):
        tool_rationale: str = Field(
            description=TOOL_RATIONALE_DESC
        )
        instance_id: int = Field(
            description="The instance ID of the object to open. If you don't know the instance ID, you can use robot_detect to get all detected objects and their IDs."
        )
    
    def _open(store: TempJsonStore, tool_rationale: str, instance_id: int) -> dict:
        response = open_by_instance_id(instance_id)
        return {
            "success": response.success,
            "instance_uid": response.instance_uid,
        }
    
    open_tool = StructuredTool.from_function(
        func=lambda tool_rationale, instance_id: _open(store, tool_rationale, instance_id),
        name="robot_open",
        description=(
            "Open an object by its instance ID. "
            "You can use robot_detect to get the instance ID of the object you want to open."
        ),
        args_schema=OpenInput,
    )
    
    return [navigate_tool, detect_tool, pick_tool, open_tool]

TOOL_RATIONALE_DESC = (
    "Explain briefly why this tool is being called. The rationale should clarify how this tool helps move the reasoning forward — "
    "what is already known, what new insight is expected from the call, what uncertainty or open question this tool is meant to resolve, and why, comapring to other tools, this is the most optimal next step to do. "
    "Avoid vague or redundant justifications; focus on the unique purpose of this tool in the current context."
)

def create_pause_and_think_tool() -> List[StructuredTool]:

    class PauseAndThinkInput(BaseModel):
        recent_activity: str = Field(
            description="What the agent has been doing recently (e.g., which tools were used, what the goal was, what queries were attempted)"
        )
        current_findings: str = Field(
            description="What the agent currently knows or believes to be true, based on previous tool results or reasoning"
        )
        open_questions: str = Field(
            description="What is still unclear, ambiguous, or unresolved — areas where more information or disambiguation is needed"
        )
        next_step_plan: str = Field(
            description="What the agent intends to do next and why — a plan to resolve uncertainties, confirm identity, or proceed toward task completion"
        )

    def _pause_and_think_fn(
        recent_activity: str,
        current_findings: str,
        open_questions: str,
        next_step_plan: str,
    ) -> bool:
        # Dummy implementation: always return True
        return f"This is the next step you propose: {next_step_plan}"

    pause_tool = StructuredTool.from_function(
        func=_pause_and_think_fn,
        name="pause_and_think",
        description=(
            "Use this to pause and reflect on your reasoning so far. This tool helps you summarize what you’ve done, what you’ve learned, what’s still uncertain, and what you plan to do next.\n\n"
            "- You should call this tool **frequently**, especially when:\n"
            "  • You’ve made several tool calls and want to consolidate your progress\n"
            "  • You’re about to change strategies or time ranges\n"
            "  • You’ve found some relevant records but haven’t reached a confident conclusion yet\n"
            "- You can call this tool even when things are going well — it helps you stay organized and deliberate.\n"
            "- Must include four fields: `recent_activity`, `current_findings`, `open_questions`, and `next_step_plan`.\n"
            "- Must be called **alone** in a single iteration.\n"
        ),
        args_schema=PauseAndThinkInput,
    )

    return [pause_tool]

def create_memory_terminate_tool(memory: MilvusMemory) -> List[StructuredTool]:

    class MemoryTerminateInput(BaseModel):
        summary: str = Field(
            description="A short explanation of what is being retrieved and why"
        )
        instance_description: str = Field(
            description="A physical description of the object instance to retrieve, focusing on its appearance (e.g., 'a purple cup', 'a transparent glass with golden handle')"
        )
        position: List[float] = Field(
            description="3D target coordinate for the object retrieval task, as [x, y, z]"
        )
        theta: float = Field(
            description="Orientation angle in radians"
        )
        record_ids: List[int] = Field(
            description=(
                "A set of memory record IDs that are directly useful for downstream planning and execution. "
                "Include the most recent verified record that shows the object at `position`/`theta`, "
                "plus any additional records that provide supporting evidence for re-identification, "
                "landmark context, or visibility transitions. "
                "The robot should be able to rely on this evidence pack alone to localize and interactively "
                "retrieve the target instance."
            )
        )

    def _terminate_fn(
        summary: str,
        instance_description: str,
        position: List[float],
        theta: float,
        record_ids: List[int],
    ) -> bool:
        # Dummy implementation: always return True
        records = []
        for record_id in record_ids:
            record = eval(memory.get_by_id(record_id))
            if record:
                records += record
        return str(records)

    terminate_tool = StructuredTool.from_function(
        func=_terminate_fn,
        name="terminate",
        description=(
            "Finalize the MEMORY-ONLY retrieval once the object's identity and LAST-KNOWN STATE are grounded in memory.\n\n"
            "- Required fields:\n"
            "  - `summary`: What to retrieve and why these records support it.\n"
            "  - `instance_description`: Visual cues for re-identification of the exact instance.\n"
            "  - `position`: 3D coordinate [x, y, z] from the most recent verified record of the object.\n"
            "  - `theta`: Orientation (radians) from the same record as `position`.\n"
            "  - `record_ids`: Set of memory record IDs that provide actionable evidence for downstream robot planning. "
            "Include the most recent verified record plus any additional supporting records needed for "
            "identity confirmation, container disambiguation, or planning approach. "
            "The robot should be able to rely on these records to continue interactive retrieval.\n\n"
            "- Constraint: Must be called ALONE (no other tools in the same step)."
        ),
        args_schema=MemoryTerminateInput,
    )

    return [terminate_tool]

def create_memory_search_tools(memory: MilvusMemory):

    class TextRetrieverInputWithTime(BaseModel):
        tool_rationale: str = Field(
            description=TOOL_RATIONALE_DESC
        )
        x: str = Field(
            description=(
                "A natural language description of the scene or object to search for in memory. "
                "This description will be embedded and used for vector similarity search against past memory captions. "
                "**Do not use this field to search for time or location directly — use dedicated time and position tools for that.** "
                "Examples of valid input include 'a person sitting at a table' or 'a green car near the garage'. "
                "**Note:** This performs approximate semantic search using text embeddings and always returns the top-k most similar results, "
                "even if none of them are exact or relevant matches."
            )
        )
        start_time: Optional[str] = Field(
            default=None,
            description="Start search time in YYYY-MM-DD HH:MM:SS format. If provided, only search for observations after this time."
        )
        end_time: Optional[str] = Field(
            default=None,
            description="End search time in YYYY-MM-DD HH:MM:SS format. If provided, only search for observations before this time."
        )
        k: conint(ge=int(1), le=int(50)) = Field(
            default=8,
            description="The number of top similar memory results to return. These are ranked by vector similarity between your query and memory captions."
        )

    class PositionRetrieverInputWithTime(BaseModel):
        tool_rationale: str = Field(
            description=TOOL_RATIONALE_DESC
        )
        position: List[float] = Field(
            description="The 3D position [x, y, z] to search around in memory. Each value should be a float."
        )
        start_time: Optional[str] = Field(
            default=None,
            description="Start search time in YYYY-MM-DD HH:MM:SS format. If provided, only search for observations after this time."
        )
        end_time: Optional[str] = Field(
            default=None,
            description="End search time in YYYY-MM-DD HH:MM:SS format. If provided, only search for observations before this time."
        )
        k: conint(ge=int(1), le=int(50)) = Field(
            default=8,
            description="The number of top similar memory results to return. These are ranked by vector similarity between your query and memory captions."
        )

    class TimeRetrieverInput(BaseModel):
        tool_rationale: str = Field(
            description=TOOL_RATIONALE_DESC
        )
        time: str = Field(
            description="The specific timestamp to search near, in YYYY-MM-DD HH:MM:SS format. Returns the most relevant observations near that moment."
        )
        k: conint(ge=int(1), le=int(50)) = Field(
            default=8,
            description="The number of top similar memory results to return. These are ranked by vector similarity between your query and memory captions."
        )
        
    class TimeRangeCountInput(BaseModel):
        tool_rationale: str = Field(
            description=TOOL_RATIONALE_DESC
        )
        start_time: Optional[str] = Field(
            default=None,
            description="Start time of the interval to count records from, in YYYY-MM-DD HH:MM:SS format. Optional."
        )
        end_time: Optional[str] = Field(
            default=None,
            description="End time of the interval to count records to, in YYYY-MM-DD HH:MM:SS format. Optional."
        )

    txt_time_tool = StructuredTool.from_function(
        func=lambda tool_rationale, x, start_time=None, end_time=None, k=8: memory.search_by_txt_and_time(x, start_time, end_time, k),
        name="search_in_memory_by_text_within_time_range",
        description="Search memory by text caption with optional time constraints. Retrieves memories based on text relevance.",
        args_schema=TextRetrieverInputWithTime
    )

    pos_time_tool = StructuredTool.from_function(
        func=lambda tool_rationale, position, start_time=None, end_time=None, k=8: memory.search_by_position_and_time(position, start_time, end_time, k),
        name="search_in_memory_by_position_within_time_range",
        description="Search memory based on 3D position with optional time constraints. Retrieves memories spatially near the given position.",
        args_schema=PositionRetrieverInputWithTime
    )

    time_tool = StructuredTool.from_function(
        func=lambda tool_rationale, time, k=8: memory.search_by_time(time, k),
        name="search_in_memory_by_time",
        description="Search memory for observations that occurred close to a specific timestamp.",
        args_schema=TimeRetrieverInput
    )
    
    count_tool = StructuredTool.from_function(
        func=lambda tool_rationale, start_time=None, end_time=None: memory.count_records_by_time(start_time, end_time),
        name="get_record_count_within_time_range",
        description="Return the number of memory records within a given time range. If no time bounds are provided, count all records.",
        args_schema=TimeRangeCountInput
    )

    return [txt_time_tool, pos_time_tool, time_tool, count_tool]

def create_memory_inspection_tool(memory: MilvusMemory) -> StructuredTool:

    class MemoryInspectionInput(BaseModel):
        tool_rationale: str = Field(
            description=TOOL_RATIONALE_DESC
        )
        record_id: List[int] = Field(
            description=(
                "A list of memory record IDs to inspect. "
                "Each ID should be an integer previously returned by a search tool. "
                "Use this when you need to visually verify one or more candidate memories "
                "before deciding what to retrieve."
            )
        )

    def _inspect_memory_record(tool_rationale: str, record_id: List[int]) -> Dict[int, str]:
        """
        For every ID in `record_id`, load the corresponding memory entry and
        return {id: image_path}.  The caller can then decide which visual
        evidence is most relevant.  (Image is not yet base64‑encoded for
        bandwidth efficiency; encode it client‑side if needed.)
        """
        id_to_path = {}
        image_path_fn = lambda vidpath, frame: os.path.join(vidpath, f"{frame:06d}.png")

        for rid in record_id:
            docs = memory.get_by_id(rid)
            if docs is None or len(docs) == 0:
                continue  # silently skip missing records
            record = eval(docs)[0]
            vidpath = record["vidpath"]
            start_frame = int(record["start_frame"])
            end_frame = int(record["end_frame"])
            frame = (start_frame + end_frame) // 2
            id_to_path[rid] = image_path_fn(vidpath, frame)

        return id_to_path

    inspection_tool = StructuredTool.from_function(
        func=_inspect_memory_record,
        name="inspect_observations_in_memory",
        description=(
            "Inspect one or more memory records by ID and obtain a quick visual "
            "snapshot (middle frame) for each.  Use this to validate hypotheses "
            "before finalizing a retrieval."
        ),
        args_schema=MemoryInspectionInput
    )

    return [inspection_tool]


def create_recall_best_matches_terminate_tool(memory: MilvusMemory) -> StructuredTool:
    
    class BestMatchTerminateInput(BaseModel):
        summary: str = Field(
            description="A short explanation of what is being retrieved and why"
        )
        record_ids: List[int] = Field(
            description="IDs of the best-matching records. In any case, you are encouraged to at least output 1 answer. If you believe the query object does not appear in your past memory at all, you are allowed to make your best guesses by common sense."
        )
    
    def _terminate_fn(
        summary: str,
        record_ids: List[int]
    ) -> str:
        records = []
        for record_id in record_ids:
            record = eval(memory.get_by_id(record_id))
            if record:
                records += record
        return str(records)

    terminate_tool = StructuredTool.from_function(
        func=_terminate_fn,
        name="recall_best_matches_terminate",
        description=(
            "Use this to finalize the task once you are confident about what to retrieve. "
            "You should call this tool when you have identified the best matching records based on your search and reasoning.\n\n"
        ),
        args_schema=BestMatchTerminateInput
    )
    
    return [terminate_tool]

class ToolAgent:
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        agent_history: Annotated[Sequence[BaseMessage], add_messages]
        
    @staticmethod
    def from_agent_to(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1] if messages else None
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for call in last_message.tool_calls:
                if "terminate" in call.get("name"):
                    return "next"
                elif "pause_and_think" in call.get("name"):
                    return "reflection"
        return "action"
    
    def __init__(self, memory: MilvusMemory, vlm_flex, vlm, logger=None):
        self.memory = memory
        self.vlm_flex = vlm_flex
        self.vlm = vlm
        self.logger = logger
        
        self.setup_tools(memory)
        
        self.agent_call_count = 0
        
    def setup_tools(self, memory: MilvusMemory):
        search_tools = create_memory_search_tools(memory)
        inspect_tools = create_memory_inspection_tool(memory)
        response_tools = create_recall_best_matches_terminate_tool(memory)
        reflect_tools = create_pause_and_think_tool()

        self.all_tools = search_tools + inspect_tools + response_tools + reflect_tools
        self.all_tool_definitions = [convert_to_openai_function(t) for t in self.all_tools]

        self.tools = search_tools + inspect_tools + response_tools
        self.tool_definitions = [convert_to_openai_function(t) for t in self.tools]
        
        self.reflect_tools = reflect_tools
        self.reflect_tool_definitions = [convert_to_openai_function(t) for t in self.reflect_tools]
        self.response_tools = response_tools
        self.response_tool_definitions = [convert_to_openai_function(t) for t in self.response_tools]
        
    def agent(self, state: AgentState):
        raise NotImplementedError("This is a base class. Use a subclass that implements the agent logic.")
    
    def build_graph(self): # TODO need to fix this
        workflow = StateGraph(ToolAgent.AgentState)
        
        workflow.add_node("agent", lambda state: try_except_continue(state, self.agent))
        workflow.add_node("reflection_action", ToolNode(self.reflect_tools))
        workflow.add_node("action", ToolNode(self.all_tools))

        workflow.add_edge("action", "agent")
        workflow.add_conditional_edges(
            "agent",
            ToolAgent.from_agent_to,
            {
                "next": END,
                "action": "action",
            },
        )
        workflow.add_edge("reflection_action", "agent")

        workflow.set_entry_point("agent")
        self.graph = workflow.compile()
            
    
def create_recall_best_matches_tool(memory: MilvusMemory, vlm_flex, vlm, logger=None):
    class BestMatchAgent:
        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]
            agent_history: Annotated[Sequence[BaseMessage], add_messages]
            
        @staticmethod
        def from_agent_to(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1] if messages else None
            
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for call in last_message.tool_calls:
                    if "terminate" in call.get("name"):
                        return "next"
            return "action"
    
        def __init__(self, memory, vlm_flex, vlm, logger=None):
            self.memory = memory
            self.vlm_flex = vlm_flex
            self.vlm = vlm
            self.logger = logger
            
            self.setup_tools(memory)
            
            prompt_dir = str(os.path.dirname(__file__)) + '/../prompts/best_matches_tool/'
            self.agent_prompt = file_to_string(prompt_dir+'agent_prompt.txt')
            self.agent_reflection_prompt = file_to_string(prompt_dir+'agent_reflection_prompt.txt')
            self.agent_gen_only_prompt = file_to_string(prompt_dir+'agent_gen_only_prompt.txt')
            
            self.agent_call_count = 0
            
        def setup_tools(self, memory: MilvusMemory):
            search_tools = create_memory_search_tools(memory)
            inspect_tools = create_memory_inspection_tool(memory)
            response_tools = create_recall_best_matches_terminate_tool(memory)
            reflect_tools = create_pause_and_think_tool()

            self.all_tools = search_tools + inspect_tools + response_tools + reflect_tools
            self.all_tool_definitions = [convert_to_openai_function(t) for t in self.all_tools]

            self.tools = search_tools + inspect_tools + response_tools
            self.tool_definitions = [convert_to_openai_function(t) for t in self.tools]
            
            self.reflect_tools = reflect_tools
            self.reflect_tool_definitions = [convert_to_openai_function(t) for t in self.reflect_tools]
            self.response_tools = response_tools
            self.response_tool_definitions = [convert_to_openai_function(t) for t in self.response_tools]
            
        def agent(self, state: AgentState):
            max_agent_call_count = 10 
            n_reflection_intervals = 5
            
            messages = state["messages"]
            
            additional_search_history = []
            last_message = messages[-1]
            if isinstance(last_message, ToolMessage):
                # ===  Step 1: Find last AIMessage with tool_calls
                idx = len(messages) - 1
                last_ai_idx = None
                while idx >= 0:
                    msg = messages[idx]
                    if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                        last_ai_idx = idx
                        break
                    idx -= 1

                # ===  Step 2: Append all following ToolMessages into search_in_time_history
                if last_ai_idx is not None:
                    image_messages = []
                    for msg in messages[last_ai_idx+1:]:
                        if isinstance(msg, ToolMessage):
                            original_msg_content = copy.copy(msg.content)
                            if isinstance(msg.content, str):
                                msg.content = parse_and_pretty_print_tool_message(msg.content)
                            additional_search_history.append(msg)
                            
                            if isinstance(original_msg_content, str) and is_image_inspection_result(original_msg_content):
                                inspection = eval(original_msg_content)
                                for id, path in inspection.items():
                                    content = get_image_message_for_record(id, path, msg.tool_call_id)
                                    message = HumanMessage(content=content)
                                    image_messages.append(message)
                            if self.logger:
                                self.logger.info(f"[BEST MATCHES] Tool Response: {msg.content}")
                                
                    additional_search_history += image_messages
            
            chat_history = copy.deepcopy(state.get("agent_history", []))
            chat_history += additional_search_history
                    
            model = self.vlm
            model_flex = self.vlm_flex
            if self.agent_call_count < max_agent_call_count:
                if self.agent_call_count % n_reflection_intervals == 0:
                    prompt = self.agent_reflection_prompt
                    current_tool_defs = self.reflect_tool_definitions
                else:
                    prompt = self.agent_prompt
                    current_tool_defs = self.tool_definitions
            else:
                prompt = self.agent_gen_only_prompt
                current_tool_defs = self.response_tool_definitions
                
            model = model.bind_tools(current_tool_defs)
            model_flex = model_flex.bind_tools(current_tool_defs)
            tool_names = [tool['name'] for tool in current_tool_defs]
            tool_list_str = "\n".join([f"{i+1}. {name}" for i, name in enumerate(tool_names)])
                
            question_str = (
                f"Your peer agent wants to retrieve at most {self.k} memory records that best match the following description:\n"
                f"→ {self.description.strip()}\n"
            )
            if self.image_message:
                question_str += f"Your peer agent also provides an image where the target object was last observed:\n"
            
            # "Based on what you’ve observed and what you already know, what should you do next?"
            question_content = [{"type": "text", "text": question_str}]
            if self.image_message:
                question_content += self.image_message
            
            chat_template = [
                ("human", "You are a memory retrieval agent. Your job is to retrieve memory records that best match the object described. Based on your tool calls and the results you've received so far, continue reasoning and searching."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("system", prompt),
                ("system", "{fact_prompt}"),
                HumanMessage(content=question_content),
                ("human", "{caller_context_text}"),
                ("system", "Now decide your next action. Use tools to continue searching or terminate if you are confident. Reason carefully based on what you've done, what you know, and what the user ultimately needs. Pay attention to the time constraints if they are provided.")
            ]
            if self.agent_call_count < max_agent_call_count:
                chat_template += [
                    ("system", f"You must strictly follow the JSON output format. As a reminder, these are available tools: \n{tool_list_str}. You must use one of the tools to continue searching or finalize your decision without any additional explanation."),
                    ("system", f"🔄 You are allowed up to **{max_agent_call_count} iterations** total. This is iteration **#{self.agent_call_count}**.\nEach iteration consists of one full round of tool calls — even if you issue multiple tools in parallel, that still counts as one iteration.")
                ]
            else:
                chat_template += [
                    ("system", f"You must strictly follow the JSON output format. Since you have already reached the maximum number of iterations, you should finalize your decision now by calling: {tool_list_str}"),
                ]
            
            chat_prompt = ChatPromptTemplate.from_messages(chat_template)
            
            chained_model = chat_prompt | model
            chained_flex_model = chat_prompt | model_flex

            fact_prompt = f"Here are some facts for your context:\n" \
                      f"1. {self.memory.get_memory_stats_for_llm()}\n" \
                      f"2. You have been patrolling in a dynamic household or office environment, so objects you saw before may have been moved, or its status may be changed.\n"\
                      f"3. {self.time_str}\n"

            caller_context_text = self.caller_context if self.caller_context else "None"
            caller_context_text = f"Additional context from the caller agent:\n→ {caller_context_text}"
            
            input = {
                "chat_history": chat_history,
                "fact_prompt": fact_prompt,
                "caller_context_text": caller_context_text
            }
            response = safe_gpt_invoke(chained_flex_model, chained_model, input)
            
            if self.logger:
                if hasattr(response, "tool_calls") and response.tool_calls:
                    for call in response.tool_calls:
                        args_str = ", ".join(f"{k}={repr(v)}" for k, v in call.get("args", {}).items())
                        log_str = f"{call.get('name')}({args_str})"
                        self.logger.info(f"[BEST MATCHES] Tool call: {log_str}")
                else:
                    self.logger.info(f"[BEST MATCHES] {response}")
                    
            self.agent_call_count += 1
            return {"messages": [response], "agent_history": additional_search_history + [response]}
        
        def build_graph(self):
            workflow = StateGraph(BestMatchAgent.AgentState)
            
            workflow.add_node("agent", lambda state: try_except_continue(state, self.agent))
            workflow.add_node("action", ToolNode(self.all_tools))

            workflow.add_edge("action", "agent")
            workflow.add_conditional_edges(
                "agent",
                BestMatchAgent.from_agent_to,
                {
                    "next": END,
                    "action": "action",
                },
            )
            
            workflow.set_entry_point("agent")
            self.graph = workflow.compile()
            
        def run(self, 
                tool_rationale: str,
                description: str, 
                visual_cue_from_record_id: Optional[int] = None,
                search_start_time: Optional[str] = None, 
                search_end_time: Optional[str] = None,
                k: Optional[int] = 5,
                caller_context: Optional[str] = None,
            ) -> List[Dict]:
            
            if self.logger:
                self.logger.info(
                    f"[BEST_MATCH] Running tool with description: {description}, "
                    f"visual_cue_from_record_id: {visual_cue_from_record_id}, "
                    f"search_start_time: {search_start_time}, "
                    f"search_end_time: {search_end_time}"
                )
                
            self.description = description.strip()
            self.visual_cue_from_record_id = visual_cue_from_record_id
            
            self.visual_cue_from_record_id = visual_cue_from_record_id
            if self.visual_cue_from_record_id is not None:
                self.viz_path = get_viz_path(self.memory, self.visual_cue_from_record_id)
                self.image_message = get_image_message_for_record(
                    self.visual_cue_from_record_id, 
                    self.viz_path, 
                )
            else:
                self.viz_path = None
                self.image_message = None
            
            self.search_start_time = search_start_time
            self.search_end_time = search_end_time
            self.k = k
            self.caller_context = caller_context
            
            self.setup_tools(self.memory)
            self.build_graph()
            
            self.agent_call_count = 0
            
            content = []
            content += [{"type": "text", "text": f"Task: Recall at most k memory records best match objects:\n→ {description.strip()}"}]
            if self.visual_cue_from_record_id is not None:
                content += [{"type": "text", "text": f"From previous search, you determined that this instance has appeared in record {visual_cue_from_record_id}. This is the image from your previous observation. Please find most-relevant records related to this instance:"}]
                content += get_image_message_for_record(
                    self.visual_cue_from_record_id, 
                    self.viz_path, 
                )
                
            time_str = "Your peer agent is only interested in information within the following time window:"
            
            start_t_str, end_t_str = self.memory.get_db_time_range()
            if search_start_time is not None and search_end_time is None:
                self.search_end_time = end_t_str
            elif search_start_time is None and search_end_time is not None:
                self.search_start_time = start_t_str
            
            if search_start_time is None and search_end_time is None:
                time_str += "\n→ None"
            else:
                time_str += f"\n→ Start: {search_start_time}"
                time_str += f"\n→ End: {search_end_time}"
                
            self.time_str = time_str
            
            content += [{"type": "text", "text": self.time_str}]
            
            inputs = {
                "messages": [
                    HumanMessage(content=content),
                ]
            }
            state = self.graph.invoke(inputs)
            
            output = {
                "tool_name": "recall_best_matches",
                "summary": "",
                "records": []
            }
            
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls"):
                tool_call = last_message.tool_calls[0] if last_message.tool_calls else None
                if tool_call and (type(tool_call) is dict) and ("args" in tool_call.keys()) and type(tool_call["args"]) is dict:
                    output["summary"] = tool_call["args"].get("summary", "")

                    record_ids = tool_call["args"].get("record_ids", [])
                    records = []
                    for record_id in record_ids:
                        record = eval(memory.get_by_id(record_id))
                        if record:
                            records += record
                    output["records"] = records
                    
            return output
        
    tool_runner = BestMatchAgent(memory, vlm_flex, vlm, logger)
            
    class BestMatchesInput(BaseModel):
        tool_rationale: str = Field(
            description=TOOL_RATIONALE_DESC
        )
        description: str = Field(
            description=(
                "Text query.\n"
                "• Text-only: this DEFINES the target to retrieve (e.g., 'a book on the table', 'the algebra book').\n"
                "• With visual_cue_from_record_id: this acts as a VERBAL POINTER to the object IN that image "
                "(e.g., 'the red mug on the left')."
            )
        )
        visual_cue_from_record_id: Optional[int] = Field(
            default=None,
            description=(
                "ID of a memory record whose image serves as the visual ANCHOR. "
                "When provided, the tool queries for the SAME INSTANCE seen in that image. "
                "Use 'description' to indicate which object in the image you mean (a stand-in for circling it)."
            )
        )
        search_start_time: Optional[str] = Field(
            default=None,
            description="Start time (inclusive) for searching memory. Can be an ISO string or datetime. Leave blank for no lower bound."
        )
        search_end_time: Optional[str] = Field(
            default=None,
            description="End time (inclusive) for searching memory. Can be an ISO string or datetime. Leave blank for no upper bound."
        )
        k: conint(ge=int(1), le=int(10)) = Field(
            default=5,
            description="Maximum number of records to return that best match the query. Must be between 1 and 10."
        )
        caller_context: Optional[str] = Field(
            default=None,
            description="Additional free-text context from the caller agent to help guide the retrieval — e.g., what the caller is trying to do, what it is uncertain about, or how the results will be used."
        )
            
    best_match_tool = StructuredTool.from_function(
        func=tool_runner.run,
        name="recall_best_matches",
        description=(
           "Recall up to k memory records that best match the request.\n"
            "• Text-only: 'description' DEFINES the target (semantic search).\n"
            "• With visual_cue_from_record_id: the image ANCHORS the instance; 'description' VERBALLY POINTS to the object in that image. "
            "Return the most relevant/recallable sightings accordingly. Use time filters for recency."
        ),
        args_schema=BestMatchesInput,
    )
    
    return [best_match_tool]

def create_recall_last_seen_terminate_tool(memory: MilvusMemory) -> StructuredTool:
    class LastSeenTerminateInput(BaseModel):
        summary: str = Field(
            description="A short explanation of what object was retrieved and why this memory record is considered the most recent valid sighting."
        )
        record_id: int = Field(
            description="The ID of the last seen memory record that matches the query."
        )

    def _terminate_fn(summary: str, record_id: int) -> str:
        record = memory.get_by_id(record_id)
        if record:
            return str(record)
        return "No record found with the given ID."

    terminate_tool = StructuredTool.from_function(
        func=_terminate_fn,
        name="recall_last_seen_terminate",
        description=(
            "Call this tool when you have visually confirmed the most recent memory record where the target object was last seen. "
            "If no such record exists, return an empty result."
        ),
        args_schema=LastSeenTerminateInput
    )
    
    return [terminate_tool]

def create_recall_last_seen_tool(memory: MilvusMemory, vlm_flex, vlm, logger=None) -> StructuredTool:

    class LastSeenAgent:
        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]
            agent_history: Annotated[Sequence[BaseMessage], add_messages]
    
        @staticmethod
        def from_agent_to(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1] if messages else None
            
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for call in last_message.tool_calls:
                    if "terminate" in call.get("name"):
                        return "next"
            return "action"
        
        def __init__(self, memory, vlm_flex, vlm, logger=None):
            self.memory = memory
            self.vlm_flex = vlm_flex
            self.vlm = vlm
            self.logger = logger
            
            self.setup_tools(memory)
            
            prompt_dir = str(os.path.dirname(__file__)) + '/../prompts/last_seen_tool/'
            self.agent_prompt = file_to_string(prompt_dir+'agent_prompt.txt')
            self.agent_reflection_prompt = file_to_string(prompt_dir+'agent_reflection_prompt.txt')
            self.agent_gen_only_prompt = file_to_string(prompt_dir+'agent_gen_only_prompt.txt')
            
            self.agent_call_count = 0
        
        def setup_tools(self, memory: MilvusMemory):
            search_tools = create_memory_search_tools(memory)
            inspect_tools = create_memory_inspection_tool(memory)
            response_tools = create_recall_last_seen_terminate_tool(memory)
            reflect_tools = create_pause_and_think_tool()
            
            self.all_tools = search_tools + inspect_tools + response_tools + reflect_tools
            self.all_tool_definitions = [convert_to_openai_function(t) for t in self.all_tools]
            
            self.tools = search_tools + inspect_tools + response_tools
            self.tool_definitions = [convert_to_openai_function(t) for t in self.tools]
            
            self.reflect_tools = reflect_tools
            self.reflect_tool_definitions = [convert_to_openai_function(t) for t in self.reflect_tools]
            self.response_tools = response_tools
            self.response_tool_definitions = [convert_to_openai_function(t) for t in self.response_tools]

        def agent(self, state: AgentState):
            max_agent_call_count = 10 
            n_reflection_intervals = 5
            
            messages = state["messages"]
            
            additional_search_history = []
            last_message = messages[-1]
            if isinstance(last_message, ToolMessage):
                # ===  Step 1: Find last AIMessage with tool_calls
                idx = len(messages) - 1
                last_ai_idx = None
                while idx >= 0:
                    msg = messages[idx]
                    if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                        last_ai_idx = idx
                        break
                    idx -= 1

                # ===  Step 2: Append all following ToolMessages into search_in_time_history
                if last_ai_idx is not None:
                    image_messages = []
                    for msg in messages[last_ai_idx+1:]:
                        if isinstance(msg, ToolMessage):
                            original_msg_content = copy.copy(msg.content)
                            if isinstance(msg.content, str):
                                msg.content = parse_and_pretty_print_tool_message(msg.content)
                            additional_search_history.append(msg)
                            
                            if isinstance(original_msg_content, str) and is_image_inspection_result(original_msg_content):
                                inspection = eval(original_msg_content)
                                for id, path in inspection.items():
                                    content = get_image_message_for_record(id, path, msg.tool_call_id)
                                    message = HumanMessage(content=content)
                                    image_messages.append(message)
                            if self.logger:
                                self.logger.info(f"[LAST SEEN] Tool Response: {msg.content}")
                    
                    additional_search_history += image_messages
            
            chat_history = copy.deepcopy(state.get("agent_history", []))
            chat_history += additional_search_history
                    
            model = self.vlm
            model_flex = self.vlm_flex
            if self.agent_call_count < max_agent_call_count:
                if self.agent_call_count % n_reflection_intervals == 0:
                    prompt = self.agent_reflection_prompt
                    current_tool_defs = self.reflect_tool_definitions
                else:
                    prompt = self.agent_prompt
                    current_tool_defs = self.tool_definitions
            else:
                prompt = self.agent_gen_only_prompt
                current_tool_defs = self.response_tool_definitions
                
            model = model.bind_tools(current_tool_defs)
            model_flex = self.vlm_flex.bind_tools(current_tool_defs)
            tool_names = [tool['name'] for tool in current_tool_defs]
            tool_list_str = "\n".join([f"{i+1}. {name}" for i, name in enumerate(tool_names)])
                
            question_str = (
                f"Your peer agent wants to find when and where the following object was last seen in memory:\n"
                f"→ {self.description.strip()}\n"
            )
            if self.image_message:
                question_str += f"Your peer agent also provides an image where the target object was last observed:\n"
            
            # "Based on what you’ve observed and what you already know, what should you do next?"
            question_content = [{"type": "text", "text": question_str}]
            if self.image_message:
                question_content += self.image_message
            
            chat_template = [
                ("human", "You are a memory retrieval agent. Your job is to find the most recent memory record where the described object instance was last seen. Use tools to inspect and reason carefully before finalizing."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("system", prompt),
                ("system", "{fact_prompt}"),
                HumanMessage(content=question_content),
                ("human", "{caller_context_text}"),
                ("system", "Now decide your next action. Use tools to continue searching or terminate if you are confident. Reason carefully based on what you've done, what you know, and what the user ultimately needs. Pay attention to the time constraints if they are provided.")
            ]
            if self.agent_call_count < max_agent_call_count:
                chat_template += [
                    ("system", f"You must strictly follow the JSON output format. As a reminder, these are available tools: \n{tool_list_str}. You must use one of the tools to continue searching or finalize your decision without any additional explanation."),
                    ("system", f"🔄 You are allowed up to **{max_agent_call_count} iterations** total. This is iteration **#{self.agent_call_count}**.\nEach iteration consists of one full round of tool calls — even if you issue multiple tools in parallel, that still counts as one iteration.")
                ]
            else:
                chat_template += [
                    ("system", f"You must strictly follow the JSON output format. Since you have already reached the maximum number of iterations, you should finalize your decision now by calling: {tool_list_str}"),
                ]
                
            chat_prompt = ChatPromptTemplate.from_messages(chat_template)
            
            chained_model = chat_prompt | model
            chained_flex_model = chat_prompt | model_flex

            fact_prompt = f"Here are some facts for your context:\n" \
                      f"1. {self.memory.get_memory_stats_for_llm()}\n" \
                      f"2. You have been patrolling in a dynamic household or office environment, so objects you saw before may have been moved, or its status may be changed.\n"\
                      f"3. {self.time_str}\n"
            
            caller_context_text = self.caller_context if self.caller_context else "None"
            caller_context_text = f"Additional context from the caller agent:\n→ {caller_context_text}"
            
            input = {
                "chat_history": chat_history,
                "fact_prompt": fact_prompt,
                "caller_context_text": caller_context_text
            }
            response = safe_gpt_invoke(chained_flex_model, chained_model, input)
            
            if self.logger:
                if hasattr(response, "tool_calls") and response.tool_calls:
                    for call in response.tool_calls:
                        args_str = ", ".join(f"{k}={repr(v)}" for k, v in call.get("args", {}).items())
                        log_str = f"{call.get('name')}({args_str})"
                        self.logger.info(f"[LAST SEEN] Tool call: {log_str}")
                else:
                    self.logger.info(f"[LAST SEEN] {response}")
                    
            self.agent_call_count += 1
            return {"messages": [response], "agent_history": additional_search_history + [response]}
        
        def build_graph(self):
            workflow = StateGraph(LastSeenAgent.AgentState)
            
            workflow.add_node("agent", lambda state: try_except_continue(state, self.agent))
            workflow.add_node("action", ToolNode(self.all_tools))
            
            workflow.add_edge("action", "agent")
            workflow.add_conditional_edges(
                "agent",
                LastSeenAgent.from_agent_to,
                {
                    "next": END,
                    "action": "action",
                },
            )
            
            workflow.set_entry_point("agent")
            self.graph = workflow.compile()
        
        def run(self, 
            tool_rationale: str,
            description: str, 
            visual_cue_from_record_id: Optional[int] = None,
            search_start_time: Optional[str] = None, 
            search_end_time: Optional[str] = None,
            caller_context: Optional[str] = None,
        ) -> List[Dict]:
            
            if self.logger:
                self.logger.info(
                    f"[LAST_SEEN] Running tool with description: {description}, "
                    f"visual_cue_from_record_id: {visual_cue_from_record_id}, "
                    f"search_start_time: {search_start_time}, "
                    f"search_end_time: {search_end_time}"
                )
                
            self.description = description.strip()
            self.visual_cue_from_record_id = visual_cue_from_record_id
            
            if self.visual_cue_from_record_id is not None:
                self.viz_path = get_viz_path(self.memory, self.visual_cue_from_record_id)
                self.image_message = get_image_message_for_record(
                    self.visual_cue_from_record_id, 
                    self.viz_path, 
                )
            else:
                self.viz_path = None
                self.image_message = None
            
            self.search_start_time = search_start_time
            self.search_end_time = search_end_time
            self.caller_context = caller_context
            
            self.setup_tools(self.memory)
            self.build_graph()
            
            self.agent_call_count = 0
            
            content = []
            question_str = (
                f"Your peer agent wants to find when and where the following object was last seen in memory:\n"
                f"→ {self.description.strip()}\n"
            )
            if self.image_message is not None:
                question_str += f"Your peer agent also provides an image where the target object was last observed:\n"
            
            # "Based on what you’ve observed and what you already know, what should you do next?"
            content += [{"type": "text", "text": question_str}]
            if self.image_message:
                content += self.image_message
            
            time_str = "Your peer agent is only interested in information within the following time window:"
            
            start_t_str, end_t_str = self.memory.get_db_time_range()
            if search_start_time is not None and search_end_time is None:
                self.search_end_time = end_t_str
            elif search_start_time is None and search_end_time is not None:
                self.search_start_time = start_t_str
            
            if search_start_time is None and search_end_time is None:
                time_str += "\n→ None"
            else:
                time_str += f"\n→ Start: {search_start_time}"
                time_str += f"\n→ End: {search_end_time}"
                
            self.time_str = time_str
            
            content += [{"type": "text", "text": self.time_str}]
            
            inputs = {
                "messages": [
                    HumanMessage(content=content),
                ]
            }
            state = self.graph.invoke(inputs)
            
            output = {
                "tool_name": "recall_last_seen_terminate",
                "summary": "",
                "records": []
            }
            
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls"):
                tool_call = last_message.tool_calls[0] if last_message.tool_calls else None
                if tool_call and (type(tool_call) is dict) and ("args" in tool_call.keys()) and type(tool_call["args"]) is dict:
                    output["summary"] = tool_call["args"].get("summary", "")

                    record_id = tool_call["args"].get("record_id", -1)
                    if record_id == -1:
                        records = []
                    else:
                        record = eval(memory.get_by_id(record_id))
                        if record:
                            records = record
                        else:
                            records = []
                    output["records"] = records
                    
            return output

    tool_runner = LastSeenAgent(memory, vlm_flex, vlm, logger)

    class LastSeenInput(BaseModel):
        tool_rationale: str = Field(
            description=TOOL_RATIONALE_DESC
        )
        description: str = Field(
            description=(
                "Text query.\n"
                "• Text-only: DEFINES the target to retrieve (e.g., 'the algebra book', 'the red mug on the kitchen table').\n"
                "• With visual_cue_from_record_id: acts as a VERBAL POINTER to the object IN that reference image "
                "(e.g., 'the red mug next to the painting')."
            )
        )
        visual_cue_from_record_id: Optional[int] = Field(
            default=None,
            description=(
                "ID of a memory record whose image serves as the visual ANCHOR. "
                "When set, the tool tracks the SAME INSTANCE seen in that image; "
                "'description' indicates which object in the image you mean."
            )
        )
        search_start_time: Optional[str] = Field(
            default=None,
            description="Start time (inclusive) for searching memory. Can be an ISO string or datetime. Leave blank for no lower bound."
        )
        search_end_time: Optional[str] = Field(
            default=None,
            description="End time (inclusive) for searching memory. Can be an ISO string or datetime. Leave blank for no upper bound."
        )
        caller_context: Optional[str] = Field(
            default=None,
            description="Additional free-text context from the caller agent to help guide the retrieval — e.g., what the caller is trying to do, what it is uncertain about, or how the results will be used."
        )
        
    last_seen_tool = StructuredTool.from_function(
        func=tool_runner.run,
        name="recall_last_seen",
        description=(
            "Return the most recent memory record where the target appears.\n"
            "• Text-only: 'description' DEFINES the target.\n"
            "• With visual_cue_from_record_id: the image ANCHORS the instance; 'description' POINTS to the object in that image.\n"
            "Use time fields to bound recency; scene hints in text guide retrieval but do not constrain later sightings."
        ),
        args_schema=LastSeenInput,
    )
        
    return [last_seen_tool]
        
def create_recall_all_terminate_tool(memory: MilvusMemory) -> StructuredTool:
    
    class RecallAllTerminateInput(BaseModel):
        summary: str = Field(
            description="A short explanation of what set of records is being returned and why they are relevant"
        )
        record_ids: List[int] = Field(
            description=(
                "List of memory record IDs where the described object plausibly appears. "
                "You must not return more than 50. If there are more, return the most representative and informative ones."
            )
        )
    
    def _terminate_fn(
        summary: str,
        record_ids: List[int]
    ) -> str:
        records = []
        for record_id in record_ids[:50]:  # Enforce hard cap
            record = eval(memory.get_by_id(record_id))
            if record:
                records += record
        return str(records)

    terminate_tool = StructuredTool.from_function(
        func=_terminate_fn,
        name="recall_all_terminate",
        description=(
            "Use this to finalize the recall-all task once you have gathered all plausible records where the object may appear. "
            "This tool should be called when your retrieval is complete and you've curated a set of representative results."
        ),
        args_schema=RecallAllTerminateInput
    )
    
    return [terminate_tool]


def create_recall_all_tool(memory: MilvusMemory, vlm_flex, vlm, logger=None) -> StructuredTool:

    class RecallAllAgent:
        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]
            agent_history: Annotated[Sequence[BaseMessage], add_messages]
            
        @staticmethod
        def from_agent_to(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1] if messages else None
            
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for call in last_message.tool_calls:
                    if "terminate" in call.get("name"):
                        return "next"
            return "action"

        def __init__(self, memory, vlm_flex, vlm, logger=None):
            self.memory = memory
            self.vlm_flex = vlm_flex
            self.vlm = vlm
            self.logger = logger
            
            self.setup_tools(memory)
            
            prompt_dir = str(os.path.dirname(__file__)) + '/../prompts/recall_all_tool/'
            self.agent_prompt = file_to_string(prompt_dir+'agent_prompt.txt')
            self.agent_reflection_prompt = file_to_string(prompt_dir+'agent_reflection_prompt.txt')
            self.agent_gen_only_prompt = file_to_string(prompt_dir+'agent_gen_only_prompt.txt')
            
            self.agent_call_count = 0
            self.max_agent_call_count = 5
            
        def setup_tools(self, memory: MilvusMemory):
            search_tools = create_memory_search_tools(memory)
            inspect_tools = create_memory_inspection_tool(memory)
            response_tools = create_recall_all_terminate_tool(memory)
            reflect_tools = create_pause_and_think_tool()
            
            self.all_tools = search_tools + inspect_tools + response_tools + reflect_tools
            self.all_tool_definitions = [convert_to_openai_function(t) for t in self.all_tools]
            
            self.tools = search_tools + inspect_tools + response_tools
            self.tool_definitions = [convert_to_openai_function(t) for t in self.tools]
            
            self.reflect_tools = reflect_tools
            self.reflect_tool_definitions = [convert_to_openai_function(t) for t in self.reflect_tools]
            self.response_tools = response_tools
            self.response_tool_definitions = [convert_to_openai_function(t) for t in self.response_tools]
        
        def agent(self, state: AgentState):
            max_agent_call_count = 10 
            n_reflection_intervals = 5
            
            messages = state["messages"]
            
            additional_search_history = []
            last_message = messages[-1]
            if isinstance(last_message, ToolMessage):
                # ===  Step 1: Find last AIMessage with tool_calls
                idx = len(messages) - 1
                last_ai_idx = None
                while idx >= 0:
                    msg = messages[idx]
                    if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                        last_ai_idx = idx
                        break
                    idx -= 1

                # ===  Step 2: Append all following ToolMessages
                if last_ai_idx is not None:
                    image_messages = []
                    for msg in messages[last_ai_idx+1:]:
                        if isinstance(msg, ToolMessage):
                            original_msg_content = copy.copy(msg.content)
                            if isinstance(msg.content, str):
                                msg.content = parse_and_pretty_print_tool_message(msg.content)
                            additional_search_history.append(msg)
                            
                            if isinstance(original_msg_content, str) and is_image_inspection_result(original_msg_content):
                                inspection = eval(original_msg_content)
                                for id, path in inspection.items():
                                    content = get_image_message_for_record(id, path, msg.tool_call_id)
                                    message = HumanMessage(content=content)
                                    image_messages.append(message)
                            if self.logger:
                                self.logger.info(f"[RECALL ALL] Tool Response: {msg.content}")
                    
                    additional_search_history += image_messages
            
            chat_history = copy.deepcopy(state.get("agent_history", []))
            chat_history += additional_search_history
                    
            model = self.vlm
            model_flex = self.vlm_flex
            if self.agent_call_count < max_agent_call_count:
                if self.agent_call_count % n_reflection_intervals == 0:
                    prompt = self.agent_reflection_prompt
                    current_tool_defs = self.reflect_tool_definitions
                else:
                    prompt = self.agent_prompt
                    current_tool_defs = self.tool_definitions
            else:
                prompt = self.agent_gen_only_prompt
                current_tool_defs = self.response_tool_definitions
                
            model = model.bind_tools(current_tool_defs)
            model_flex = model_flex.bind_tools(current_tool_defs)
            tool_names = [tool['name'] for tool in current_tool_defs]
            tool_list_str = "\n".join([f"{i+1}. {name}" for i, name in enumerate(tool_names)])
                
            question_str = (
                f"Your peer agent wants to find all plausible past memory records where the following object may have appeared::\n"
                f"→ {self.description.strip()}\n"
            )
            if self.image_message:
                question_str += f"Your peer agent also provides an image where the target object was last observed:\n"
            
            # "Based on what you’ve observed and what you already know, what should you do next?"
            question_content = [{"type": "text", "text": question_str}]
            if self.image_message:
                question_content += self.image_message
            
            chat_template = [
                ("human", "You are a memory retrieval agent. Your job is to retrieve all plausible memory records where the described object may have appeared. Use tools to inspect and reason carefully before finalizing."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("system", prompt),
                ("system", "{fact_prompt}"),
                HumanMessage(content=question_content),
                ("human", "{caller_context_text}"),
                ("system", "Now decide your next action. Use tools to continue searching or terminate if you are confident. Reason carefully based on what you've done, what you know, and what the user ultimately needs. Pay attention to the time constraints if they are provided.")
            ]
            if self.agent_call_count < max_agent_call_count:
                chat_template += [
                    ("system", f"You must strictly follow the JSON output format. As a reminder, these are available tools: \n{tool_list_str}. You must use one of the tools to continue searching or finalize your decision without any additional explanation."),
                    ("system", f"🔄 You are allowed up to **{max_agent_call_count} iterations** total. This is iteration **#{self.agent_call_count}**.\nEach iteration consists of one full round of tool calls — even if you issue multiple tools in parallel, that still counts as one iteration.")
                ]
            else:
                chat_template += [
                    ("system", f"You must strictly follow the JSON output format. Since you have already reached the maximum number of iterations, you should finalize your decision now by calling: {tool_list_str}"),
                ]
                
            chat_prompt = ChatPromptTemplate.from_messages(chat_template)
            
            chained_model = chat_prompt | model
            chained_flex_model = chat_prompt | model_flex

            fact_prompt = f"Here are some facts for your context:\n" \
                      f"1. {self.memory.get_memory_stats_for_llm()}\n" \
                      f"2. You have been patrolling in a dynamic household or office environment, so objects you saw before may have been moved, or its status may be changed.\n" \
                      f"3. {self.time_str}\n"
            
            caller_context_text = self.caller_context if self.caller_context else "None"
            caller_context_text = f"Additional context from the caller agent:\n→ {caller_context_text}"
            
            input = {
                "chat_history": chat_history,
                "fact_prompt": fact_prompt,
                "caller_context_text": caller_context_text
            }
            response = safe_gpt_invoke(chained_flex_model, chained_model, input)
            
            if self.logger:
                if hasattr(response, "tool_calls") and response.tool_calls:
                    for call in response.tool_calls:
                        args_str = ", ".join(f"{k}={repr(v)}" for k, v in call.get("args", {}).items())
                        log_str = f"{call.get('name')}({args_str})"
                        self.logger.info(f"[RECALL ALL] Tool call: {log_str}")
                else:
                    self.logger.info(f"[RECALL ALL] {response}")
                    
            self.agent_call_count += 1
            return {"messages": [response], "agent_history": additional_search_history + [response]}
        
        def build_graph(self):
            workflow = StateGraph(RecallAllAgent.AgentState)
            
            workflow.add_node("agent", lambda state: try_except_continue(state, self.agent))
            workflow.add_node("action", ToolNode(self.all_tools))
            
            workflow.add_edge("action", "agent")
            workflow.add_conditional_edges(
                "agent",
                RecallAllAgent.from_agent_to,
                {
                    "next": END,
                    "action": "action",
                },
            )
            
            workflow.set_entry_point("agent")
            self.graph = workflow.compile()
        
        def run(self, 
            tool_rationale: str,
            description: str, 
            visual_cue_from_record_id: Optional[int] = None,
            search_start_time: Optional[str] = None, 
            search_end_time: Optional[str] = None,
            caller_context: Optional[str] = None,
        ) -> List[Dict]:
            
            if self.logger:
                self.logger.info(
                    f"[RECALL ALL] Running tool with description: {description}, "
                    f"visual_cue_from_record_id: {visual_cue_from_record_id}, "
                    f"search_start_time: {search_start_time}, "
                    f"search_end_time: {search_end_time}"
                )
                
            self.description = description.strip()
            self.visual_cue_from_record_id = visual_cue_from_record_id
            
            if self.visual_cue_from_record_id is not None:
                self.viz_path = get_viz_path(self.memory, self.visual_cue_from_record_id)
                self.image_message = get_image_message_for_record(
                    self.visual_cue_from_record_id, 
                    self.viz_path, 
                )
            else:
                self.viz_path = None
                self.image_message = None
            
            self.search_start_time = search_start_time
            self.search_end_time = search_end_time
            self.caller_context = caller_context
            
            self.setup_tools(self.memory)
            self.build_graph()
            
            self.agent_call_count = 0
            
            content = []
            question_str = (
                f"Your peer agent wants to find all plausible memory records where the following object may have appeared:\n"
                f"→ {self.description.strip()}\n"
            )
            if self.image_message is not None:
                question_str += f"Your peer agent also provides an image where the target object was previously observed:\n"
            
            content += [{"type": "text", "text": question_str}]
            if self.image_message:
                content += self.image_message
            
            time_str = "Your peer agent is only interested in information within the following time window:"
            
            start_t_str, end_t_str = self.memory.get_db_time_range()
            if search_start_time is not None and search_end_time is None:
                self.search_end_time = end_t_str
            elif search_start_time is None and search_end_time is not None:
                self.search_start_time = start_t_str
            
            if search_start_time is None and search_end_time is None:
                time_str += "\n→ None"
            else:
                time_str += f"\n→ Start: {search_start_time}"
                time_str += f"\n→ End: {search_end_time}"
                
            self.time_str = time_str
            
            content += [{"type": "text", "text": self.time_str}]
            
            inputs = {
                "messages": [
                    HumanMessage(content=content),
                ]
            }
            state = self.graph.invoke(inputs)
            
            output = {
                "tool_name": "recall_all_terminate",
                "summary": "",
                "records": []
            }
            
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls"):
                tool_call = last_message.tool_calls[0] if last_message.tool_calls else None
                if tool_call and (type(tool_call) is dict) and ("args" in tool_call.keys()) and type(tool_call["args"]) is dict:
                    output["summary"] = tool_call["args"].get("summary", "")

                    record_ids = tool_call["args"].get("record_ids", [])
                    records = []
                    for record_id in record_ids:
                        record = eval(memory.get_by_id(record_id))
                        if record:
                            records += record
                    output["records"] = records
                    
            return output


    tool_runner = RecallAllAgent(memory, vlm_flex, vlm, logger)

    class RecallAllInput(BaseModel):
        tool_rationale: str = Field(
            description=TOOL_RATIONALE_DESC
        )
        description: str = Field(
            description=(
                "Text query.\n"
                "• Text-only: DEFINES the target to recall (e.g., 'a red mug', 'the algebra book').\n"
                "• With visual_cue_from_record_id: acts as a VERBAL POINTER to the object IN that image "
                "(e.g., 'the red mug on the left'). The image anchors the instance."
            )
        )
        visual_cue_from_record_id: Optional[int] = Field(
            default=None,
            description=(
                "ID of a memory record whose image serves as the visual ANCHOR. "
                "When provided, the tool recalls appearances of the SAME INSTANCE seen in that image; "
                "'description' indicates which object in the image you mean."
            )
        )
        search_start_time: Optional[str] = Field(
            default=None,
            description="Start time (inclusive) for searching memory. Can be an ISO string or datetime. Leave blank for no lower bound."
        )
        search_end_time: Optional[str] = Field(
            default=None,
            description="End time (inclusive) for searching memory. Can be an ISO string or datetime. Leave blank for no upper bound."
        )
        caller_context: Optional[str] = Field(
            default=None,
            description="Additional free-text context from the caller agent to help guide the retrieval — e.g., what the caller is trying to do, what it is uncertain about, or how the results will be used."
        )
            
    recall_all_tool = StructuredTool.from_function(
        func=tool_runner.run,
        name="recall_all",
        description=(
            "Recall all plausible memory records where the target appears within the given time window.\n"
            "• Text-only: 'description' DEFINES the target (semantic retrieval).\n"
            "• With visual_cue_from_record_id: the image ANCHORS the instance; 'description' VERBALLY POINTS to it in the image.\n"
            "Useful for typical locations, usage patterns, and frequency. Results may include near-misses—verify as needed."
        ),
        args_schema=RecallAllInput,
    )
    
    return [recall_all_tool]

def create_search_in_time_evaluation_tool():
    
    class ReviewObjectReferenceAndRetrievalTerminateInput(BaseModel):
        review_rationale: str = Field(
            description="A concise explanation of your reasoning for both answers. Include how the agent’s actions support your selected records and note any ambiguity or uncertainty."
        )
        reference_resolution_record_id: int = Field(
            description="The memory record ID that best shows which object the agent took the user to be referring to, satisfying any key constraints in the user query (e.g., spatial, temporal, descriptive)."
        )
        retrieval_grounding_record_id: int = Field(
            description="The memory record ID that shows where the agent retrieved the object from, and whether this was the most recent valid sighting of the same object."
        )

    def _terminate_review_fn(
        review_rationale: str,
        reference_resolution_record_id: int,
        retrieval_grounding_record_id: int,
    ) -> bool:
        # Dummy implementation: always return True
        return True

    review_terminate_tool = StructuredTool.from_function(
        func=_terminate_review_fn,
        name="review_object_reference_and_retrieval_terminate",
        description=(
            "Use this tool to **finalize your review** of the agent’s object retrieval decision.\n"
            "You must answer two questions based solely on the agent’s tool use and reasoning trace:\n"
            "1. Which memory record shows the object instance the agent believed the user was referring to?\n"
            "2. Which memory record shows where the agent retrieved that object from — and is it the most recent sighting?\n\n"
        ),
        args_schema=ReviewObjectReferenceAndRetrievalTerminateInput,
    )

    return [review_terminate_tool]