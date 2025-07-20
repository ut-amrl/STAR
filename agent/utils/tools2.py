import sys
import os
import re
from typing import List, Dict, Optional
from typing import Annotated, Sequence, TypedDict
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from pydantic import conint

from langchain.tools import StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from memory.memory import MilvusMemory
from agent.utils.function_wrapper import FunctionsWrapper
from agent.utils.utils import *

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

def create_memory_terminate_tool() -> List[StructuredTool]:

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
            description="A list of memory record IDs that support the decision"
        )

    def _terminate_fn(
        summary: str,
        instance_description: str,
        position: List[float],
        theta: float,
        record_ids: List[int],
    ) -> bool:
        # Dummy implementation: always return True
        return True

    terminate_tool = StructuredTool.from_function(
        func=_terminate_fn,
        name="terminate",
        description=(
            "Use this to **finalize the task** once you are confident about what to retrieve and where to go.\n\n"
            "- Required fields:\n"
            "  - `position`: 3D target coordinate (e.g., `[x, y, z]`)\n"
            "  - `theta`: Orientation angle in radians\n"
            "  - `record_ids`: A list of record IDs that support your conclusion\n"
            "  - `summary`: A short explanation of what is being retrieved and why\n\n"
            "- Constraint: Must be called **alone** (no other tools should be used in the same step)."
        ),
        args_schema=MemoryTerminateInput,
    )

    return [terminate_tool]
        

def create_memory_search_tools(memory: MilvusMemory):

    TOOL_RATIONALE_DESC = (
        "Explain briefly why this tool is being called. The rationale should clarify how this tool helps move the reasoning forward — "
        "what is already known, what new insight is expected from the call, and what uncertainty or open question this tool is meant to resolve. "
        "Avoid vague or redundant justifications; focus on the unique purpose of this tool in the current context."
    )

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

def create_db_txt_search_k_tool(memory: MilvusMemory):
    class TextRetrieverWithKInput(BaseModel):
        x: str = Field(
            description="The query to search for in the memory. This should describe what you're trying to retrieve, like 'a blue mug on the table'."
        )
        k: conint(ge=int(1), le=int(50)) = Field(
            default=8,
            description="The number of top memory matches to retrieve. Must be between 1 and 50."
        )

    def _search_with_k(x: str, k: int):
        return memory.search_by_text(x, k=k)

    txt_retriever_k_tool = StructuredTool.from_function(
        func=_search_with_k,
        name="retrieve_top_k_from_text",
        description="Search memory for the top-k most relevant past observations using a natural language query.",
        args_schema=TextRetrieverWithKInput
    )
    return [txt_retriever_k_tool]

def create_db_txt_backward_search_tool(memory: MilvusMemory):
    pass 

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
            record = memory.get_by_id(record_id)
            if record:
                records.append(record)
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
    
def create_recall_best_matches_tool(
    memory: MilvusMemory,
    llm,
    llm_raw,
    vlm,
    vlm_raw,
    logger=None
):
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
    
        def __init__(self, memory, llm, llm_raw, vlm, vlm_raw, logger=None):
            self.memory = memory
            self.llm = llm
            self.llm_raw = llm_raw
            self.vlm = vlm
            self.vlm_raw = vlm_raw
            self.logger = logger
            
            self.setup_tools(memory)
            
            prompt_dir = str(os.path.dirname(__file__)) + '/../prompts/best_matches_tool/'
            self.agent_prompt = file_to_string(prompt_dir+'agent_prompt.txt')
            self.agent_gen_only_prompt = file_to_string(prompt_dir+'agent_gen_only_prompt.txt')
            
            self.agent_call_count = 0
            
        def setup_tools(self, memory: MilvusMemory):
            search_tools = create_memory_search_tools(memory)
            inspect_tools = create_memory_inspection_tool(memory)
            response_tools = create_recall_best_matches_terminate_tool(memory)
            reflect_tools = create_pause_and_think_tool()
            
            self.tools = search_tools + inspect_tools + response_tools
            self.tool_definitions = [convert_to_openai_function(t) for t in self.tools]
            
            self.reflect_tools = reflect_tools
            self.reflect_tool_definitions = [convert_to_openai_function(t) for t in self.reflect_tools]
            self.response_tools = response_tools
            self.response_tool_definitions = [convert_to_openai_function(t) for t in self.response_tools]
            
        def agent(self, state: AgentState):
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
                            if isinstance(msg.content, str):
                                msg.content = parse_and_pretty_print_tool_message(msg.content)
                            additional_search_history.append(msg)
                            
                            if isinstance(msg.content, str) and is_image_inspection_result(msg.content):
                                inspection = eval(msg.content)
                                for id, path in inspection.items():
                                    content = get_image_message_for_record(id, path, msg.tool_call_id)
                                    message = HumanMessage(content=content)
                                    image_messages.append(message)
                            if self.logger:
                                self.logger.info(f"[BEST MATCHES] Tool Response: {msg.content}")
                                
                    # if len(image_messages) > 0:
                    #     chat_prompt = ChatPromptTemplate.from_messages([
                    #         MessagesPlaceholder(variable_name="chat_history"),
                    #         ("user", "Description of the image(s) you have seen."),
                    #     ])
                    #     chained_model = chat_prompt | self.vlm_raw
                    #     response = chained_model.invoke({
                    #         "chat_history": image_messages,
                    #     })
                    #     print(response)
                    #     import pdb; pdb.set_trace()
                    
                    additional_search_history += image_messages
            
            chat_history = state.get("agent_history", [])
            chat_history += additional_search_history
                    
            max_agent_call_count = 8

            model = self.vlm
            if self.agent_call_count < max_agent_call_count:
                prompt = self.agent_prompt
                model = model.bind_tools(self.tool_definitions)
            else:
                prompt = self.agent_gen_only_prompt
                model = model.bind_tools(self.response_tool_definitions)
                
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
                ("system", "Now decide your next action. Use tools to continue searching or terminate if you are confident. Reason carefully based on what you’ve done, what you know, and what the user ultimately needs.")
            ]
            chat_prompt = ChatPromptTemplate.from_messages(chat_template)
            
            chained_model = chat_prompt | model
            
            fact_prompt = f"Here are some facts for your context:\n" \
                      f"1. {self.memory.get_memory_stats_for_llm()}\n" \
                      f"2. You have been patrolling in a dynamic household or office environment, so objects you saw before may have been moved, or its status may be changed.\n"
            caller_context_text = self.caller_context if self.caller_context else "None"
            caller_context_text = f"Additional context from the caller agent:\n→ {caller_context_text}"
            
            response = chained_model.invoke({
                "chat_history": chat_history,
                "fact_prompt": fact_prompt,
                "caller_context_text": caller_context_text
            })
            
            if self.logger:
                if hasattr(response, "tool_calls") and response.tool_calls:
                    for call in response.tool_calls:
                        args_str = ", ".join(f"{k}={repr(v)}" for k, v in call.get("args", {}).items())
                        log_str = f"{call.get('name')}({args_str})"
                        self.logger.info(f"[BEST MATCH] Tool call: {log_str}")
                else:
                    self.logger.info(f"[BEST MATCH] {response}")
                    
            self.agent_call_count += 1
            return {"messages": [response], "agent_history": additional_search_history + [response]}
        
        def build_graph(self):
            workflow = StateGraph(BestMatchAgent.AgentState)
            
            workflow.add_node("agent", lambda state: try_except_continue(state, self.agent))
            workflow.add_node("action", ToolNode(self.tools))
            
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
            time_str = "\n\nTime constraints:"
            if search_start_time:
                time_str += f"\n→ Start: {search_start_time}"
            if search_end_time:
                time_str += f"\n→ End: {search_end_time}"
            if not search_start_time and not search_end_time:
                time_str += "\n→ None"
            content += [{"type": "text", "text": time_str}]
            
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
                        record = memory.get_by_id(record_id)
                        if record:
                            records.append(record)
                    output["records"] = records
                    
            return output
        
    tool_runner = BestMatchAgent(memory, llm, llm_raw, vlm, vlm_raw, logger)
            
    class BestMatchesInput(BaseModel):
        description: str = Field(
            description="Text description of the object or scene to search for, e.g., 'the red mug on the kitchen table'."
        )
        visual_cue_from_record_id: Optional[int] = Field(
            default=None,
            description="ID of a memory record that contains an image of the object to be retrieved. Used as a visual cue for grounding."
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
            "Recall up to k memory records that best match a given object or scene description. "
            "You may also provide a visual cue (via record ID) and a time window to guide retrieval. "
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

def create_recall_last_seen_tool(
    memory: MilvusMemory,
    llm,
    llm_raw,
    vlm,
    vlm_raw,
    logger=None
) -> StructuredTool:
    
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
        
        def __init__(self, memory, llm, llm_raw, vlm, vlm_raw, logger=None):
            self.memory = memory
            self.llm = llm
            self.llm_raw = llm_raw
            self.vlm = vlm
            self.vlm_raw = vlm_raw
            self.logger = logger
            
            self.setup_tools(memory)
            
            prompt_dir = str(os.path.dirname(__file__)) + '/../prompts/last_seen_tool/'
            self.agent_prompt = file_to_string(prompt_dir+'agent_prompt.txt')
            self.agent_gen_only_prompt = file_to_string(prompt_dir+'agent_gen_only_prompt.txt')
            
            self.agent_call_count = 0
        
        def setup_tools(self, memory: MilvusMemory):
            search_tools = create_memory_search_tools(memory)
            inspect_tools = create_memory_inspection_tool(memory)
            response_tools = create_recall_last_seen_terminate_tool(memory)
            reflect_tools = create_pause_and_think_tool()
            
            self.tools = search_tools + inspect_tools + response_tools
            self.tool_definitions = [convert_to_openai_function(t) for t in self.tools]
            
            self.reflect_tools = reflect_tools
            self.reflect_tool_definitions = [convert_to_openai_function(t) for t in self.reflect_tools]
            self.response_tools = response_tools
            self.response_tool_definitions = [convert_to_openai_function(t) for t in self.response_tools]

        def agent(self, state: AgentState):
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
                            if isinstance(msg.content, str):
                                msg.content = parse_and_pretty_print_tool_message(msg.content)
                            additional_search_history.append(msg)
                            
                            if isinstance(msg.content, str) and is_image_inspection_result(msg.content):
                                inspection = eval(msg.content)
                                for id, path in inspection.items():
                                    content = get_image_message_for_record(id, path, msg.tool_call_id)
                                    message = HumanMessage(content=content)
                                    image_messages.append(message)
                            if self.logger:
                                self.logger.info(f"[BEST MATCHES] Tool Response: {msg.content}")
                    
                    additional_search_history += image_messages
            
            chat_history = state.get("agent_history", [])
            chat_history += additional_search_history
                    
            max_agent_call_count = 8

            model = self.vlm
            if self.agent_call_count < max_agent_call_count:
                prompt = self.agent_prompt
                model = model.bind_tools(self.tool_definitions)
            else:
                prompt = self.agent_gen_only_prompt
                model = model.bind_tools(self.response_tool_definitions)
                
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
                ("system", "Now decide your next action. Use tools to continue searching or terminate if you are confident. Reason carefully based on what you’ve done, what you know, and what the user ultimately needs.")
            ]
            chat_prompt = ChatPromptTemplate.from_messages(chat_template)
            
            chained_model = chat_prompt | model
            
            fact_prompt = f"Here are some facts for your context:\n" \
                      f"1. {self.memory.get_memory_stats_for_llm()}\n" \
                      f"2. You have been patrolling in a dynamic household or office environment, so objects you saw before may have been moved, or its status may be changed.\n"
            caller_context_text = self.caller_context if self.caller_context else "None"
            caller_context_text = f"Additional context from the caller agent:\n→ {caller_context_text}"
            
            try:
                response = chained_model.invoke({
                    "chat_history": chat_history,
                    "fact_prompt": fact_prompt,
                    "caller_context_text": caller_context_text
                })
            except Exception as e:
                import pdb; pdb.set_trace()
            
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
            workflow.add_node("action", ToolNode(self.tools))
            
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
            
            time_str = "\n\nTime constraints:"
            if search_start_time:
                time_str += f"\n→ Start: {search_start_time}"
            if search_end_time:
                time_str += f"\n→ End: {search_end_time}"
            if not search_start_time and not search_end_time:
                time_str += "\n→ None"
            content += [{"type": "text", "text": time_str}]
            
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
                        record = memory.get_by_id(record_id)
                        if record:
                            records = [record]
                        else:
                            records = []
                    output["records"] = records
                    
            return output
        
    tool_runner = LastSeenAgent(memory, llm, llm_raw, vlm, vlm_raw, logger)

    class LastSeenInput(BaseModel):
        description: str = Field(
            description="Text description of the object or scene to search for, e.g., 'the red mug on the kitchen table'."
        )
        visual_cue_from_record_id: Optional[int] = Field(
            default=None,
            description="ID of a memory record that contains an image of the object to be retrieved. Used as a visual cue for grounding."
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
            "Recall the last seen memory record that best matches a given object or scene description. "
        ),
        args_schema=LastSeenInput,
    )
        
    return [last_seen_tool]
        

def create_recall_all_tool(
    memory: MilvusMemory,
    llm,
    llm_raw,
    vlm,
    vlm_raw,
    logger=None
) -> StructuredTool:
    
    def from_agent_to(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "generate"
        else:
            return "action"
    
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        output: Annotated[Sequence, replace_messages] = None
    
    class RecallAllAgent:
        def __init__(self, memory, llm, llm_raw, vlm, vlm_raw, logger=None):
            self.memory = memory
            self.llm = llm
            self.llm_raw = llm_raw
            self.vlm = vlm
            self.vlm_raw = vlm_raw
            self.logger = logger
            
            self.setup_tools(memory)
            
            prompt_dir = str(os.path.dirname(__file__)) + '/../prompts/recall_all_tool/'
            self.agent_prompt = file_to_string(prompt_dir+'agent_prompt.txt')
            self.agent_gen_only_prompt = file_to_string(prompt_dir+'agent_gen_only_prompt.txt')
            
            self.agent_call_count = 0
            self.previous_tool_requests = "I have already used the following retrieval tools and the results are included below. Do not repeat them:\n"
            
        def setup_tools(self, memory: MilvusMemory):
            self.tools = create_db_txt_search_k_tool(memory)
            self.tool_definitions = [convert_to_openai_function(t) for t in self.tools]
            
        def agent(self, state: AgentState):
            pass
        
        def generate(self, state: AgentState):
            pass
        
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
        
        def run(self, 
                user_task: str, 
                history_summary: str, 
                current_task: str, 
                instance_description: str, 
                search_start_time: Optional[str] = None, 
                search_end_time: Optional[str] = None) -> List[Dict]:
            
            if self.logger:
                self.logger.info(
                    f"[RECALL_ALL] Running tool with user_task: {user_task}, "
                    f"current_task: {current_task}, "
                    f"instance_description: {instance_description}, "
                    f"search_start_time: {search_start_time}, search_end_time: {search_end_time}"
                )
                
            self.user_task = user_task
            self.history_summary = history_summary
            self.current_task = current_task
            self.instance_description = instance_description
            self.search_start_time = search_start_time
            self.search_end_time = search_end_time
            
            # TODO implement the actual logic here
            return []
        
    class RecallAllInput(BaseModel):
        user_task: str = Field(
            description="The high-level task the user wants to perform. For example, 'bring me the book I was reading yesterday'."
        )
        history_summary: str = Field(
            description="A summary of the prior reasoning or dialog history that provides context for this recall."
        )
        current_task: str = Field(
            description="The current subtask being worked on. For example, 'retrieve all possible matching observations for analysis'."
        )
        instance_description: str = Field(
            description="A description of the object or event you want to recall from memory. E.g., 'red suitcase', or 'any instance of the blue mug on the table'."
        )
        search_start_time: Optional[str] = Field(
            default=None,
            description="(Optional) Start of time window, format 'YYYY-MM-DD HH:MM:SS'. Results after this time will be considered."
        )
        search_end_time: Optional[str] = Field(
            default=None,
            description="(Optional) End of time window, format 'YYYY-MM-DD HH:MM:SS'. Results before this time will be considered."
        )
        
    tool_runner = RecallAllAgent(memory, llm, llm_raw, vlm, vlm_raw, logger)
    recall_all_tool = StructuredTool.from_function(
        func=lambda user_task, history_summary, current_task, instance_description, search_start_time=None, search_end_time=None: 
            tool_runner.run(user_task, history_summary, current_task, instance_description, search_start_time, search_end_time),
        name="recall_all",
        description="Recall all memory records that match a given object or scene description. Use this when you want to see the full pattern or history of an object across time.",
        args_schema=RecallAllInput
    )
    return [recall_all_tool]

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
                    for k, v in val.items():
                        image_paths[int(k)] = v
                except Exception:
                    continue

            memory_messages = []
            is_grouped = (
                isinstance(self.memory_records, list)
                and all(isinstance(x, dict) and "instance_desc" in x for x in self.memory_records)
            )

            if is_grouped:
                for i, group in enumerate(self.memory_records):
                    instance_desc = group.get("instance_desc", "(no description)")
                    records = group.get("records", [])
                    memory_messages.append({"type": "text", "text": f"--- Instance {i+1}: {instance_desc} ---"})
                    for record in records:
                        memory_messages += self._record_to_message(record, image_paths)
            else:
                for record in self.memory_records:
                    memory_messages += self._record_to_message(record, image_paths)

            return memory_messages

        def _record_to_message(self, record: Dict, image_paths: Dict[int, str]) -> List[Dict]:
            msgs = [{"type": "text", "text": parse_db_records_for_llm(record)}]
            if int(record["id"]) in image_paths:
                image_path = image_paths[int(record["id"])]
                img = PILImage.open(image_path).convert("RGB")
                img = img.resize((512, 512), PILImage.BILINEAR)
                
                draw = ImageDraw.Draw(img)
                text = f"record_id: {record['id']}"
                font = ImageFont.load_default()
                text_size = draw.textbbox((0, 0), text, font=font)
                padding = 4
                bg_rect = (
                    text_size[0] - padding,
                    text_size[1] - padding,
                    text_size[2] + padding,
                    text_size[3] + padding
                )
                draw.rectangle(bg_rect, fill=(0, 0, 0))
                draw.text((text_size[0], text_size[1]), text, fill=(255, 255, 255), font=font)

                buffer = BytesIO()
                img.save(buffer, format="PNG")
                img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                msgs.append(get_vlm_img_message(img_b64, type="gpt"))
            return msgs
            
        def decide(self, state: AgentState):
            messages = state["messages"]
            
            model = self.vlm
            if self.decide_call_count > 2:
                prompt = self.decide_gen_only_prompt
            else:
                prompt = self.decide_prompt
                model = model.bind_tools(self.tool_definitions)
                
            memory_messages = self._parse_memory_records(messages)
            chat_prompt = ChatPromptTemplate.from_messages([
                HumanMessage(content=memory_messages),
                # ("human", self.previous_tool_requests),
                ("user", prompt),
                ("human", "{question}"),
                ("system", "Remember to follow the json format strictly and only use the tools provided. Do not generate any text outside of tool calls. If you are not sure, call the most appropriate tool to make a best answer.")
            ])
            chained_model = chat_prompt | model
            question  = f"User Task: {self.user_task}\n" \
                        f"History Summary: {self.history_summary}\n" \
                        f"Current Task: {self.current_task}\n"
                        
            response = chained_model.invoke({"question": question, "memory_records": memory_messages})
        
            if self.logger:
                if getattr(response, 'tool_calls'):
                    self.logger.info(f"[DETERMINE_SEARCH_INSTANCE] decide() - Tool calls present: {response.tool_calls}")
                else:
                    self.logger.info(f"[DETERMINE_SEARCH_INSTANCE] decide() - {response.content}.")
        
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
            valid_found_values = {"yes", "no", "unknown"}
            
            prompt = self.decide_gen_only_prompt
            
            memory_messages = self._parse_memory_records(messages)
            chat_prompt = ChatPromptTemplate.from_messages([
                HumanMessage(content=memory_messages),
                ("user", prompt),
                ("human", "{question}"),
                ("system", "Remember to follow the json format strictly and only use the tools provided. Do not generate any text outside of tool calls. If you are not sure, call the most appropriate tool to make a best answer.")
            ])
            model = self.vlm
            chained_model = chat_prompt | model
            question  = f"User Task: {self.user_task}\n" \
                        f"History Summary: {self.history_summary}\n" \
                        f"Current Task: {self.current_task}\n"
                        
            response = chained_model.invoke({"question": question, "memory_records": memory_messages})

            parsed = eval(response.content)
            for key in keys_to_check_for:
                if key not in parsed:
                    raise ValueError("Missing required keys during generate. Retrying...")
            
            found_value = parsed["found_in_memory"]
            if found_value not in valid_found_values:
                raise ValueError(f"Invalid value for found_in_memory: '{found_value}'. Must be one of {valid_found_values}.")
            output = {
                "found_in_memory": found_value,
                "instance_desc": parsed["instance_desc"],
            }
            target_record = None
            if found_value == "yes":
                if self.memory_records and isinstance(self.memory_records[0], dict) and "instance_desc" in self.memory_records[0]:
                    # Grouped format
                    for group in self.memory_records:
                        for record in group.get("records", []):
                            if int(record["id"]) == int(parsed["record_id"]):
                                target_record = record
                                break
                        if target_record:
                            break
                else:
                    # Flat format
                    for record in self.memory_records:
                        if int(record["id"]) == int(parsed["record_id"]):
                            target_record = record
                            break
                if target_record is None:
                    raise ValueError(f"Could not find record with ID {parsed['record_id']} in memory records. Retrying...")

                image_path_fn = lambda vidpath, frame: os.path.join(vidpath, f"{frame:06d}.png")
                vidpath = target_record["vidpath"]
                start_frame = int(target_record["start_frame"])
                end_frame = int(target_record["end_frame"])
                frame = (start_frame + end_frame) // 2
                instance_viz_path = image_path_fn(vidpath, frame)
            else:
                instance_viz_path = None

            output["instance_viz_path"] = instance_viz_path
            output["past_observations"] = [target_record] if target_record else []
                
            if self.logger:
                self.logger.info(f"[DETERMINE_SEARCH_INSTANCE] generate() - Output keys: found_in_memory={output['found_in_memory']}, instance_desc={output['instance_desc']}, instance_viz_path={output['instance_viz_path']}")
                
            return {"messages": [response], "output": output}
        
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
            description="(Optional) Memory records retrieved earlier. Can be in one of two formats:\n"
                        "1. A flat list of memory records, where each item includes fields like caption, time, position, and possibly an image path.\n"
                        "2. A grouped list, where each item includes an 'instance_description' (a string) and 'records' (a list of memory records as above) "
                        "representing related observations for a specific instance."
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

def create_determine_unique_instances_tool(
    memory: MilvusMemory,
    llm,
    llm_raw,
    vlm,
    vlm_raw,
    logger=None
):
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
        
    class DetermineUniqueInstancesAgent:
        def __init__(self, memory, llm, llm_raw, vlm, vlm_raw, logger=None):
            self.memory = memory
            self.llm = llm
            self.llm_raw = llm_raw
            self.vlm = vlm
            self.vlm_raw = vlm_raw
            self.logger = logger
            
            self.setup_tools(memory)
            
            prompt_dir = str(os.path.dirname(__file__)) + '/../prompts/determine_unique_instances_tool/'
            self.decide_prompt = file_to_string(prompt_dir+'decide_prompt.txt')
            self.decide_gen_only_prompt = file_to_string(prompt_dir+'decide_gen_only_prompt.txt')
            self.generate_prompt = file_to_string(prompt_dir+'generate_prompt.txt')
            
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
                    for k, v in val.items():
                        image_paths[int(k)] = v
                except Exception:
                    continue

            memory_messages = []
            is_grouped = (
                isinstance(self.memory_records, list)
                and all(isinstance(x, dict) and "instance_desc" in x for x in self.memory_records)
            )

            if is_grouped:
                for i, group in enumerate(self.memory_records):
                    instance_desc = group.get("instance_desc", "(no description)")
                    records = group.get("records", [])
                    memory_messages.append({"type": "text", "text": f"--- Instance {i+1}: {instance_desc} ---"})
                    for record in records:
                        memory_messages += self._record_to_message(record, image_paths)
            else:
                for record in self.memory_records:
                    memory_messages += self._record_to_message(record, image_paths)

            return memory_messages

        def _record_to_message(self, record: Dict, image_paths: Dict[int, str]) -> List[Dict]:
            msgs = [{"type": "text", "text": parse_db_records_for_llm(record)}]
            if int(record["id"]) in image_paths:
                image_path = image_paths[int(record["id"])]
                img = PILImage.open(image_path).convert("RGB")
                img = img.resize((512, 512), PILImage.BILINEAR)
                
                draw = ImageDraw.Draw(img)
                text = f"record_id: {record['id']}"
                font = ImageFont.load_default()
                text_size = draw.textbbox((0, 0), text, font=font)
                padding = 4
                bg_rect = (
                    text_size[0] - padding,
                    text_size[1] - padding,
                    text_size[2] + padding,
                    text_size[3] + padding
                )
                draw.rectangle(bg_rect, fill=(0, 0, 0))
                draw.text((text_size[0], text_size[1]), text, fill=(255, 255, 255), font=font)

                buffer = BytesIO()
                img.save(buffer, format="PNG")
                img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                msgs.append(get_vlm_img_message(img_b64, type="gpt"))
            return msgs
        
        def decide(self, state: AgentState):
            messages = state["messages"]
            
            model = self.vlm
            if self.decide_call_count > 2:
                prompt = self.decide_gen_only_prompt
            else:
                prompt = self.decide_prompt
                model = model.bind_tools(self.tool_definitions)
                
            memory_messages = self._parse_memory_records(messages)
            chat_prompt = ChatPromptTemplate.from_messages([
                MessagesPlaceholder(variable_name="chat_history"),
                # ("human", "{memory_records}"),
                HumanMessage(content=memory_messages),
                ("system", prompt),
                ("human", "{question}"),
                ("system", "Remember to follow the json format strictly and only use the tools provided. Do not generate any text outside of tool calls.")
            ])
            chained_model = chat_prompt | model
            question  = f"User Task: {self.user_task}\n" \
                        f"History Summary: {self.history_summary}\n" \
                        f"Current Task: {self.current_task}\n"
            response = chained_model.invoke({
                "chat_history": messages[1:],
                "question": question, 
                # "memory_records": memory_messages
            })
            
            # from openai import OpenAI
            # client = OpenAI() 
            # messages = [
            #     {"role": "system", "content": prompt},  # text only
            #     {"role": "user", "content": memory_messages + [{"type": "text", "text": question}]},
            #     {"role": "system", "content": "Remember to follow the json format strictly and only use the tools provided. Do not generate any text outside of tool calls."}
            # ]
            # response = client.chat.completions.create(
            #     model="gpt-4o",
            #     messages=messages
            # )
            # import pdb; pdb.set_trace()
            # try:
            #     stripped_response = response.choices[0].message.content.strip()
            #     parsed = parse_json(stripped_response)
            #     parsed = parsed["tool_input"]["response"]
            # except Exception as e:
            #     raise ValueError(f"Failed to parse response: {e}. Response content: {response.choices[0].message.content}. Retrying...")
            
            if self.logger:
                if getattr(response, 'tool_calls'):
                    self.logger.info(f"[DETERMINE_UNIQUE_INSTANCES] decide() - Tool calls present: {response.tool_calls}")
                else:
                    self.logger.info(f"[DETERMINE_UNIQUE_INSTANCES] decide() - {response.content}.")
                
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call['name'] != "__conversational_response":
                        args = re.sub(r'^\{(.*)\}$', r'(\1)', str(tool_call['args'])) # remove curly braces
                        self.previous_tool_requests += f" {tool_call['name']} tool with the arguments: {args}.\n"
                        
            self.decide_call_count += 1
            return {"messages": [response]}
        
        def generate(self, state: AgentState):
            messages = state["messages"]
            keys_to_check_for = ["instance_desc", "record_ids"]
            
            prompt = self.generate_prompt
            # chat_prompt = ChatPromptTemplate.from_messages([
            #     # MessagesPlaceholder(variable_name="chat_history"),
            #     ("system", prompt),
            #     ("user", "{memory_records}"),
            #     ("user", "{question}"),
            #     ("system", "Remember to follow the json format strictly and only use the tools provided. Do not generate any text outside of tool calls. You must now list all distinct, plausible instances using rich, **visual-appearance-based** descriptions.")
            # ])
            # model = self.vlm
            # chained_model = chat_prompt | model
            question  = f"User Task: {self.user_task}\n" \
                        f"History Summary: {self.history_summary}\n" \
                        f"Current Task: {self.current_task}\n"
            
            memory_messages = self._parse_memory_records(messages)
            
            ###### Debug image messages #######
            # def dump_memory_messages(memory_messages, save_dir="debug/dumped_memory_messages"):
            #     os.makedirs(save_dir, exist_ok=True)
            #     msg_list = []
            #     for i, msg in enumerate(memory_messages):
            #         if msg["type"] == "text":
            #             with open(os.path.join(save_dir, f"msg_{i:03d}.txt"), "w") as f:
            #                 f.write(msg["text"])
            #             msg_list.append({"type": "text", "file": f"msg_{i:03d}.txt"})
            #         elif msg["type"] in ["image", "image_url"]:
            #             img_data = msg["image"] if "image" in msg else msg["image_url"]["url"]
            #             encoded = img_data.split("base64,")[-1]
            #             with open(os.path.join(save_dir, f"msg_{i:03d}.png"), "wb") as f:
            #                 f.write(base64.b64decode(encoded))
            #             msg_list.append({"type": "image", "file": f"msg_{i:03d}.png"})
            #     with open(os.path.join(save_dir, "messages.json"), "w") as f:
            #         json.dump(msg_list, f, indent=2)

            # dump_memory_messages(memory_messages)
            # import pdb; pdb.set_trace()
            
            # visualize_memory_messages(memory_messages)
            # import pdb; pdb.set_trace()
            
            # system_prompt = (
            #     "You are an expert instance summarizer.\n"
            #     "You are given a list of memory records consisting of textual descriptions and visual observations.\n"
            #     "Please caption all images you saw. You should give a concise image description, as well as the record id on the images. In the description, you should include the visual appearance of the object, such as color, shape, size, and any other relevant features.\n"
            # )
            # model = self.vlm_raw
            # chat_prompt = ChatPromptTemplate.from_messages([
            #     ("human", system_prompt),
            #     ("human", "{memory_records}"),
            # ])
            # chained_model = chat_prompt | model
            # response = chained_model.invoke({
            #     "memory_records": memory_messages})
            # import pdb; pdb.set_trace()
            ###### Debug image messages #######
            
            # response = chained_model.invoke({
            #     # "chat_history": messages[1:],
            #     "question": question, 
            #     "memory_records": memory_messages})
            
            from openai import OpenAI
            client = OpenAI() 
            messages = [
                {"role": "system", "content": prompt},  # text only
                {"role": "user", "content": memory_messages + [{"type": "text", "text": question}]},
                {"role": "system", "content": "Remember to follow the json format strictly and only use the tools provided. Do not generate any text outside of tool calls. You must now list all distinct, plausible instances using rich, **visual-appearance-based** descriptions."}
            ]
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            try:
                stripped_response = response.choices[0].message.content.strip()
                parsed = parse_json(stripped_response)
                parsed = parsed["tool_input"]["response"]
            except Exception as e:
                raise ValueError(f"Failed to parse response: {e}. Response content: {response.choices[0].message.content}. Retrying...")
                
            # Build a mapping from record ID to record for fast lookup
            ids_to_records = {int(record["id"]): record for record in self.memory_records}
            # parsed = eval(response.content)
            if type(parsed) is not list:
                raise ValueError("Expected a list of unique instances, but got something else. Retrying...")
            for item in parsed:
                for key in keys_to_check_for:
                    if key not in item:
                        raise ValueError("Missing required keys during generate. Retrying...")

                # Parse record_ids into a list of ints
                record_ids_str = item["record_ids"]
                if not isinstance(record_ids_str, str):
                    raise ValueError(f"record_ids must be a string, got {type(record_ids_str)}")
                try:
                    record_ids = [int(x.strip()) for x in record_ids_str.split(",") if x.strip().isdigit()]
                except Exception as e:
                    raise ValueError(f"Failed to parse record_ids '{record_ids_str}': {e}")
                # Check that the parsed list matches the expected format (at least one int, no junk)
                if not record_ids or ",".join(str(i) for i in record_ids) != ",".join(x.strip() for x in record_ids_str.split(",") if x.strip()):
                    raise ValueError(f"record_ids format is invalid: '{record_ids_str}'")
                # Check that all record_ids exist in ids_to_records
                for rid in record_ids:
                    if rid not in ids_to_records:
                        raise ValueError(f"record_id {rid} not found in memory_records. Retrying...")
                item["record_ids"] = record_ids
                
            output = []
            for item in parsed:
                output_item = {}
                output_item["instance_desc"] = item["instance_desc"]
                output_item["records"] = []
                for id in item["record_ids"]:
                    output_item["records"].append(ids_to_records[id])
                output.append(output_item)
                
            if self.logger:
                self.logger.info(f"[DETERMINE_UNIQUE_INSTANCES] generate() - Output contains {len(output)} unique instances.")
                for i, inst in enumerate(output):
                    self.logger.info(f"Instance {i+1}: desc={inst['instance_desc']}, records={len(inst['records'])}")
        
            # return {"messages": [response], "output": output}
            return {"output": output}
        
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
                    f"[DETERMINE_UNIQUE_INSTANCES] Running tool with user_task: {user_task}, "
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
                       f"Memory Records: {memory_records if memory_records else 'None'}\n" \
                       "Based on the information retrieved from your previous tool calls, could you list out all unique instances and their task-relevant observations?"
            inputs = { "messages": [
                    (("user", question))
                ]
            }
            state = self.graph.invoke(inputs)
            
            output = state.get("output", [])
            return output
        
    class DetermineUniqueInstancesInput(BaseModel):
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
        
    tool_runner = DetermineUniqueInstancesAgent(memory, llm, llm_raw, vlm, vlm_raw, logger)
    tool = StructuredTool.from_function(
        func=lambda user_task, history_summary, current_task, memory_records: 
            tool_runner.run(user_task, history_summary, current_task, memory_records),
        name="determine_unique_instances_from_working_memory",
        description="Determine unique instances from the memory records based on the high-level user task, current reasoning step, and optionally retrieved memory records. This would be used to identify distinct objects or events in the user's context.",
        args_schema=DetermineUniqueInstancesInput
    )
    return [tool]
    
def visualize_memory_messages(memory_messages, output_dir="debug/unique_instances"):
    os.makedirs(output_dir, exist_ok=True)

    i = 0
    current_text = "unknown"

    for msg in memory_messages:
        if msg["type"] == "text":
            # Save this for naming the image later
            current_text = msg["text"].strip().replace("\n", " ")[:100]  # truncate long text
        elif msg["type"] in ["image", "image_url"]:
            # Decode base64 image
            if msg["type"] == "image":
                img_data = msg["image"].split("base64,")[-1]
            else:
                img_data = msg["image_url"]["url"].split("base64,")[-1]

            try:
                img_bytes = base64.b64decode(img_data)
                img = PILImage.open(BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                print(f"[Warning] Failed to decode image at index {i}: {e}")
                continue

            # Use index + short text preview to name file
            filename = f"msg_{i:03d}.png"
            img_path = os.path.join(output_dir, filename)
            img.save(img_path)

            # Also save the text in a .txt file
            txt_path = os.path.join(output_dir, f"msg_{i:03d}.txt")
            with open(txt_path, "w") as f:
                f.write(current_text)

            i += 1