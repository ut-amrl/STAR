import os
from langchain_openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage

from agent.utils.debug import get_logger
from agent.utils.function_wrapper import FunctionsWrapper # TODO need to clean up FunctionsWrapper
from agent.utils.tools2 import *

import roslib; roslib.load_manifest('amrl_msgs')
from amrl_msgs.srv import (
    GetImageSrvResponse,
    GetImageAtPoseSrvResponse, 
    PickObjectSrvResponse,
    GetVisibleObjectsSrvResponse,
    SemanticObjectDetectionSrvResponse
)

class Agent:
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        search_in_time_history: Annotated[Sequence[BaseMessage], add_messages]
        
    @staticmethod
    def from_search_in_time_to(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1] if messages else None
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for call in last_message.tool_calls:
                if call.get("name") == "terminate":
                    return "next"
        return "search_in_time_action"
    
    def __init__(self,
                 verbose: bool = False,
                 navigate_fn: Callable[[List[float], float], GetImageAtPoseSrvResponse] = None,
                 find_object_fn: Callable[[str], List[List[int]]] = None,
                 observe_fn: Callable[[], GetImageSrvResponse] = None,
                 pick_fn: Callable[[str], PickObjectSrvResponse] = None
    ):
        self.verbose = verbose
        
        self.navigate_fn = navigate_fn
        self.find_object_fn = find_object_fn
        self.observe_fn = observe_fn
        self.pick_fn = pick_fn
        
        self.logger = get_logger()
        
        self.task: Task = None
        
        self.llm_raw = ChatOpenAI(model="o3", temperature=1, api_key=os.environ.get("OPENAI_API_KEY"))
        self.llm = FunctionsWrapper(self.llm_raw)
        self.vlm_raw = ChatOpenAI(model="o3", temperature=1, api_key=os.environ.get("OPENAI_API_KEY"))
        self.vlm = FunctionsWrapper(self.vlm_raw)
        
        prompt_dir = os.path.join(os.path.dirname(__file__), "prompts", "agent3")
        self.search_in_time_prompt = file_to_string(os.path.join(prompt_dir, "search_in_time_prompt.txt"))
        
        self.temporal_tools, self.spatial_tools = None, None
        self.temporal_tool_definitions = None
        self.spatial_tool_definitions = None
        
    def set_task(self, task_desc: str):
        self.task = Task(task_desc)
        if self.verbose:
            print(f"Task set: {task_desc}")
            
    def setup_tools(self, memory: MilvusMemory):
        search_tools = create_memory_search_tools(memory)
        inspect_tools = create_memory_inspection_tool(memory)
        response_tools = create_memory_terminate_tool()
        self.temporal_tools = search_tools + inspect_tools + response_tools
        self.temporal_tool_definitions = [convert_to_openai_function(t) for t in self.temporal_tools]
            
    def search_in_time(self, state: AgentState):
        messages = state["messages"]
        
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
            for msg in messages[last_ai_idx+1:]:
                if isinstance(msg, ToolMessage):
                    if isinstance(msg.content, str):
                        msg.content = parse_and_pretty_print_tool_message(msg.content)
                    state["search_in_time_history"].append(msg)
                    
                    if isinstance(msg.content, str) and is_image_inspection_result(msg.content):
                        inspection = eval(msg.content)
                        for id, path in inspection.items():
                            content = get_image_message_for_record(id, path)
                            message = HumanMessage(content=content)
                            state["search_in_time_history"].append(message)
                    
                    if self.logger:
                        self.logger.info(f"[SEARCH IN TIME] Tool Response: {msg.content}")
        
        chat_history = state.get("search_in_time_history", [])
        
        model = self.vlm
        model = model.bind_tools(self.temporal_tool_definitions)
        
        # Extract tool names from self.temporal_tool_definitions
        tool_names = [tool['name'] for tool in self.temporal_tool_definitions]
        tool_list_str = "\n".join([f"{i+1}. {name}" for i, name in enumerate(tool_names)])

        chat_prompt = ChatPromptTemplate.from_messages([
            ("human", f"User has asked you to fulfill this task: {self.task.task_desc}. You are a memory-capable robot assistant. Your goal is to **help the user retrieve a physical object in the real world** by reasoning over **past observations stored in memory**. Right now, you need to decide what to do next based on the chat history of the tools you called previously as well as tool responses. "),
            ("human", "This is previous tool calls and the responses:"),
            MessagesPlaceholder("chat_history"),
            ("human", self.search_in_time_prompt),
            ("system", "{fact_prompt}"),
            ("human", "{question}"),
            ("system", "Please determine the next action to take! Remember you can only call the provided tools, and stick strictly to the JSON format. Reason carefully about memory search instance, world search instance, and tool call history before making a decision. If you want to pause a second to think about what to do, use `__conversational_reponse` to draft your reply."),
            ("system", f"As a reminder, you can only call the following tools: \n{tool_list_str}"),
        ])
        chained_model = chat_prompt | model
        question = f"User Task: {self.task.task_desc}\n" \
                   f"What should you do next?"
        fact_prompt = f"Here are some facts for your context:\n" \
                      f"{self.memory.get_memory_stats_for_llm()}"
                      
        response = chained_model.invoke({
            "chat_history": chat_history,
            "fact_prompt": fact_prompt,
            "question": question,
        })

        if self.logger:
            if hasattr(response, "tool_calls") and response.tool_calls:
                for call in response.tool_calls:
                    args_str = ", ".join(f"{k}={repr(v)}" for k, v in call.get("args", {}).items())
                    log_str = f"{call.get('name')}({args_str})"
                    self.logger.info(f"[SEARCH IN TIME] Tool call: {log_str}")
            else:
                self.logger.info(f"[SEARCH IN TIME] {response}")

        return {"messages": [response], "search_in_time_history": [response]}
    
    def prepare_search_in_space(self, state: AgentState):
        messages = state["messages"]
        last_message = messages[-1] if messages else None
        
        # Check if response contains a 'terminate' tool call
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for call in last_message.tool_calls:
                if call.get("name") == "terminate":
                    fn_args = call.get("args", {})
                    summary = fn_args["summary"]
                    instance_description = fn_args.get("instance_description", "")
                    position = fn_args["position"]
                    theta = fn_args["theta"]
                    
                    records = []
                    record_ids = [int(x) for x in fn_args["record_ids"]]
                    for record_id in record_ids:
                        record = eval(self.memory.get_by_id(record_id))[0]
                        record["position"] = eval(record["position"])
                        records.append(record)
                    
                    self.task.search_proposal = SearchProposal(summary=summary,
                                                          instance_description=instance_description,
                                                          position=position,
                                                          theta=theta,
                                                          records=records,)
                    if self.logger:
                        self.logger.info(f"[PREPARE SEARCH IN SPACE] Search proposal prepared: {self.task.search_proposal}")
                    return
                
        # TODO: Should never go here; add fallback logic
        import pdb; pdb.set_trace()
    
    def search_in_space(self, state: AgentState):
        if not self.task.search_proposal:
            # TODO: Handle the case where search proposal is not set
            if self.logger:
                self.logger.warning("[SEARCH IN SPACE] No search proposal set. Cannot proceed with search in space.")
            return
        
        nav_response = self.navigate_fn(
            self.task.search_proposal.position,
            self.task.search_proposal.theta
        )
        if not nav_response.success:
            if self.logger:
                self.logger.error(f"[SEARCH IN SPACE] Navigation failed!")
            import pdb; pdb.set_trace() # NOTE: This should not happen
            return
        if self.logger:
            self.logger.info(f"[SEARCH IN SPACE] Navigation successful to position {self.task.search_proposal.position} with theta {self.task.search_proposal.theta}.")
        
        find_response = self.find_object_fn(self.task.search_proposal.instance_description) 
        if not find_response:
            if self.logger:
                self.logger.error(f"[SEARCH IN SPACE] Object not found in the current view.")
            return
        if self.logger:
            self.logger.info(f"[SEARCH IN SPACE] Object found in the current view: {find_response}.")
            
        pick_response = self.pick_fn(self.task.search_proposal.instance_description)
        if not pick_response.success:
            if self.logger:
                self.logger.error(f"[SEARCH IN SPACE] Pick operation failed!")
        
        self.task.search_proposal.instance_name = pick_response.instance_uid
        self.task.search_proposal.has_picked = pick_response.success
        if self.logger:
            self.logger.info(f"[SEARCH IN SPACE] Pick operation successful: {self.task.search_proposal.instance_name} (has_picked={self.task.search_proposal.has_picked}).")
        return
            
    def build_graph(self):
        """
        Build the graph for the agent.
        """
        workflow = StateGraph(Agent.AgentState)
        
        workflow.add_node("search_in_time", lambda state: try_except_continue(state, self.search_in_time))
        workflow.add_node("search_in_time_action", ToolNode(self.temporal_tools))
        workflow.add_node("prepare_search_in_space", lambda state: try_except_continue(state, self.prepare_search_in_space))
        workflow.add_node("search_in_space", lambda state: try_except_continue(state, self.search_in_space))
        
        workflow.add_edge("search_in_time_action", "search_in_time")
        workflow.add_conditional_edges(
            "search_in_time",
            Agent.from_search_in_time_to,
            {
                "search_in_time_action": "search_in_time_action",
                "next": "prepare_search_in_space",
            }
        )
        workflow.add_edge("prepare_search_in_space", "search_in_space")
        # TODO
        workflow.add_edge("search_in_space", END)
        
        workflow.set_entry_point("search_in_time")
        self.graph = workflow.compile()
        
    def set_memory(self, memory: MilvusMemory):
        self.memory = memory
        self.setup_tools(memory)
        
    def run(self, question: str, today: str, graph_type: str):
        if self.logger:
            self.logger.info(f"User question: {question}. Today's date is: {today}.")
        
        self.set_task(question)
        self.today_str = today
        
        self.build_graph()
        
        inputs = { "messages": [
                (("user", self.task.task_desc)),
            ]
        }
        state = self.graph.invoke(inputs)
        
        if self.logger:
            self.logger.info("")
        
        return self.task.search_proposal
        