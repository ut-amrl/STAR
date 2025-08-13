import os
from langchain_openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage

from agent.utils.debug import get_logger
from agent.utils.function_wrapper import FunctionsWrapper
from agent.utils.tools import *

import roslib; roslib.load_manifest('amrl_msgs')
from amrl_msgs.srv import (
    GetImageSrvResponse,
    GetImageAtPoseSrvResponse, 
    PickObjectSrvResponse,
)

class ReplanLowLevelAgent:
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        history: Annotated[Sequence[BaseMessage], add_messages]
        toolcalls: Annotated[Sequence[BaseMessage], add_messages]
        next_state: str
        
    @staticmethod
    def from_agent_to(state: AgentState):
        next_state = state.get("next_state", "agent")
        return next_state
    
    def __init__(self, 
                 prompt_type: str,
                 verbose: bool = False,
                 navigate_fn: Callable[[List[float], float], GetImageAtPoseSrvResponse] = None,
                 find_object_fn: Callable[[str], List[List[int]]] = None,
                 pick_fn: Callable[[str], PickObjectSrvResponse] = None,
                 logdir: str = None,
                 logger_prefix: str = "",):
        self.prompt_type = prompt_type
        self.verbose = verbose
        
        self.navigate_fn = navigate_fn
        self.find_object_fn = find_object_fn
        self.pick_fn = pick_fn
        
        self.logger = get_logger(logdir=logdir, prefix=logger_prefix, flatten=True) if logdir else get_logger(prefix=logger_prefix, flatten=True)
        
        self.task: Task = None
        
        self.vlm_raw = ChatOpenAI(model="o3", temperature=1, api_key=os.environ.get("OPENAI_API_KEY"))
        self.vlm = FunctionsWrapper(self.vlm_raw)
        
        prompt_dir = os.path.join(os.path.dirname(__file__), "prompts", f"replan_{self.prompt_type}", "low_level_agent")
        self.agent_prompt = file_to_string(os.path.join(prompt_dir, "agent_prompt.txt"))
        self.agent_gen_only_prompt = file_to_string(os.path.join(prompt_dir, "agent_gen_only_prompt.txt"))
        self.agent_reflect_prompt = file_to_string(os.path.join(prompt_dir, "agent_reflect_prompt.txt"))
        
        eval_prompt_dir = os.path.join(os.path.dirname(__file__), "prompts", "evaluation")
        self.eval_ref_and_ret_prompt = file_to_string(os.path.join(eval_prompt_dir, "eval_ref_and_ret_prompt.txt"))
        
        self.search_in_time_cnt = 0
        self.json_store = TempJsonStore()
        
        self.searched_poses = []
        self.searched_visible_instances = []
        
        self.search_tools = None
        self.reflection_tools = None
        self.terminate_tools = None
        self.eval_tools = None
        self.search_tool_definitions = None
        self.reflection_tool_definitions = None
        self.terminate_tool_definitions = None
        self.eval_tool_definitions = None
        
    def set_task(self, task_desc: str):
        self.task = Task(task_desc)
        if self.verbose:
            print(f"Task set: {task_desc}")
            
    def setup_tools(self, memory: MilvusMemory):
        memory_search_tools = create_memory_search_tools(memory)
        inspect_tools = create_memory_inspection_tool(memory)
        terminate_tools = create_memory_terminate_tool()
        reflection_tools = create_pause_and_think_tool()
        robot_tools = create_physical_skills(self.json_store)
        
        self.search_tools = memory_search_tools + inspect_tools + robot_tools
        self.search_tool_definitions = [convert_to_openai_function(t) for t in self.search_tools]
        
        self.reflection_tools = reflection_tools
        self.reflection_tool_definitions = [convert_to_openai_function(t) for t in self.reflection_tools]
        self.terminate_tools = terminate_tools
        self.terminate_tool_definitions = [convert_to_openai_function(t) for t in self.terminate_tools]
        
        eval_tools = create_search_in_time_evaluation_tool()
        self.eval_tools = eval_tools
        self.eval_tool_definitions = [convert_to_openai_function(t) for t in self.eval_tools]
        
    def flush_tool_threads(self):
        """
        Wait until all background tool calls finish, then shut down the pool.
        Call once you’re done with this agent instance.
        """
        if hasattr(self, "_tool_pool") and self._tool_pool is not None:
            self._tool_pool.shutdown(wait=True)
            self._tool_pool = None
            
        def _close_logger(logger):
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
        _close_logger(self.logger)
        
    def agent(self, state: AgentState):
        max_search_in_time_cnt = 20
        n_reflection_intervals = 5
        max_search_in_space_cnt = 3
        
        messages = state["messages"]
        
        additional_search_history = []
        last_tool_calls = []
        
        last_message = messages[-1]
        if isinstance(last_message, ToolMessage):
            # ===  Step 1: Find last AIMessage with tool_calls
            idx = len(messages) - 1
            last_ai_idx = None
            while idx >= 0:
                msg = messages[idx]
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                    last_tool_calls = copy.deepcopy(msg)
                    last_ai_idx = idx
                    break
                idx -= 1

            # ===  Step 2: Append all following ToolMessages into `history`
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
                        elif isinstance(original_msg_content, str) and has_file_id(original_msg_content):
                            file_id = get_file_id(original_msg_content)
                            content = get_file_id_messages(self.json_store, file_id)
                            message = HumanMessage(content=content)
                            if len(content) > 1:
                                image_messages.append(message)
                                
                        if self.logger:
                            self.logger.info(f"[SEARCH] Tool Response: {msg.content}")
                
                additional_search_history += image_messages
                
        if hasattr(last_tool_calls, "tool_calls") and last_tool_calls.tool_calls:
            for tool_call in last_tool_calls.tool_calls:
                
                if tool_call["name"] == "robot_detect":
                    normalized = last_message.content.replace("{{", "{").replace("}}", "}").replace("{{", "{").replace("}}", "}").replace("{{", "{").replace("}}", "}")
                    parsed = json.loads(normalized)
                    visible_instances = parsed.get("visible_instances", [])
                    if visible_instances:
                        self.searched_visible_instances.append(visible_instances)
                    break
                
                elif tool_call["name"] == "robot_navigate":
                    fn_args = tool_call.get("args", {})
                    position = fn_args.get("pos")
                    theta = fn_args.get("theta")
                    self.searched_poses.append((position, theta))
                    if len(self.searched_poses) > max_search_in_space_cnt:
                        self.search_in_time_cnt = max_search_in_time_cnt + 1 # Force to end search in time
                    break
                
                elif tool_call["name"] == "robot_pick":
                    pos = [-1, -1, -1]  # Default position if no search proposal
                    theta = -1
                    if len(self.searched_poses) > 0:
                        pos = self.searched_poses[-1][0]
                        theta = self.searched_poses[-1][1]
                    self.task.search_proposal = SearchProposal(
                        summary="",
                        instance_description="",
                        position=pos,
                        theta=theta,
                        records=[]
                    )
                    fn_args = tool_call["args"]
                    normalized = last_message.content.replace("{{", "{").replace("}}", "}").replace("{{", "{").replace("}}", "}").replace("{{", "{").replace("}}", "}")
                    parsed = json.loads(normalized)
                    self.task.search_proposal.has_picked = parsed.get("success", False)
                    self.task.search_proposal.instance_name = parsed.get("instance_uid", "")
                    if len(self.searched_visible_instances) > 0:
                        self.task.search_proposal.visible_instances = self.searched_visible_instances[-1]
                    return {
                        "history": additional_search_history,
                        "toolcalls": last_tool_calls,
                        "next_state": "end"
                    }
                
        next_state = state.get("next_state", "agent")
        if next_state == "end":
            return {
                "history": additional_search_history,
                "toolcalls": last_tool_calls,
                "next_state": "end",
            }
                
        chat_history = copy.deepcopy(state.get("history", []))
        chat_history += additional_search_history
        
        model = self.vlm
        if self.search_in_time_cnt < max_search_in_time_cnt:
            if self.search_in_time_cnt % n_reflection_intervals == 0:
                current_tool_defs = self.reflection_tool_definitions
            else:
                current_tool_defs = self.search_tool_definitions
        else:
            current_tool_defs = self.terminate_tool_definitions
            
        model = model.bind_tools(current_tool_defs)
        tool_names = [tool['name'] for tool in current_tool_defs]
        tool_list_str = "\n".join([f"{i+1}. {name}" for i, name in enumerate(tool_names)])
        
        # Select prompt template
        if self.search_in_time_cnt < max_search_in_time_cnt:
            if self.search_in_time_cnt % n_reflection_intervals == 0:
                prompt = self.agent_reflect_prompt
            else:
                prompt = self.agent_prompt
        else:
            prompt = self.agent_gen_only_prompt
            
        chat_template = [
            ("human", f"User has asked you to fulfill this task: {self.task.task_desc}. You are a memory-capable robot assistant. Your goal is to **help the user retrieve a physical object in the real world** by reasoning over **past observations stored in memory**. Right now, you need to decide what to do next based on the chat history of the tools you called previously as well as tool responses. "),
            ("human", "This is previous tool calls and the responses:"),
            MessagesPlaceholder("chat_history"),
            ("system", prompt),
            ("system", "{fact_prompt}"),
            ("human", "{question}"),
            ("system", "You should now decide the **next action** based on everything you've seen so far. Use the available tools to continue your memory search, or finalize your decision if you're confident. Reason carefully about what you have done, what you have known, what your current subgoal is, and what user's task is to decide what to do next."),
        ]
        
        if len(self.searched_poses) > 0:
            searched_poses_str = "\n".join([f"{i+1}. Position: {pos[0]}, Theta: {pos[1]}" for i, pos in enumerate(self.searched_poses)])
        else:
            searched_poses_str = "None."
        if self.search_in_time_cnt < max_search_in_time_cnt:
            chat_template += [
                ("system", f"You must strictly follow the JSON output format and do not provide any additional contexts. As a reminder, these are available tools: \n{tool_list_str}"),
                ("system", f"🔄 You are allowed up to **{max_search_in_time_cnt} iterations** total. This is iteration **#{self.search_in_time_cnt}**.\nEach iteration consists of one full round of tool calls — even if you issue multiple tools in parallel, that still counts as one iteration."),
                ("system", f"You are allowed to search in real=world (navigate and search in space) up to **{max_search_in_space_cnt} iterations**. You have searched in space **{len(self.searched_poses)}** times so far: \n{searched_poses_str}"),
            ]
        else:
            chat_template += [
                ("system", f"You must strictly follow the JSON output format. Since you have already reached the maximum number of iterations, you should finalize your decision now by calling `terminate` tool with right JSON format."),
            ]
        chat_prompt = ChatPromptTemplate.from_messages(chat_template)
        
        chained_model = chat_prompt | model
        question = f"User Task: {self.task.task_desc}\n" \
                   f"Have you figured out which instance is user referring to? Do you know where this instance was last seen? Do you know how to search this item in the real-world? What should you do next?"
        fact_prompt = f"Here are some facts for your context:\n" \
                      f"1. {self.memory.get_memory_stats_for_llm()}\n" \
                      f"2. You have been patrolling in a dynamic household or office environment, so objects you saw before may have been moved, or its status may be changed.\n" \
                      f"3. {self.memory.get_contiguous_record_groups_for_llm()}\n"
        
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
                    self.logger.info(f"[SEARCH] Tool call: {log_str}")
            else:
                self.logger.info(f"[SEARCH] {response}")

        self.search_in_time_cnt += 1
        
        next_state = "agent"
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_call_names = [call.get("name") for call in response.tool_calls]
            if "terminate" in tool_call_names:
                next_state = "terminate"
            elif "pause_and_think" in tool_call_names:
                next_state = "reflection"
                
            if "robot_navigate" in tool_call_names:
                for tool_call in response.tool_calls:
                    if tool_call.get("name") == "robot_navigate":
                        fn_args = tool_call.get("args", {})
                        position = fn_args.get("pos")
                        theta = fn_args.get("theta")
                        self.searched_poses.append((position, theta))
                        break
                
                if len(self.searched_poses) > max_search_in_space_cnt:
                    next_state = "terminate"
                    if self.logger:
                        self.logger.warning(f"[SEARCH] Reached maximum search in space iterations ({max_search_in_space_cnt}). Ending search.")
        
        return {
            "messages": [response], 
            "history": additional_search_history + [response],
            "toolcalls": last_tool_calls,
            "next_state": next_state,
        }
    
    def prepare_search_in_space(self, state: AgentState):
        messages = state["messages"]
        last_message = messages[-2] if messages else None
        
        # Check if response contains a 'terminate' tool call
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for call in last_message.tool_calls:
                if call.get("name") == "terminate":
                    fn_args = call.get("args", {})
                    summary = fn_args["summary"]
                    instance_description = fn_args.get("instance_description", "")
                    position = fn_args["position"]
                    theta = fn_args["theta"]
                    
                    self.searched_poses.append((position, theta))
                    
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
        
    def evaluate(self, state: AgentState):
        
        if type(state.get("history", [])[-1]) == ToolMessage:
            chat_history = copy.deepcopy(state.get("history", []))
        else:
            chat_history = copy.deepcopy(state.get("history", []))[:-1]
        
        if self.task.search_proposal is None:
            pos = [-1, -1, -1]  # Default position if no search proposal
            theta = -1
            if len(self.searched_poses) > 0:
                pos = self.searched_poses[-1][0]
                theta = self.searched_poses[-1][1]
            self.task.search_proposal = SearchProposal(
                summary="",
                instance_description="",
                position=pos,
                theta=theta,
                records=[]
            )
        
        model = self.vlm
        model = model.bind_tools(self.eval_tool_definitions)
        
        chat_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("chat_history"),
            ("system", self.eval_ref_and_ret_prompt),
        ])
        chained_model = chat_prompt | model
        
        try:
            response = chained_model.invoke({
                "chat_history": chat_history,
            })
        except Exception as e:
            import pdb; pdb.set_trace()
        
        if hasattr(response, "tool_calls") and response.tool_calls:
            for call in response.tool_calls:
                if call.get("name") == "review_object_reference_and_retrieval_terminate":
                    
                    fn_args = call.get("args", {})
                    review_rationale = fn_args["review_rationale"]
                    reference_resolution_record_id = int(fn_args["reference_resolution_record_id"])
                    retrieval_grounding_record_id = int(fn_args["retrieval_grounding_record_id"])
                    
                    if reference_resolution_record_id == -1:
                        reference_resolution_record = None
                    else:
                        records = eval(self.memory.get_by_id(reference_resolution_record_id))
                        if len(records) < 1:
                            reference_resolution_record = None
                        else:
                            reference_resolution_record = records[0]
                    
                    if retrieval_grounding_record_id == -1:
                        retrieval_grounding_record = None
                    else:
                        records = eval(self.memory.get_by_id(retrieval_grounding_record_id))
                        if len(records) < 1:
                            retrieval_grounding_record = None
                        else:
                            retrieval_grounding_record = records[0]
                    
                    if self.logger:
                        self.logger.info(f"[EVALUATE SEARCH] Reference resolution record: {reference_resolution_record}")
                        self.logger.info(f"[EVALUATE SEARCH] Retrieval grounding record: {retrieval_grounding_record}")
                        
                    self.task.search_proposal.reference_resolution_records = reference_resolution_record
                    self.task.search_proposal.retrieval_grounding_records = retrieval_grounding_record
                    
                    return
                
        raise ValueError("No terminate tool call found in the response")
    
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
            return
        if self.logger:
            self.logger.info(f"[SEARCH IN SPACE] Navigation successful to position {self.task.search_proposal.position} with theta {self.task.search_proposal.theta}.")
        
        find_response = self.find_object_fn(
            self.task.search_proposal.instance_description,
            self.task.search_proposal.get_viz_path()
        ) 
        if not find_response:
            if self.logger:
                self.logger.error(f"[SEARCH IN SPACE] Object not found in the current view.")
            return
        if self.logger:
            self.logger.info(f"[SEARCH IN SPACE] Object found in the current view: {find_response}.")
        self.task.search_proposal.visible_instances = find_response.visible_instances
            
        pick_response = self.pick_fn(
            # self.task.search_proposal.instance_description,
            self.class_type,  # TODO
            find_response.id
        )
        if not pick_response.success:
            if self.logger:
                self.logger.error(f"[SEARCH IN SPACE] Pick operation failed!")
        
        self.task.search_proposal.instance_name = pick_response.instance_uid
        self.task.search_proposal.has_picked = pick_response.success
        if self.logger:
            self.logger.info(f"[SEARCH IN SPACE] Pick operation successful: {self.task.search_proposal.instance_name} (has_picked={self.task.search_proposal.has_picked}).")
        return
    
    def build_graph(self):
        workfllow = StateGraph(ReplanLowLevelAgent.AgentState)
        
        workfllow.add_node("agent", lambda state: try_except_continue(state, self.agent))
        workfllow.add_node("search_action", ToolNode(self.search_tools))
        workfllow.add_node("reflection_action", ToolNode(self.reflection_tools))
        workfllow.add_node("terminate_action", ToolNode(self.terminate_tools))
        workfllow.add_node("prepare_search_in_space", lambda state: try_except_continue(state, self.prepare_search_in_space))
        workfllow.add_node("search_in_space", lambda state: try_except_continue(state, self.search_in_space))
        workfllow.add_node("evaluate", lambda state: try_except_continue(state, self.evaluate))
        
        workfllow.add_conditional_edges(
            "agent",
            ReplanLowLevelAgent.from_agent_to,
            {   
                "agent": "search_action",
                "reflection": "reflection_action",
                "terminate": "terminate_action",
                "end": "evaluate",
            }
        )
        workfllow.add_edge("search_action", "agent")
        workfllow.add_edge("reflection_action", "agent")
        workfllow.add_edge("terminate_action", "prepare_search_in_space")
        workfllow.add_edge("prepare_search_in_space", "search_in_space")
        workfllow.add_edge("search_in_space", "evaluate")
        workfllow.add_edge("evaluate", END)
        
        workfllow.set_entry_point("agent")
        self.graph = workfllow.compile()
        
    def set_memory(self, memory: MilvusMemory):
        self.memory = memory
        self.setup_tools(memory)
        
    def run(self, question: str, class_type: str):
        
        self.class_type = class_type
        
        if self.logger:
            self.logger.info("=============== START ===============")
            self.logger.info(f"User question: {question}.")
            
        self.set_task(question)
        
        self.search_in_time_cnt = 0
        self.searched_poses = []
        self.searched_visible_instances = []
        self.task.search_proposal = None
        
        self.build_graph()
        
        inputs = { "messages": [
                (("user", self.task.task_desc)),
            ]
        }
        
        config = {"recursion_limit": 60}
        state = self.graph.invoke(inputs, config=config)
        
        if self.logger:
            self.logger.info("=============== END =============== \n\n\n")
            
        toolcalls = []
        for msg in state.get("toolcalls", []):
            toolcalls += msg.tool_calls
        self.task.search_proposal.searched_poses = self.searched_poses
        return {
            "task_result": self.task.search_proposal,
            "toolcalls": toolcalls,
        }