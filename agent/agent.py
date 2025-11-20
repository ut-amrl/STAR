import os
from langchain_openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage

from agent.utils.debug import get_logger
from agent.utils.function_wrapper import FunctionsWrapper
from agent.utils.tools import *

class Agent:
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        history: Annotated[Sequence[BaseMessage], add_messages]
        toolcalls: Annotated[Sequence, add_messages]
        next_state: str
        
    @staticmethod
    def from_agent_to(state: AgentState):
        next_state = state.get("next_state", "agent")
        return next_state
    
    @staticmethod
    def from_search_in_space_to(state: AgentState):
        next_state = state.get("next_state", "terminate")
        return next_state
    
    def __init__(self,
                 verbose: bool = False,
                 logdir: str = None,
                 logger_prefix: str = "",
                 is_interactive: bool = False,
                 robot_model: str = ""
    ):
        self.verbose = verbose
        self.is_interactive = is_interactive
        self.robot_model = robot_model

        self.logger = get_logger(logdir=logdir, prefix=logger_prefix, flatten=True) if logdir else get_logger(prefix=logger_prefix, flatten=True)
        
        self.task: Task = None
        
        # TODO: clean up model initialization
        self.vlm_raw = ChatOpenAI(model="o3", temperature=1, api_key=os.environ.get("OPENAI_API_KEY"))
        self.vlm = FunctionsWrapper(self.vlm_raw)
        self.vlm_raw_flex = ChatOpenAI(model="o3", temperature=1, api_key=os.environ.get("OPENAI_API_KEY"), service_tier="flex")
        self.vlm_flex = FunctionsWrapper(self.vlm_raw_flex)
        
        eval_prompt_dir = os.path.join(os.path.dirname(__file__), "prompts", "evaluation")
        self.eval_ref_and_ret_prompt = file_to_string(os.path.join(eval_prompt_dir, "eval_ref_and_ret_prompt.txt"))

        prompt_prefix = robot_model if robot_model == "" else f"{robot_model}_"
        search_in_space_prompt_dir = os.path.join(os.path.dirname(__file__), "prompts", "search_in_space")
        self.search_in_space_prompt = file_to_string(os.path.join(search_in_space_prompt_dir, f"{prompt_prefix}search_in_space_prompt.txt"))

        self.search_in_time_cnt = 0
        self.search_in_space_cnt = 0
        self.json_store = TempJsonStore()
        
        self.searched_poses = []
        self.searched_visible_instances = []
        
        self.exec_env = {} # TODO delete me
        
    def set_task(self, task_desc: str):
        self.task = Task(task_desc)
        if self.verbose:
            print(f"Task set: {task_desc}")
        
    def setup_tools(self, memory: MilvusMemory):
        if self.robot_model == "":
            robot_tools = create_physical_skills(self.json_store)
        elif self.robot_model == "tiago":
            robot_tools = create_tiago_physical_skills(self.json_store)
        else:
            raise ValueError(f"Unsupported robot model: {self.robot_model}")
        self.search_in_space_tools = robot_tools
        self.search_in_space_tool_definitions = [convert_to_openai_function(t) for t in self.search_in_space_tools]
        
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
        raise NotImplementedError("This method should be implemented in subclasses.")
    
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
        model_flex = self.vlm_flex
        model = model.bind_tools(self.eval_tool_definitions)
        model_flex = model_flex.bind_tools(self.eval_tool_definitions)
        
        chat_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("chat_history"),
            ("system", self.eval_ref_and_ret_prompt),
        ])
        chained_model = chat_prompt | model
        chained_flex_model = chat_prompt | model_flex

        input = {
            "chat_history": chat_history,
        }
        response = safe_gpt_invoke(chained_flex_model, chained_model, input)
        
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
        if self.robot_model == "tiago":
            max_search_in_space_cnt = 1
        elif self.is_interactive:
            max_search_in_space_cnt = 5
        else:
            max_search_in_space_cnt = 3    
        
        messages = state["messages"]
        
        # import pdb; pdb.set_trace()
        
        additional_search_history = []
        last_tool_calls = []
        
        last_message = messages[-1]
        if self.search_in_space_cnt > 0 and isinstance(last_message, ToolMessage):
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
                
            if last_ai_idx is not None:
                image_messages = []
                for msg in messages[last_ai_idx+1:]:
                    if isinstance(msg, ToolMessage):
                        original_msg_content = copy.copy(msg.content)
                        if isinstance(msg.content, str):
                            msg.content = parse_and_pretty_print_tool_message(msg.content)
                        additional_search_history.append(msg)
                        
                    if isinstance(original_msg_content, str) and has_file_id(original_msg_content):
                            file_id = get_file_id(original_msg_content)
                            content = get_file_id_messages(self.json_store, file_id)
                            message = HumanMessage(content=content)
                            if len(content) > 1:
                                image_messages.append(message)
                                
                additional_search_history += image_messages
                        
        if hasattr(last_tool_calls, "tool_calls") and last_tool_calls.tool_calls:
            for tool_call in last_tool_calls.tool_calls:
                if tool_call["name"] == "robot_detect":
                    normalized = last_message.content.replace("{{", "{").replace("}}", "}").replace("{{", "{").replace("}}", "}").replace("{{", "{").replace("}}", "}")
                    parsed = json.loads(normalized)
                    visible_instances = parsed.get("visible_instances", [])
                    if visible_instances:
                        self.searched_visible_instances.append(visible_instances)
                    instance_ids = parsed.get("instance_ids", [])
                    success = parsed.get("success", False)
                    
                    if not success or instance_ids is None or len(instance_ids) == 0:
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
                        if len(self.searched_visible_instances) > 0:
                            self.task.search_proposal.visible_instances = self.searched_visible_instances[-1]
                        return {
                            "history": additional_search_history,
                            "toolcalls": last_tool_calls,
                            "next_state": "end"
                        }
                        
                    break
                
                elif tool_call["name"] == "robot_navigate":
                    fn_args = tool_call.get("args", {})
                    position = fn_args.get("pos")
                    theta = fn_args.get("theta")
                    self.searched_poses.append((position, theta))
                    
                    normalized = last_message.content.replace("{{", "{").replace("}}", "}").replace("{{", "{").replace("}}", "}").replace("{{", "{").replace("}}", "}")
                    parsed = json.loads(normalized)
                    success = parsed.get("success", False)
                    if not success or max_search_in_space_cnt == 1:
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
                        if len(self.searched_visible_instances) > 0:
                            self.task.search_proposal.visible_instances = self.searched_visible_instances[-1]
                        return {
                            "history": additional_search_history,
                            "toolcalls": last_tool_calls,
                            "next_state": "end"
                        }
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
                    
                elif tool_call["name"] == "robot_open":
                    normalized = last_message.content.replace("{{", "{").replace("}}", "}").replace("{{", "{").replace("}}", "}").replace("{{", "{").replace("}}", "}")
                    parsed = json.loads(normalized)
                    success = parsed.get("success", False)
                    if not success:
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
                        if len(self.searched_visible_instances) > 0:
                            self.task.search_proposal.visible_instances = self.searched_visible_instances[-1]
                        return {
                            "history": additional_search_history,
                            "toolcalls": last_tool_calls,
                            "next_state": "end"
                        }
        
        next_state = state.get("next_state", "agent")
        if next_state == "end" or self.search_in_space_cnt >= max_search_in_space_cnt:
            return {
                "history": additional_search_history,
                "toolcalls": last_tool_calls,
                "next_state": "end",
            }
        
        chat_history = copy.deepcopy(state.get("history", []))
        chat_history += additional_search_history
        
        model = self.vlm
        model_flex = self.vlm_flex
        model = model.bind_tools(self.search_in_space_tool_definitions)
        model_flex = model_flex.bind_tools(self.search_in_space_tool_definitions)
        prompt = self.search_in_space_prompt
        
        chat_template = [
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
        chat_template += [
            ("system", f"You must strictly follow the JSON output format and do not provide any additional contexts."),
            ("system", f"🔄 You are allowed up to **{max_search_in_space_cnt} iterations** total. This is iteration **#{self.search_in_space_cnt}**.\nEach iteration consists of one full round of tool calls — even if you issue multiple tools in parallel, that still counts as one iteration."),
            ("system", f"You are allowed to search in real=world (navigate and search in space) up to **{max_search_in_space_cnt} iterations**. You have searched in space **{len(self.searched_poses)}** times so far: \n{searched_poses_str}"),
        ]
        chat_prompt = ChatPromptTemplate.from_messages(chat_template)
        
        chained_model = chat_prompt | model
        chained_flex_model = chat_prompt | model_flex
        question = f"User Task: {self.task.task_desc}\n" \
                   f"Have you figured out which instance is user referring to? Do you know where this instance was last seen? Do you know how to search this item in the real-world? What should you do next?"
        fact_prompt = f"Here are some facts for your context:\n" \
                      f"1. {self.memory.get_memory_stats_for_llm()}\n" \
                      f"2. You have been patrolling in a dynamic household or office environment, so objects you saw before may have been moved, or its status may be changed.\n" \
                      f"3. {self.memory.get_contiguous_record_groups_for_llm()}\n"
        
        input = {
            "chat_history": chat_history,
            "fact_prompt": fact_prompt,
            "question": question,
        }
        response = safe_gpt_invoke(chained_flex_model, chained_model, input)
        
        if self.logger:
            if hasattr(response, "tool_calls") and response.tool_calls:
                for call in response.tool_calls:
                    args_str = ", ".join(f"{k}={repr(v)}" for k, v in call.get("args", {}).items())
                    log_str = f"{call.get('name')}({args_str})"
                    self.logger.info(f"[SEARCH IN SPACE] Tool call: {log_str}")
            else:
                self.logger.info(f"[SEARCH IN SPACE] {response}")

        self.search_in_space_cnt += 1
        
        next_state = "agent"
        return {
            "messages": [response], 
            "history": additional_search_history + [response],
            "toolcalls": last_tool_calls,
            "next_state": next_state,
        }
        
    def build_graph(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def set_memory(self, memory: MilvusMemory):
        self.memory = memory
        self.setup_tools(memory)
        
    def run(self, question: str):
        raise NotImplementedError("This method should be implemented in subclasses.")