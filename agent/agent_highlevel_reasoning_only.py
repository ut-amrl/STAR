import os
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage

from agent.utils.tools import *
from agent.agent import Agent

class ReasoningHighLevelAgent(Agent):
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        history: Annotated[Sequence[BaseMessage], add_messages]
        toolcalls: Annotated[Sequence, add_messages]
        next_state: str
        
    def __init__(self,
                 prompt_type: str = "gt",  
                 verbose: bool = False,
                 logdir: str = None,
                 logger_prefix: str = "",
                 is_interactive: bool = False
    ):
        super().__init__(verbose, logdir, logger_prefix, is_interactive)

        prompt_dir = os.path.join(os.path.dirname(__file__), "prompts", prompt_type, "high_level_agent")
        self.agent_prompt = file_to_string(os.path.join(prompt_dir, "search_in_time_prompt.txt"))
        self.agent_gen_only_prompt = file_to_string(os.path.join(prompt_dir, "search_in_time_gen_only_prompt.txt"))
        self.agent_reflect_prompt = file_to_string(os.path.join(prompt_dir, "search_in_time_reflection_prompt.txt"))
        
    def set_task(self, task_desc: str):
        return super().set_task(task_desc)
            
    def setup_tools(self, memory: MilvusMemory):
        eval_tools = create_search_in_time_evaluation_tool()
        self.eval_tools = eval_tools
        self.eval_tool_definitions = [convert_to_openai_function(t) for t in self.eval_tools]

        recall_best_matches_tool = create_recall_best_matches_tool(memory, self.vlm_flex, self.vlm, logger=self.logger if self.logger else None)
        recall_last_seen_tool = create_recall_last_seen_tool(memory, self.vlm_flex, self.vlm, logger=self.logger if self.logger else None)
        recall_all_tools = create_recall_all_tool(memory, self.vlm_flex, self.vlm, logger=self.logger if self.logger else None)

        search_tools = recall_best_matches_tool + recall_last_seen_tool + recall_all_tools
        inspect_tools = create_memory_inspection_tool(memory)
        response_tools = create_memory_terminate_tool(memory)
        reflection_tools = create_pause_and_think_tool()
        
        self.temporal_tools = search_tools + inspect_tools + response_tools
        self.temporal_tool_definitions = [convert_to_openai_function(t) for t in self.temporal_tools]

        self.reflection_tools = reflection_tools
        self.reflection_tool_definitions = [convert_to_openai_function(t) for t in self.reflection_tools]
        self.temporal_search_terminate_tool = response_tools
        self.temporal_search_terminate_tool_definitions = [convert_to_openai_function(t) for t in self.temporal_search_terminate_tool]
        
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
        max_search_in_time_cnt = 10
        n_reflection_intervals = 5
        
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
                             
                        elif isinstance(original_msg_content, str) and is_recall_tool_result(original_msg_content):
                            records = eval(original_msg_content).get("records", [])
                            if is_recall_all_result(original_msg_content):
                                records = sorted(records, key=lambda d: float(d["timestamp"]))
                            for record in records:
                                content = get_image_message_for_record(
                                    int(record["id"]), 
                                    get_viz_path(self.memory, int(record["id"])),
                                    msg.tool_call_id)
                                message = HumanMessage(content=content)
                                image_messages.append(message)
                                
                        if self.logger:
                            self.logger.info(f"[SEARCH] Tool Response: {msg.content}")
                
                additional_search_history += image_messages
                
        if hasattr(last_tool_calls, "tool_calls") and last_tool_calls.tool_calls:
            for tool_call in last_tool_calls.tool_calls:
                
                if tool_call["name"] == "terminate":
                    fn_args = tool_call["args"]
                    summary = fn_args["summary"]
                    isinstance_description = fn_args.get("instance_description", "")
                    position = fn_args["position"]
                    theta = fn_args["theta"]
                    
                    if self.logger:
                        self.logger.info(f"[SEARCH] Search proposal summary: {summary}. instance descritption = {isinstance_description}, pos = {position}, theta = {theta}")

                    return {
                        "history": additional_search_history,
                        "toolcalls": last_tool_calls,
                        "next_state": "search_in_space",
                    }
                    
        chat_history = copy.deepcopy(state.get("history", []))
        chat_history += additional_search_history
        
        model = self.vlm
        model_flex = self.vlm_flex
        if self.search_in_time_cnt < max_search_in_time_cnt:
            if self.search_in_time_cnt % n_reflection_intervals == 0:
                current_tool_defs = self.reflection_tool_definitions
            else:
                current_tool_defs = self.temporal_tool_definitions
        else:
            current_tool_defs = self.temporal_search_terminate_tool_definitions

        model = model.bind_tools(current_tool_defs)
        model_flex = model_flex.bind_tools(current_tool_defs)
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
        if self.search_in_time_cnt < max_search_in_time_cnt:
            chat_template += [
                ("system", f"You must strictly follow the JSON output format. As a reminder, these are available tools: \n{tool_list_str}"),
                ("system", f"🔄 You are allowed up to **{max_search_in_time_cnt} iterations** total. This is iteration **#{self.search_in_time_cnt}**.\nEach iteration consists of one full round of tool calls — even if you issue multiple tools in parallel, that still counts as one iteration.")
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
        return {
            "messages": [response], 
            "history": additional_search_history + [response],
            "toolcalls": last_tool_calls,
            "next_state": next_state,
        }
    
    def build_graph(self):
        """
        Build the graph for the agent.
        """
        workflow = StateGraph(Agent.AgentState)
        
        workflow.add_node("search_in_time", lambda state: try_except_continue(state, self.agent))
        workflow.add_node("search_in_time_action", ToolNode(self.temporal_tools))
        workflow.add_node("search_in_time_reflection_action", ToolNode(self.reflection_tools))
        workflow.add_node("search_in_time_terminate_action", ToolNode(self.temporal_search_terminate_tool))
        workflow.add_node("evaluate", lambda state: try_except_continue(state, self.evaluate))
        
        workflow.add_conditional_edges(
            "search_in_time",
            Agent.from_agent_to,
            {
                "agent": "search_in_time_action",
                "reflection": "search_in_time_reflection_action",
                "terminate": "search_in_time_terminate_action",
                "search_in_space": "evaluate",
            }
        )
        workflow.add_edge("search_in_time_action", "search_in_time")
        workflow.add_edge("search_in_time_reflection_action", "search_in_time")
        workflow.add_edge("search_in_time_terminate_action", "search_in_time")
        workflow.add_edge("evaluate", END)
        
        workflow.set_entry_point("search_in_time")
        self.graph = workflow.compile()
        
    def set_memory(self, memory: MilvusMemory):
        self.memory = memory
        self.setup_tools(memory)
        
    def run(self, question: str, eval: bool = False, class_type: str = None):
        self.class_type = class_type  # TODO delete this after testing
        if self.logger:
            self.logger.info("=============== START ===============")
            self.logger.info(f"User question: {question}.")
        
        self.set_task(question)
        
        self.search_in_time_cnt = 0
        self.search_in_space_cnt = 0
        self.searched_poses = []
        self.searched_visible_instances = []
        self.task.search_proposal = None
        
        self.build_graph()
        
        inputs = { "messages": [
                (("user", self.task.task_desc)),
            ]
        }

        config = {"recursion_limit": 100}
        state = self.graph.invoke(inputs, config=config)
        
        if self.logger:
            self.logger.info("=============== END =============== \n\n\n")
        
        toolcalls = []
        for msg in state.get("toolcalls", []):
            toolcalls += msg.tool_calls
        return {
            "task_result": self.task.search_proposal,
            "toolcalls": toolcalls,
        }