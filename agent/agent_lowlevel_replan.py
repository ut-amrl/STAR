import os
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage

from agent.utils.tools import *
from agent.utils.utils import is_task_terminate_result
from agent.agent import Agent

class STARAgent(Agent):
    def __init__(self, 
                 prompt_type: str = "caption",
                 verbose: bool = False,
                 logdir: str = None,
                 logger_prefix: str = "",
                 is_interactive: bool = False,
                 robot_model: str = ""
                 ):
        
        super().__init__(verbose, logdir, logger_prefix, is_interactive, robot_model)

        prompt_dir = os.path.join(os.path.dirname(__file__), "prompts", f"replan_{prompt_type}", "low_level_agent")
        prompt_prefix = robot_model if robot_model == "" else f"{robot_model}_"
        self.agent_prompt = file_to_string(os.path.join(prompt_dir, f"{prompt_prefix}agent_prompt.txt"))
        self.agent_gen_only_prompt = file_to_string(os.path.join(prompt_dir, f"{prompt_prefix}agent_gen_only_prompt.txt"))
        self.agent_reflect_prompt = file_to_string(os.path.join(prompt_dir, f"{prompt_prefix}agent_reflect_prompt.txt"))

    def set_task(self, task_desc: str):
        return super().set_task(task_desc)
            
    def setup_tools(self, memory: MilvusMemory):
        super().setup_tools(memory)
        
        memory_search_tools = create_memory_search_tools(memory)
        inspect_tools = create_memory_inspection_tool(memory)
        terminate_tools = create_memory_terminate_tool(memory)
        reflection_tools = create_pause_and_think_tool()
        if self.robot_model == "tiago":
            robot_tools = create_tiago_physical_skills(self.json_store)
        else:
            robot_tools = create_physical_skills(self.json_store)

        self.search_tools = memory_search_tools + inspect_tools + robot_tools
        self.search_tool_definitions = [convert_to_openai_function(t) for t in self.search_tools]
        
        self.reflection_tools = reflection_tools
        self.reflection_tool_definitions = [convert_to_openai_function(t) for t in self.reflection_tools]
        self.terminate_tools = terminate_tools
        self.terminate_tool_definitions = [convert_to_openai_function(t) for t in self.terminate_tools]
        
    def flush_tool_threads(self):
        return super().flush_tool_threads()
       
    def agent(self, state: Agent.AgentState):
        max_search_in_time_cnt = 20
        n_reflection_intervals = 5
        if self.robot_model == "tiago":
            max_search_in_space_cnt = 2
        else:
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
                                
                        elif is_search_in_time_terminate_result(msg):
                            normalized = original_msg_content.replace("{{", "{").replace("}}", "}").replace("{{", "{").replace("}}", "}")
                            records = ast.literal_eval(normalized)
                            records = sorted(records, key=lambda d: float(d["timestamp"]))
                            for record in records:
                                content = get_image_message_for_record(
                                    int(record["id"]), 
                                    get_viz_path(self.memory, int(record["id"])),
                                    msg.tool_call_id)
                                message = HumanMessage(content=content)
                                image_messages.append(message)
                                
                        elif is_task_terminate_result(msg):
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
                    
                    if self.robot_model == "tiago": # TODO delete this after testing
                        def match_ahg_waypoint(position: list[float], theta: float) -> bool:
                            def _ang_diff(a: float, b: float) -> float:
                                """Smallest signed difference a-b wrapped to [-pi, pi]."""
                                d = (a - b + math.pi) % (2 * math.pi) - math.pi
                                return d

                            x, y = position[0], position[1]
                            waypoints = {
                                "kitchen_counter": ((-0.88, -5.01), -1.57),
                                "bookshelf": ((-3.18, -4.77), -1.57),
                                "coffee_table_next_to_astro": ((-4.08, -4.48), 3.14),
                                "black_round_table": ((-4.18, -3.05), 1.57),
                                "round_table_btw_red_chairs": ((-3.41, 0.50), 1.57),
                                "coffee_table_btw_red_chairs": ((-0.39, 2.22), 0.0),
                                "living_room_table": ((-0.47, 6.94), -1.57),
                                "coffee_table_next_to_tv": ((0.17, 7.91), 0.0),
                                "left_corner": ((-4.29, 10.55), 1.57),
                            }
                            
                            dist_th = 1.0  # meters
                            angle_th = math.radians(45.0)  # 45 degrees -> radians

                            # Collect all waypoints that satisfy both distance and orientation thresholds
                            candidates = []
                            for name, ((wx, wy), wyaw) in waypoints.items():
                                dist = math.hypot(x - wx, y - wy)
                                dtheta = abs(_ang_diff(theta, wyaw))
                                if dist <= dist_th and dtheta <= angle_th:
                                    candidates.append((dist, name, (wx, wy, wyaw)))

                            if not candidates:
                                return False

                            # Pick the closest by distance if multiple match
                            _, best_wp, (wx, wy, wyaw) = min(candidates, key=lambda t: t[0])
                            return best_wp
                        wp = match_ahg_waypoint(position, theta)
                        if self.target_wp == wp:
                            self.task.search_proposal = SearchProposal(
                                summary="",
                                instance_description="",
                                position=position,
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
                    
                    if len(self.searched_poses) > max_search_in_space_cnt:
                        return {
                            "history": additional_search_history,
                            "toolcalls": last_tool_calls,
                            "next_state": "search_in_space",
                        }
                
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
                    
                elif tool_call["name"] == "terminate":
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
                
        return {
            "messages": [response], 
            "history": additional_search_history + [response],
            "toolcalls": last_tool_calls,
            "next_state": next_state,
        }
    
    def evaluate(self, state: Agent.AgentState):
        return super().evaluate(state)

    def search_in_space(self, state: Agent.AgentState):
        return super().search_in_space(state)
    
    def build_graph(self):
        workflow = StateGraph(STARAgent.AgentState)
        
        workflow.add_node("agent", lambda state: try_except_continue(state, self.agent))
        workflow.add_node("search_action", ToolNode(self.search_tools))
        workflow.add_node("reflection_action", ToolNode(self.reflection_tools))
        workflow.add_node("terminate_action", ToolNode(self.terminate_tools))
        workflow.add_node("search_in_space", lambda state: try_except_continue(state, self.search_in_space))
        workflow.add_node("search_in_space_action", ToolNode(self.search_in_space_tools))
        workflow.add_node("evaluate", lambda state: try_except_continue(state, self.evaluate))
        
        workflow.add_conditional_edges(
            "agent",
            STARAgent.from_agent_to,
            {   
                "agent": "search_action",
                "reflection": "reflection_action",
                "terminate": "terminate_action",
                "search_in_space": "search_in_space",
                "end": "evaluate",
            }
        )
        workflow.add_edge("search_action", "agent")
        workflow.add_edge("reflection_action", "agent")
        workflow.add_edge("terminate_action", "agent")
        
        workflow.add_conditional_edges(
            "search_in_space",
            STARAgent.from_search_in_space_to,
            {
                "end": "evaluate",
                "agent": "search_in_space_action",
            }
        )
        workflow.add_edge("search_in_space_action", "search_in_space")
        workflow.add_edge("evaluate", END)
        
        workflow.set_entry_point("agent")
        self.graph = workflow.compile()
        
    def set_memory(self, memory: MilvusMemory):
        self.memory = memory
        self.setup_tools(memory)
        
    def run(self, question: str):
        
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