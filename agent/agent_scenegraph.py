from agent.utils.tools import *
from agent.agent import Agent
import random

class SceneGraphAgent(Agent):
    def __init__(self, 
                 verbose: bool = False,
                 logdir: str = None,
                 logger_prefix: str = "",
                 is_interactive: bool = False
        ):
        super().__init__(verbose, logdir, logger_prefix, is_interactive)

        search_in_space_prompt_dir = os.path.join(os.path.dirname(__file__), "prompts", "search_in_space_sg")
        self.search_in_space_prompt = file_to_string(os.path.join(search_in_space_prompt_dir, "search_in_space_prompt.txt"))
        
        self.memory_sg = None
        
    def set_task(self, task_desc: str):
        return super().set_task(task_desc)
    
    def setup_tools(self, memory: MilvusMemory):
        super().setup_tools(memory)
        
    def flush_tool_threads(self):
        return super().flush_tool_threads()
    
    def agent(self, state: Agent.AgentState):
        pass
    
    def evaluate(self, state: Agent.AgentState):
        return super().evaluate(state)

    def search_in_space(self, state: Agent.AgentState):
        if self.is_interactive:
            max_search_in_space_cnt = 5
        else:
            max_search_in_space_cnt = 3    
        
        messages = state["messages"]
        
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
                            "next_state": "terminate"
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
                            "next_state": "terminate"
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
                        "next_state": "terminate"
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
                            "next_state": "terminate"
                        }
        
        next_state = state.get("next_state", "agent")
        if next_state == "terminate" or self.search_in_space_cnt >= max_search_in_space_cnt:
            return {
                "history": additional_search_history,
                "toolcalls": last_tool_calls,
                "next_state": "terminate",
            }
        
        chat_history = copy.deepcopy(state.get("history", []))
        chat_history += additional_search_history
        
        model = self.vlm
        model_flex = self.vlm_flex
        model = model.bind_tools(self.search_in_space_tool_definitions)
        model_flex = model_flex.bind_tools(self.search_in_space_tool_definitions)
        prompt = self.search_in_space_prompt

        memory_str = f"This is how I remember the environment:\n\n {self.memory_sg}"
        chat_template = [
            ("human", memory_str),
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
        fact_prompt = f"You have been patrolling in a dynamic household or office environment, so objects you saw before may have been moved, or its status may be changed.\n"

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
        workflow = StateGraph(Agent.AgentState)
        
        workflow.add_node("search_in_space", lambda state: try_except_continue(state, self.search_in_space))
        workflow.add_node("search_in_space_action", ToolNode(self.search_in_space_tools))
        
        workflow.add_conditional_edges(
            "search_in_space",
            Agent.from_search_in_space_to,
            {
                "terminate": END,
                "agent": "search_in_space_action",
            }
        )
        workflow.add_edge("search_in_space_action", "search_in_space")
        
        workflow.set_entry_point("search_in_space")
        self.graph = workflow.compile()

    def set_memory_sg(self, memory_sg):
        self.memory_sg = memory_sg
        self.setup_tools(None)

    def run(self, question: str):
        if self.memory_sg is None:
            raise ValueError("Memory scene graph must be set before running the agent (check `set_memory_sg`).")
        
        if self.logger:
            self.logger.info("=============== START ===============")
            self.logger.info(f"User question: {question}.")
            
        self.set_task(question)
        
        self.search_in_space_cnt = 0
        self.searched_poses = []
        self.searched_visible_instances = []
        self.task.search_proposal = SearchProposal(
            summary="",
            instance_description="",
            position=None,
            theta=None,
            records=[]
        )
        
        self.build_graph()
        
        inputs = { "messages": [
                (("user", self.task.task_desc)),
            ]
        }
        
        config = {"recursion_limit": 15}
        state = self.graph.invoke(inputs, config=config)
        
        if self.logger:
            self.logger.info("=============== END =============== \n\n\n")
            
        toolcalls = []
        for msg in state.get("toolcalls", []):
            toolcalls += msg.tool_calls
        self.task.search_proposal.searched_poses = self.searched_poses
        
        self.memory_sg = None
        return {
            "task_result": self.task.search_proposal,
            "toolcalls": toolcalls,
        }