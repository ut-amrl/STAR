import os
from langchain_openai import ChatOpenAI

from agent.utils.debug import get_logger
from agent.utils.function_wrapper import FunctionsWrapper # TODO need to clean up FunctionsWrapper
from agent.utils.tools2 import *

class Agent:
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        agent_history: Annotated[Sequence[BaseMessage], add_messages]
        search_in_time_history: Annotated[Sequence[BaseMessage], add_messages]
        search_in_space_history: Annotated[Sequence[BaseMessage], add_messages]
        last_response: Annotated[Sequence, replace_messages]
    
    def __init__(self,
        agent_type: str,
        allow_recaption: bool = False,
        allow_replan: bool = False,
        allow_common_sense: bool = False,
        verbose: bool = False,
    ):
        self.logger = get_logger()
        
        self.agent_type: str = agent_type
        self.allow_recaption: bool = allow_recaption
        self.allow_replan: bool = allow_replan
        self.allow_common_sense: bool = allow_common_sense
        self.verbose: bool = verbose
        
        self.task: Task = None
        
        # self.llm_raw = ChatOpenAI(model="gpt-4-turbo", api_key=os.environ.get("OPENAI_API_KEY"))
        self.llm_raw = ChatOpenAI(model="gpt-4", api_key=os.environ.get("OPENAI_API_KEY"))
        self.llm = FunctionsWrapper(self.llm_raw)
        self.vlm_raw =  ChatOpenAI(model='gpt-4o', api_key=os.environ.get("OPENAI_API_KEY"))
        self.vlm = FunctionsWrapper(self.vlm_raw)
        
        prompt_dir = os.path.join(str(os.path.dirname(__file__)), "prompts", "agent2")
        self.search_in_time_prompt = file_to_string(os.path.join(prompt_dir, 'search_in_time_prompt.txt'))
        
        self.working_memory = [[]]
        
    def set_task(self, task_desc: str):
        self.task = Task(task_desc)
        if self.verbose:
            print(f"Task set: {task_desc}")
            
    def agent(self, state: AgentState):
        pass
            
    def search_in_time(self, state: AgentState):
        messages = state["messages"]
        
        model = self.vlm
        
        history_summary = state["search_in_time_history"]
        if self.task.memory_search_instance is None:
            memory_search_instance_msg = "None"
        else:
            memory_search_instance_msg = self.task.memory_search_instance.to_message()
        if self.task.world_search_instance is None:
            world_search_instance_msg = "None"
        else:
            world_search_instance_msg = self.task.world_search_instance.to_message()
        
        escaped_history_summary = []
        for msg in history_summary:
            if hasattr(msg, "content") and isinstance(msg.content, str):
                msg.content = msg.content.replace("{", "{{").replace("}", "}}")
            escaped_history_summary.append(msg)
        
        working_memory = str(self.working_memory[-1]).replace("{", "{{").replace("}", "}}")
        
        messages = []
        messages += [("human", "This is previous tool calls and the responses (Please summary them in `history_summary`):")]
        messages += history_summary
        messages += [
            ("human", "Please follow the instructions below to determine the next action to take:"),
            ("human", self.search_in_time_prompt),
            ("system", "{fact_prompt}"),
            ("system", "Huamn will provide you with the current search instances in memory and real world. If you call any recall_* tool, they will use the current memory_search_instance as the target instance to search in memory, and world_search_instance as the target instance to search in real world."),
            ("human", memory_search_instance_msg),
            ("human", world_search_instance_msg),
            ("human", f"Here is the current working memory: {working_memory}"),
            ("system", "Please determine the next action to take! Remember you can only call the provided tools, and stick strictly to the JSON format. Reason carefully about memory search instance, world search instance, and tool call history before making a decision."),
        ]
        
        print("memory_search_instance_msg: ", memory_search_instance_msg)
        chat_prompt = ChatPromptTemplate.from_messages(messages)
        # chat_prompt = ChatPromptTemplate.from_messages([
        #     ("human", "This is previous tool calls and the responses (Please summary them in `history_summary`):"),
        #     MessagesPlaceholder("history_summary"),
        #     ("human", "Please follow the instructions below to determine the next action to take:"),
        #     ("human", self.search_in_time_prompt),
        #     ("system", "{fact_prompt}"),
        #     ("human", memory_search_instance_msg),
        #     ("human", world_search_instance_msg),
        #     ("human", f"Here is the current working memory: {self.working_memory[-1]}"),
        #     ("system", "Please determine the next action to take! Remember you can only call the provided tools, and stick strictly to the JSON format."),
        # ])
        fact_prompt = "Here are some facts about the current situation:\n" + f"1. The current date is: {self.today_str}.\n" + f"2. The user task is: {self.task.task_desc}.\n"
        chained_model = chat_prompt | model
        
        response = chained_model.invoke({
            "fact_prompt": fact_prompt, 
            # "history_summary": history_summary,
        })
        
        keys_to_check_for = ["history_summary", "current_task", "tool_call", "tool_input"]
        parsed = eval(response.content)
        for key in keys_to_check_for:
            if key not in parsed:
                raise ValueError("Missing required keys during generate. Retrying...")
        if parsed["tool_call"] not in [
            "create_or_update_memory_search_instance",
            "create_or_update_real_world_search_instance",
            "determine_unique_instances_from_latest_working_memory",
            "recall_best_match",
            "recall_last_seen",
            "search_current_target_instance_in_real_world"
        ]:
            raise ValueError(f"Invalid tool call: {parsed['tool_call']}. Retrying...")
        
        parsed_response = {
            "history_summary": parsed["history_summary"],
            "current_task": parsed["current_task"],
            "tool_call": parsed["tool_call"],
            "tool_input": parsed["tool_input"],
            "search_start_time": parsed.get("search_start_time", None),
            "search_end_time": parsed.get("search_end_time", None),
        }
        
        tool_call = parsed_response["tool_call"]
        if tool_call == "create_or_update_memory_search_instance" or \
           tool_call == "create_or_update_real_world_search_instance" or \
           tool_call == "search_current_target_instance_in_real_world":
            tool_call_str = f"{tool_call}()"
        else:
            start_time = parsed_response.get("search_start_time", None)
            end_time = parsed_response.get("search_end_time", None)
            tool_input = parsed_response["tool_input"]
            if start_time and end_time:
                tool_call_str = f"{tool_call}({tool_input}, search_start_time='{start_time}', search_end_time='{end_time}')"
            else:
                tool_call_str = f"{tool_call}({tool_input})"
        tool_call_summary = "----------------\n" 
        tool_call_summary += f"History Summary: {parsed_response['history_summary']}\n" \
                            f"Current Task: {parsed_response['current_task']}\n" \
                            f"Tool Call: {tool_call_str}\n"
        tool_call_summary += "----------------\n" 
        
        if self.logger:
            self.logger.info(f"[SEARCH_IN_TIME] search_in_time() - Current Task: {parsed_response['current_task']}; Tool Call: {tool_call_str}")
    
        return {"messages": [response], 
                # "search_in_time_history": [tool_call_summary], 
                "search_in_time_history": [response],
                "last_response": parsed_response,}
        
    def search_in_time_action(self, state: AgentState):
        tool_call_metadata = state["last_response"]
        
        tool_call = tool_call_metadata["tool_call"]
        
        output = None
        if tool_call == "create_or_update_memory_search_instance":
            output = self.create_or_update_memory_search_instance_tool.run({
                "user_task": self.task.task_desc,
                "history_summary": tool_call_metadata["history_summary"],
                "current_task": tool_call_metadata["current_task"],
                "memory_records": self.working_memory[-1]
            })
            if self.task.memory_search_instance is None:
                self.task.memory_search_instance = SearchInstance("memory")
            self.task.memory_search_instance.found = output.get("found_in_memory", "unknown")
            self.task.memory_search_instance.inst_desc = output.get("instance_desc", "")
            self.task.memory_search_instance.inst_viz_path = output.get("instance_viz_path", None)
            self.task.memory_search_instance.past_observations = output.get("past_observations", [])
        elif tool_call == "create_or_update_real_world_search_instance":
            output = self.create_or_update_real_world_instance_tool.run({
                "user_task": self.task.task_desc,
                "history_summary": tool_call_metadata["history_summary"],
                "current_task": tool_call_metadata["current_task"],
                "memory_records": self.working_memory[-1]
            })
            if self.task.world_search_instance is None:
                self.task.world_search_instance = SearchInstance("world")
            self.task.world_search_instance.found = output.get("found_in_world", "unknown")
            self.task.world_search_instance.inst_desc = output.get("instance_desc", "")
            self.task.world_search_instance.inst_viz_path = output.get("instance_viz_path", None)
            self.task.world_search_instance.past_observations = output.get("past_observations", [])
        elif tool_call == "recall_best_match":
            output = self.recall_best_match_tool.run({
                "user_task": self.task.task_desc,
                "history_summary": tool_call_metadata["history_summary"],
                "current_task": tool_call_metadata["current_task"],
                "instance_description": tool_call_metadata["tool_input"]
            })
            self.working_memory.append(output)  # Update working memory with the output
        elif tool_call == "recall_last_seen":
            import pdb; pdb.set_trace()
        elif tool_call == "determine_unique_instances_from_latest_working_memory":
            output = self.determine_unique_instances_tool.run({
                "user_task": self.task.task_desc,
                "history_summary": tool_call_metadata["history_summary"],
                "current_task": tool_call_metadata["current_task"],
                "instance_description": tool_call_metadata["tool_input"],
                "memory_records": self.working_memory[-1]
            })
            self.working_memory.append(output)  # Update working memory with the output
        elif tool_call == "search_current_target_instance_in_real_world": # Terminate
            import pdb; pdb.set_trace()
        else:
            import pdb; pdb.set_trace()

        import pdb; pdb.set_trace()

        tool_response_summary = "----------------\n"
        tool_response_summary += f"Tool Response: {output}\n"
        tool_response_summary += "----------------\n"
        
        if self.logger:
            self.logger.info(f"[SEARCH_IN_TIME] search_in_time_action() - Tool Response: {output}")
            
        return {"search_in_time_history": [str(output)], 
                "last_response": output,}
    
    def search_in_space(self, state: AgentState):
        pass
            
    def search_in_space_action(self, state: AgentState):
        pass
            
    def build_graph(self):
        """
        Build the graph for the agent.
        """
        workflow = StateGraph(Agent.AgentState)
        
        workflow.add_node("search_in_time", lambda state: try_except_continue(state, self.search_in_time))
        workflow.add_node("search_in_time_action", lambda state: try_except_continue(state, self.search_in_time_action))
        
        workflow.add_edge("search_in_time", "search_in_time_action")
        workflow.add_edge("search_in_time_action", "search_in_time")
        
        workflow.set_entry_point("search_in_time")
        self.graph = workflow.compile()
        
    def set_memory(self, memory: MilvusMemory):
        self.memory = memory
        
        # Recall Tools
        self.recall_best_match_tool = create_recall_best_match_tool(
            memory=memory,
            llm=self.llm,
            llm_raw=self.llm_raw,
            vlm=self.vlm,
            vlm_raw=self.vlm_raw,
            logger=self.logger
        )[0]
        # self.recall_last_seen_tool = create_recall_last_seen_tool(
        #     memory=memory,
        #     llm=self.llm,
        #     llm_raw=self.llm_raw,
        #     vlm=self.vlm,
        #     vlm_raw=self.vlm_raw,
        #     logger=self.logger
        # )[0]
        
        # Decision Making Tools
        determin_search_instance_tools = create_determine_search_instance_tool(
            memory=memory,
            llm=self.llm,
            llm_raw=self.llm_raw,
            vlm=self.vlm,
            vlm_raw=self.vlm_raw,
            logger=self.logger
        )
        self.create_or_update_memory_search_instance_tool = determin_search_instance_tools[0]
        self.create_or_update_real_world_instance_tool = determin_search_instance_tools[1]
        self.determine_unique_instances_tool = create_determine_unique_instances_tool(
            memory=memory,
            llm=self.llm,
            llm_raw=self.llm_raw,
            vlm=self.vlm,
            vlm_raw=self.vlm_raw,
            logger=self.logger
        )[0]
        
            
    def run(self, question: str, today: str, graph_type: str):
        self.task = Task(question)
        self.today_str = today
        
        self.build_graph()
        
        inputs = { "messages": [
                (("user", self.task.task_desc)),
            ]
        }
        state = self.graph.invoke(inputs)
        import pdb; pdb.set_trace()
        
    ##############################
    # Recall Last Seen (find_by_description)
    ##############################
    
    def _recall_last_seen_retriever(self, 
                                    current_task: str, 
                                    keyword_prompt: str, 
                                    identification_prompt: str):
        model = self.llm
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", keyword_prompt),
                ("human", "{question}"),
            ]
        )
        model = prompt | model
        question = current_task
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
            explored_record_ids = set()
            explored_positions = []
            
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
    
    def _recall_last_seen_from_txt(self, current_goal: str):
        records = self._recall_last_seen_retriever(
            current_goal, 
            self.get_param_from_txt_prompt, 
            self.find_instance_from_txt_prompt)
        if len(records) == 0:
            return None
        return records[:1]
    
    def _recall_last_seen(self, current_task: str):
        self._recall_last_seen_from_txt(current_task)