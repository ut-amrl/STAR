import os
import re
from typing import Annotated, Sequence, TypedDict
from functools import partial

from langchain.tools import StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.utils.utils import *
from agent.utils.tools import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from memory.memory import MilvusMemory

def create_db_txt_search_tool(memory):
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
    return [txt_retriever_tool, txt_time_retriever_tool]

def create_find_by_description_with_time_tool(
    memory: MilvusMemory, 
    llm, 
    vlm, 
    vlm_raw, 
    allow_recaption: bool,
    logger=None
):
    
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        retrieved_messages: Annotated[Sequence, replace_messages]
        output: Annotated[Sequence, replace_messages] = None
        
    def from_agent_to(state: AgentState):
        messages = state["messages"]

        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "generate"
        else:
            return "action"
        
    def from_generate_to(state: AgentState):
        output = state["output"]

        if output is None:
            return "recaption"
        else:
            return "end"
        
    class DBRetriever:
        def __init__(self, memory, llm, vlm, vlm_raw, allow_recaption: bool, logger=None):
            self.memory = memory
            self.llm = llm
            self.vlm = vlm
            self.vlm_raw = vlm_raw
            self.allow_recaption = allow_recaption
            self.logger = logger

            self.db_retriever_tools = create_db_txt_search_tool(memory)
            self.recall_tool_definitions = [convert_to_openai_function(t) for t in self.db_retriever_tools]
            
            prompt_dir = str(os.path.dirname(__file__)) + '/../prompts/find_specific_past_instance/'
            self.agent_prompt = file_to_string(prompt_dir+'db_txt_query_prompt.txt')
            self.agent_gen_only_prompt = file_to_string(prompt_dir+'db_txt_query_terminate_prompt.txt')
            
            self.agent_call_count = 0
            self.has_recaptioned = False
            self.previous_tool_requests = "I have already used the following retrieval tools and the results are included below. Do not repeat them:\n"
            
            self.build_graph()
                
        def agent(self, state):
            messages = state["messages"]

            model = self.llm
            if self.agent_call_count < 2:
                model = model.bind_tools(tools=self.recall_tool_definitions)
                prompt = self.agent_prompt
            else:
                prompt = self.agent_gen_only_prompt
                
            question = f"The object user wants to find is: {messages[0].content}"
                
            db_messages = []
            if self.agent_call_count == 0:
                agent_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", prompt),
                        ("human", "{question}"),
                    ]
                )
                model = agent_prompt | model
                response = model.invoke({"question": question})
            else:
                agent_prompt = ChatPromptTemplate.from_messages(
                    [
                        MessagesPlaceholder("chat_history"),
                        (("human"), self.previous_tool_requests),
                        ("system", prompt),
                        ("human", "{question}"),
                    ]
                )
                model = agent_prompt | model
                
                db_messages = filter_retrieved_record(messages[:])
                parsed_db_messages = parse_db_records_for_llm(db_messages)
                response = model.invoke({"question": question, "chat_history": parsed_db_messages})
            
            if response.tool_calls and self.logger:
                self.logger.info(f"[SPECIFIC_INSTANCE] Calling tools: {response.tool_calls}")
                
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call['name'] != "__conversational_response":
                        args = re.sub(r'^\{(.*)\}$', r'(\1)', str(tool_call['args'])) # remove curly braces
                        self.previous_tool_requests += f" {tool_call['name']} tool with the arguments: {args}.\n"
            
            self.agent_call_count += 1
            return {"messages": [response], "retrieved_messages": db_messages}
        
        def generate(self, state):
            self.agent_call_count = 0
            messages = state["messages"]
            
            model = self.llm
            prompt = self.agent_gen_only_prompt
            agent_prompt = ChatPromptTemplate.from_messages(
                [
                    MessagesPlaceholder("chat_history"),
                    (("human"), self.previous_tool_requests),
                    ("system", prompt),
                    ("human", "{question}"),
                ]
            )
            model = agent_prompt | model
            
            db_messages = state["retrieved_messages"]
            parsed_db_messages = parse_db_records_for_llm(db_messages)
            question = f"Answer the question based on the instructions and the chat history - {messages[0].content}"
            last_message = model.invoke({"question": question, "chat_history": parsed_db_messages})
            
            if type(last_message) == str:
                raise ValueError("Missing required field 'moment_ids'")
            last_response = eval(last_message.content)
            if "moment_ids" not in last_response.keys():
                raise ValueError("Missing required field 'moment_ids'")
            # Recaption if needed
            if self.allow_recaption and (not self.has_recaptioned) and len(last_response["moment_ids"]) == 0 and len(state["retrieved_messages"]) != 0:
                if self.logger:
                    n_retrieved_messages = len(state["retrieved_messages"])
                    self.logger.info(f"Recaptioning {n_retrieved_messages} retrieved message(s)...")
                return {"messages": [last_message], "output": None}
            
            # Prepare for return
            record_ids = last_response["moment_ids"]
            if type(record_ids) == str:
                record_ids = eval(record_ids)
            record_ids = [int(id) for id in record_ids]
            
            retrieved_messages = state["retrieved_messages"]
            retrieved_messages = {r["id"]: r for r in retrieved_messages}
            
            records = []
            for id in record_ids:
                if id in retrieved_messages.keys():
                    records.append(retrieved_messages[id])
            if len(record_ids) != 0 and len(records) == 0:
                raise ValueError("Missing required field 'moment_ids'")
                
            if self.logger:
                self.logger.info(f"Makeing decisions based on: {retrieved_messages}")
                self.logger.info(f"Final decision(s): {last_response}")
            
            self.has_recaptioned = False
            return {"messages": [last_message], "output": records}
        
        def recaption(self, state):
            messages = state["messages"]
            task = messages[0].content
            
            db_messages = state["retrieved_messages"]
            caption_fn = partial(caption_gpt, model=self.vlm_raw, image_path_fn=None) # Use default image_path_fn
            
            if len(db_messages) > 20:
                import pdb; pdb.set_trace()
            
            # TODO should run this in parallel
            for record in db_messages:
                caption = recaption(self.memory, task, record, caption_fn)
                record["text"] = caption # update in-place
            self.memory.flush_and_reload()
            self.has_recaptioned = True
            return {"retrieved_messages": db_messages}
        
        def _build_graph_with_recaption(self):
            workflow = StateGraph(AgentState)
            
            workflow.add_node("agent", lambda state: try_except_continue(state, self.agent))
            workflow.add_node("action", ToolNode(self.db_retriever_tools))
            workflow.add_node("generate", lambda state: try_except_continue(state, self.generate))
            workflow.add_node("recaption", lambda state: try_except_continue(state, self.recaption))
            
            workflow.add_conditional_edges(
                "agent",
                from_agent_to,
                {
                    "action": "action",
                    "generate": "generate",
                },
            )
            workflow.add_edge('action', 'agent')
            workflow.add_conditional_edges(
                "generate",
                from_generate_to,
                {
                    "recaption": "recaption",
                    "end": END,
                },
            )
            workflow.add_edge("recaption", "generate")
            
            workflow.set_entry_point("agent")
            self.graph = workflow.compile()
            
        def _build_graph_without_recaption(self):
            workflow = StateGraph(AgentState)
            
            workflow.add_node("agent", lambda state: try_except_continue(state, self.agent))
            workflow.add_node("action", ToolNode(self.db_retriever_tools))
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
                
        def build_graph(self):
            if self.allow_recaption:
                self._build_graph_with_recaption()
            else:
                self._build_graph_without_recaption()
                
        def run(self, instance_description, search_start_time, search_end_time):
            self.agent_call_count = 0
            self.has_recaptioned = False
            self.previous_tool_requests = "I have already used the following retrieval tools and the results are included below. Do not repeat them:\n"
            
            inputs = { "messages": [
                    (("user", f"Between {search_start_time} and {search_end_time}, have you observe {instance_description}? If so, when?")),
                ]
            }
            self.instance_description = instance_description
            self.search_start_time = search_start_time
            self.search_end_time = search_end_time
            
            state = self.graph.invoke(inputs)
            output = state['output']
            import time; time.sleep(1)
            return output

    tool = DBRetriever(memory, llm, vlm, vlm_raw, allow_recaption, logger)
    class RecallAnyInput(BaseModel):
        instance_description: str = Field(description="You are a robot agent with extensive past observations. This tool helps you recall the specific moment when you observed an instance matching the given description. \
                            This query argument should be a phrase such as 'a book', 'the book that was on a table', or 'an apple that was in kitchen yesterday'. \
                            The query will then search your memories for you.")
        search_start_time: str = Field(
            description="Start search time in YYYY-MM-DD HH:MM:SS format. Only search for observations made after this timestamp."
        )
        search_end_time: str = Field(
            description="End search time in YYYY-MM-DD HH:MM:SS format. Only search for observations made before this timestamp."
        )
        
    retriever_tool = StructuredTool.from_function(
        func=lambda instance_description, search_start_time, search_end_time: tool.run(
            instance_description, search_start_time, search_end_time
        ),
        name="recall_observation_of_instance",
        description="Search memory to recall the moments when you observed an instance matching the given description",
        args_schema=RecallAnyInput
        # coroutine= ... <- you can specify an async method if desired as well
    )
    return retriever_tool

def create_best_guess_tool(memory: MilvusMemory, llm, vlm, logger=None):
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        retrieved_messages: Annotated[Sequence, replace_messages]
        output: Annotated[Sequence, replace_messages]
        
    def should_continue(state: AgentState):
        messages = state["messages"]

        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        else:
            return "continue"
    
    class DBRetriever:
        def __init__(self, memory, llm, vlm, logger=None):
            self.memory = memory
            self.llm = llm
            self.vlm = vlm
            self.logger = logger

            self.db_retriever_tools = create_db_txt_search_tool(memory)
            self.recall_tool_definitions = [convert_to_openai_function(t) for t in self.db_retriever_tools]
            
            prompt_dir = str(os.path.dirname(__file__)) + '/../prompts/find_best_guess/'
            self.agent_prompt = file_to_string(prompt_dir+'db_txt_query_prompt.txt')
            self.agent_gen_only_prompt = file_to_string(prompt_dir+'db_txt_query_terminate_prompt.txt')
            
            self.previous_tool_requests = "These are the tools I have previously used so far: \n"
            self.agent_call_count = 0
            
            self._build_graph()
                
        def agent(self, state):
            messages = state["messages"]

            model = self.llm
            if self.agent_call_count < 2:
                model = model.bind_tools(tools=self.recall_tool_definitions)
                prompt = self.agent_prompt
            else:
                prompt = self.agent_gen_only_prompt
                
            agent_prompt = ChatPromptTemplate.from_messages(
                [
                    MessagesPlaceholder("chat_history"),
                    ("system", prompt),
                    (("human"), self.previous_tool_requests),
                    ("human", "{question}"),
                ]
            )
            
            model = agent_prompt | model
            
            db_messages = filter_retrieved_record(messages[:])
            parsed_db_messages = parse_db_records_for_llm(db_messages)

            question = f"The object user wants to find is: {messages[0].content}"
            
            response = model.invoke({"question": question, "chat_history": parsed_db_messages})
            
            if response.tool_calls and self.logger:
                self.logger.info(f"[BEST GUESS] Calling tools: {response.tool_calls}")

            if response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call['name'] != "__conversational_response":
                        args = re.sub(r'^\{(.*)\}$', r'(\1)', str(tool_call['args'])) # remove curly braces
                        self.previous_tool_requests += f" {tool_call['name']} tool with the arguments: {args}.\n"

            self.agent_call_count += 1
            return {"messages": [response], "retrieved_messages": db_messages}
        
        def generate(self, state):
            if self.logger:
                self.logger.info(f"Makeing decisions based on: {state['retrieved_messages']}")
                
            return 
            
        def _build_graph(self):
            workflow = StateGraph(AgentState)
            
            workflow.add_node("agent", lambda state: try_except_continue(state, self.agent))
            workflow.add_node("action", ToolNode(self.db_retriever_tools))
            workflow.add_node("generate", lambda state: try_except_continue(state, self.generate))
            
            workflow.add_conditional_edges(
                "agent",
                # Assess agent decision
                should_continue,
                {
                    # Translate the condition outputs to nodes in our graph
                    "continue": "action",
                    "end": "generate",
                },
            )
            workflow.add_edge('action', 'agent')
            workflow.add_edge("generate", END)
            
            workflow.set_entry_point("agent")
            self.graph = workflow.compile()
            
        def run(self, instance_description):
            inputs = { "messages": [
                    (("user", f"Where can I find {instance_description}?")),
                ]
            }
            self.instance_description = instance_description
            
            state = self.graph.invoke(inputs)
            return state["messages"][-1]

    tool = DBRetriever(memory, llm, vlm, logger)
    class BestGuessInput(BaseModel):
        instance_description: str = Field(description="You are a robot agent with extensive past observations. This tool helps you infer where to find the instance matching the given description based on your past observations. \
                            This query argument should be a phrase such as 'a book', 'the book that was on a table', or 'an apple that was in kitchen yesterday'. \
                            The query will then search your memories for you.")
        
    retriever_tool = StructuredTool.from_function(
        func=lambda instance_description: tool.run(
            instance_description
        ),
        name="get_location_of_instance",
        description="Search memory to infer the location where you are most likely to find an instance matching the given description",
        args_schema=BestGuessInput
        # coroutine= ... <- you can specify an async method if desired as well
    )
    return retriever_tool

def create_recall_last_tool(memory: MilvusMemory, llm, vlm):
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        keywords: Annotated[Sequence, replace_messages]
        output: Annotated[Sequence, replace_messages]
        
    class DBLastRetriever:
        def __init__(self, memory, llm, vlm, local_vlm):
            self.memory = memory
            self.llm = llm
            self.vlm = vlm
            self.local_vlm = local_vlm
            
            prompt_dir = str(os.path.dirname(__file__)) + '/../prompts/recall_last/'
            self.get_param_from_txt_prompt = file_to_string(os.path.join(prompt_dir, 'get_param_from_txt_prompt.txt'))
            self.find_instance_from_txt_prompt = file_to_string(os.path.join(prompt_dir, 'find_instance_from_txt_prompt.txt'))
            
            self._build_graph()
        
        def param(self, state):
            question = state["messages"][0]
            question = eval(question.content)
            
            model = self.llm
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("ai", self.get_param_from_txt_prompt),
                    ("human", "{question}"),
                ]
            )
            model = prompt | model
            question = f"User Task: {question}"
            response = model.invoke({"question": question})
            keywords = eval(response.content)
            return {"keywords": keywords}
        
        def query(self, state):
            keywords = state["keywords"]
            
            query = ', or '.join(keywords)
            record_found = None
            for _ in range(5):
                docs = self.memory.search_last_k_by_text(is_first_time=True, query=query)
                if docs == '':
                    break
                
                parsed_docs = parse_db_records_for_llm(eval(docs))
                
                model = self.llm
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "{docs}"),
                        ("ai", self.find_instance_from_txt_prompt),
                        ("human", "{question}"),
                    ]
                )
                model = prompt | model
                question = f"User Task: {question}. Have you seen the instance user needs in your recalled moments?"
                response = model.invoke({"question": question, "docs": parsed_docs})
                
                record_id = response.content
                if len(record_id) == 0:
                    continue
                record_id = int(eval(record_id))
                
                for record in eval(docs):
                    if int(record["id"]) == record_id:
                        record_found = copy.copy(record)
                if record_found is not None:
                    break
                
            return {"output": record_found}
        
        def _build_graph(self):
            workflow = StateGraph(AgentState)
            
            workflow.add_node("param", lambda state: try_except_continue(state, self.param))
            workflow.add_node("query", lambda state: try_except_continue(state, self.query))
            
            workflow.add_edge("param", "query")
            workflow.add_edge("query", END)
            
            workflow.set_entry_point("param")
            self.graph = workflow.compile()
            
        def run(self, instance_description, obs=None):
            self.obs = obs
            self.instance_description = instance_description
            inputs = { "messages": [
                    (("user", instance_description)),
                ]
            }
            state = self.graph.invoke(inputs)
            output = state['output']
            ret = {
                "output": output,
                "instance_description": instance_description
            }
            return ret

    tool = DBLastRetriever(memory, llm, vlm)
    class RecallLastInput(BaseModel):
        instance_description: str = Field(description="You are a robot agent with extensive past observations. This tool helps you recall the specific moment when you observed an instance matching the given description. \
                            This query argument should be a phrase such as 'a book', 'the book that was on a table', or 'an apple that was in kitchen yesterday'. \
                            The query will then search your memories for you.")
        obs: str = Field(
            description="(Optional) This is an optional message for image message that contain the instance."
        )
    retriever_tool = StructuredTool.from_function(
        func=lambda instance_description, obs: tool.run(
            instance_description, obs
        ),
        name="recall_lastest_observation_of_instance",
        description="Search memory to recall the lastest moment when you observed the instance matching the given description",
        args_schema=RecallLastInput
        # coroutine= ... <- you can specify an async method if desired as well
    )
    return retriever_tool
