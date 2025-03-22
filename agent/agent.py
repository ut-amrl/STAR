import os
from typing import Annotated, Sequence, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from utils.function_wrapper import FunctionsWrapper
from utils.utils import *
from utils.tools import create_recall_any_tool, create_find_any_at_tool

from memory.memory import MilvusMemory

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    task: Annotated[Sequence, replace_messages]
    

class Agent:
    def __init__(self, llm_type: str = "gpt-4", vlm_type: str = "gpt-4o", verbose: bool = False):
        self.verbose = verbose
        
        self.llm_type, self.vlm_type = llm_type, vlm_type
        self.llm = self._llm_selector(self.llm_type)
        self.vlm = self._vlm_selector(self.vlm_type)
        
    def set_memory(self, memory: MilvusMemory):
        self.memory = memory
        
        recall_any_tool = create_recall_any_tool(self.memory, self.llm, self.vlm)
        self.recall_tools = [recall_any_tool]
        self.recall_tool_definitions = [convert_to_openai_function(t) for t in self.recall_tools]
        
        prompt_dir = os.path.join(str(os.path.dirname(__file__)), "prompts", "agent")
        self.recall_any_prompt = file_to_string(os.path.join(prompt_dir, 'recall_any_prompt.txt'))
        
        self._build_graph()
    
    def _llm_selector(self, llm_type):
        if 'gpt-4' in llm_type:
            import os
            llm = ChatOpenAI(model='gpt-4', api_key=os.environ.get("OPENAI_API_KEY"))
            return FunctionsWrapper(llm)
        else:
            raise ValueError("Unsupported LLM type!")
        
    def _vlm_selector(self, vlm_type):
        if 'gpt-4' in vlm_type:
            import os
            model = ChatOpenAI(model='gpt-4o', api_key=os.environ.get("OPENAI_API_KEY"))
            model = FunctionsWrapper(model)
            processor = None
        elif 'gwen' in vlm_type:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto"
            )
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        else:
            raise ValueError("Unsupported VLM type!")
        return model, processor
    
    def initialize(self, state):
        messages = state["messages"]
        task = messages[0].content
        return {"task": task}
    
    def recall_any(self, state):
        model = self.llm
        model = model.bind_tools(tools=self.recall_tool_definitions)
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("ai", self.recall_any_prompt),
                ("human", "{question}"),
            ]
        )
        model = prompt | model
        question = f"User Task: {state['task']}"
        response = model.invoke({"question": question})
        return {"messages": [response]}

    def terminate(self, state):
        import pdb; pdb.set_trace()
        print()
        pass
    
    def _build_graph(self):
        from langgraph.graph import END, StateGraph
        from langgraph.prebuilt import ToolNode
        
        workflow = StateGraph(AgentState)
        
        workflow.add_node("initialize", lambda state: self.initialize(state))
        workflow.add_node("recall_any_node", lambda state: try_except_continue(state, self.recall_any))
        workflow.add_node("recall_any_action_node", ToolNode(self.recall_tools))
        workflow.add_node("terminate", lambda state: self.terminate(state))
        
        workflow.add_edge("initialize", "recall_any_node")
        workflow.add_edge("recall_any_node", "recall_any_action_node")
        workflow.add_edge("recall_any_action_node", "terminate")
        
        # workflow.add_edge("initialize", "terminate")
        workflow.add_edge("terminate", END)
        
        workflow.set_entry_point("initialize")
        self.graph = workflow.compile()
        
    def run(self, question: str):
        inputs = { 
            "messages": [
                (("user", question)),
            ],
        }
        self.graph.invoke(inputs)
        
if __name__ == "__main__":
    from utils.memloader import remember_from_paths
    memory = MilvusMemory("test", obs_savepth="data/cobot/cobot_test_1", db_ip='127.0.0.1')
    memory.reset()
    inpaths = [
        "/robodata/taijing/RobotMem/data/captions/cobot/2025-03-10-17-01-55_VILA1.5-8b_3_secs.json",
        "/robodata/taijing/RobotMem/data/captions/cobot/2025-03-10-17-00-15_VILA1.5-8b_3_secs.json",
    ]
    t_offset = 1738952666.5530548-len(inpaths)*86400 + 86400
    remember_from_paths(memory, inpaths, t_offset, viddir="/robodata/taijing/RobotMem/data/images")
    
    agent = Agent()
    agent.set_memory(memory)
    agent.run(question="Today is 2025-02-07. Where is the coffee that was on a table yesterday?")
    