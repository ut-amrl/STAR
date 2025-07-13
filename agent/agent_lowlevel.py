import os
from langchain_openai import ChatOpenAI

from agent.utils.debug import get_logger
from agent.utils.function_wrapper import FunctionsWrapper # TODO need to clean up FunctionsWrapper
from agent.utils.tools2 import *

class AgentLowLevel:
    def __init__(self):
        self.logger = get_logger()
        
        self.task: Task = None
        
        self.llm_raw = ChatOpenAI(model="o3", api_key=os.environ.get("OPENAI_API_KEY"))
        self.llm = FunctionsWrapper(self.llm_raw)
        self.vlm_raw = ChatOpenAI(model="o3", api_key=os.environ.get("OPENAI_API_KEY"))
        self.vlm = FunctionsWrapper(self.vlm_raw)
        
        prompt_dir = os.path.join(os.path.dirname(__file__), "prompts", "agent_lowlevel")