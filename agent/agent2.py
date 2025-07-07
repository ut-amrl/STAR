import os
from langchain_openai import ChatOpenAI

from agent.utils.function_wrapper import FunctionsWrapper # TODO need to clean up FunctionsWrapper
from agent.utils.tools2 import *

class Task:
    def __init__(self, task_desc: str):
        self.task_desc: str = task_desc
        self.search_instance: SearchInstance
        
        self.searched_in_space: list = []
        self.searched_in_time: list = []

class Agent:
    def __init__(self,
        agent_type: str,
        allow_recaption: bool = False,
        allow_replan: bool = False,
        allow_common_sense: bool = False,
        verbose: bool = False,
    ):
        self.agent_type: str = agent_type
        self.allow_recaption: bool = allow_recaption
        self.allow_replan: bool = allow_replan
        self.allow_common_sense: bool = allow_common_sense
        self.verbose: bool = verbose
        
        self.task: Task = None
        
        self.llm_raw = ChatOpenAI(model="gpt-4", api_key=os.environ.get("OPENAI_API_KEY"))
        self.llm = FunctionsWrapper(self.llm_raw)
        self.vlm_raw =  ChatOpenAI(model='gpt-4o', api_key=os.environ.get("OPENAI_API_KEY"))
        self.vlm = FunctionsWrapper(self.vlm_raw)
        
    def set_task(self, task_desc: str):
        self.task = Task(task_desc)
        if self.verbose:
            print(f"Task set: {task_desc}")
            
    