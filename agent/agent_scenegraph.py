from agent.utils.tools import *
from agent.agent import Agent
import random

class SceneGraphAgent(Agent):
    def __init__(self, 
                 verbose: bool = False,
                 logdir: str = None,
                 logger_prefix: str = ""):
        super().__init__(verbose, logdir, logger_prefix)
        
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
        return super().search_in_space(state)
    
    def build_graph(self):
        workflow = StateGraph(Agent.AgentState)
        
        workflow.add_node("search_in_space", lambda state: try_except_continue(state, self.search_in_space))
        workflow.add_node("search_in_space_action", ToolNode(self.search_in_space_tools))
        
        workflow.add_conditional_edges(
            "search_in_space",
            Agent.from_search_in_space_to,
            {
                "end": END,
                "agent": "search_in_space_action",
            }
        )

    def set_memory_sg(self, memory_sg):
        self.memory_sg = memory_sg

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
        self.task.search_proposal = None
        
        self.build_graph()
        
        inputs = { "messages": [
                (("user", self.task.task_desc)),
            ]
        }
        
        config = {"recursion_limit": 10}
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