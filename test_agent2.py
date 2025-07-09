import datetime
import time
import argparse

from agent.agent2 import Agent
from memory.memory import MilvusMemory
from agent.utils.skills import *
from agent.utils.tools2 import (
    create_recall_best_match_tool, 
    create_memory_inspection_tool,
    create_determine_search_instance_tool,
    create_determine_unique_instances_tool
)
from agent.utils.function_wrapper import FunctionsWrapper

def load_toy_memory(memory: MilvusMemory):
    ONE_DAY = 24 * 60 * 60  # seconds in o
    from datetime import datetime
    start_t = datetime.strptime("2025-07-08 19:30:00", "%Y-%m-%d %H:%M:%S").timestamp()
    
    records = [
        {
            "time": start_t+0.0,
            "base_position": [0.0, 0.0, 0.0],
            "base_caption": "I saw a cup",
            # "base_caption": "I saw an object. It has coffee in it.",
            "start_frame": 0,
            "end_frame": 10
        },
        {
            "time": start_t+1.0,
            "base_position": [0.56, 0.0, 1.0],
            "base_caption": "I saw a table",
            "start_frame": 11,
            "end_frame": 20
        },
        {
            "time": start_t+2.0,
            "base_position": [1.1, 1.0, 1.0],
            "base_caption": "I saw a chair",
            "start_frame": 21,
            "end_frame": 30
        },
        {
            "time": start_t+3.0,
            "base_position": [1.1, 1.0, 1.0],
            "base_caption": "I saw a cup",
            "start_frame": 31,
            "end_frame": 40
        },
        {
            "time": ONE_DAY+start_t+0.0,
            "base_position": [0.0, 0.1, 0.0],
            "base_caption": "I saw a cup",
            "start_frame": 41,
            "end_frame": 50
        },
        
    ]
 
    for record in records:
        embedding = memory.embedder.embed_query(record["base_caption"])
        memory_item = MemoryItem(
            caption=record["base_caption"],
            text_embedding=embedding,
            time=record["time"],
            position=record["base_position"],
            theta=0.0,  # Assuming theta is not used in this toy example
            vidpath="debug/toy_examples",
            start_frame=record["start_frame"],
            end_frame=record["end_frame"]
        )
        memory.insert(memory_item)

def run_recall_best_match_tool(memory: MilvusMemory):
    
    llm_raw = ChatOpenAI(model="gpt-4", api_key=os.environ.get("OPENAI_API_KEY"))
    llm = FunctionsWrapper(llm_raw)
    vlm_raw =  ChatOpenAI(model='gpt-4o', api_key=os.environ.get("OPENAI_API_KEY"))
    vlm = FunctionsWrapper(vlm_raw)
        
    tool = create_recall_best_match_tool(
        memory=memory,
        llm=llm,
        llm_raw=llm_raw,
        vlm=vlm,
        vlm_raw=vlm_raw,
        logger=None
    )[0]
    output = tool.run({
        "user_task": "Bring me a cup",
        "history_summary": "I have created a search instance for a cup. Now I need to gather information from my memory.",
        "current_task": "figure out where I can find a cup based on my memory",
        "instance_description": "cup"
    })
    print("Tool output:\n", output)
    import pdb; pdb.set_trace()
    
def run_recall_best_match_tool2(memory: MilvusMemory):
    
    llm_raw = ChatOpenAI(model="gpt-4", api_key=os.environ.get("OPENAI_API_KEY"))
    llm = FunctionsWrapper(llm_raw)
    vlm_raw =  ChatOpenAI(model='gpt-4o', api_key=os.environ.get("OPENAI_API_KEY"))
    vlm = FunctionsWrapper(vlm_raw)
        
    tool = create_recall_best_match_tool(
        memory=memory,
        llm=llm,
        llm_raw=llm_raw,
        vlm=vlm,
        vlm_raw=vlm_raw,
        logger=None
    )[0]
    output = tool.run({
        "context": "I want to find where I am most likely to find an apple.",
        "instance_description": "apple"
    })
    print("Tool output:\n", output)

def run_memory_inspection_tool(memory: MilvusMemory):
    
    llm_raw = ChatOpenAI(model="gpt-4", api_key=os.environ.get("OPENAI_API_KEY"))
    llm = FunctionsWrapper(llm_raw)
    vlm_raw =  ChatOpenAI(model='gpt-4o', api_key=os.environ.get("OPENAI_API_KEY"))
    vlm = FunctionsWrapper(vlm_raw)
    
    tool = create_memory_inspection_tool(
        memory=memory,
    )[0]
    
    from langgraph.prebuilt import ToolNode
    # tool_node = ToolNode([tool])
    # output = tool_node.invoke({
    #     "record_id": "0",
    # })
    from langchain_core.messages import HumanMessage, ToolMessage
    tool_msg = ToolMessage(
        tool_call_id="test",
        name="inspect_memory_record",
        content=tool.invoke({"record_id": 0})
    )
    user_msg = HumanMessage(content="What is shown in this image?")
    response = vlm_raw.invoke([tool_msg, user_msg])
    print(response.content)
    import pdb; pdb.set_trace()
    
def run_determine_search_intance_tool(memory: MilvusMemory):
    llm_raw = ChatOpenAI(model="gpt-4", api_key=os.environ.get("OPENAI_API_KEY"))
    llm = FunctionsWrapper(llm_raw)
    vlm_raw =  ChatOpenAI(model='gpt-4o', api_key=os.environ.get("OPENAI_API_KEY"))
    vlm = FunctionsWrapper(vlm_raw)
    
    tool = create_determine_search_instance_tool(
        memory=memory,
        llm=llm,
        llm_raw=llm_raw,
        vlm=vlm,
        vlm_raw=vlm_raw,
        logger=None
    )[0]
    
    memory_records = memory.get_all()
    memory_records = eval(memory_records)
    
    output = tool.run({
        "user_task": "Bring me a cup",
        "history_summary": "I haven't made any tool calls yet. I need to frist generate a search instance.",
        "current_task": "determine search instace",
        "memory_records": memory_records,
    })
    print("Tool output:\n", output)

def run_determine_unique_instances_tool(memory: MilvusMemory):
    llm_raw = ChatOpenAI(model="gpt-4", api_key=os.environ.get("OPENAI_API_KEY"))
    llm = FunctionsWrapper(llm_raw)
    vlm_raw =  ChatOpenAI(model='gpt-4o', api_key=os.environ.get("OPENAI_API_KEY"))
    vlm = FunctionsWrapper(vlm_raw)
    
    tool = create_determine_unique_instances_tool(
        memory=memory,
        llm=llm,
        llm_raw=llm_raw,
        vlm=vlm,
        vlm_raw=vlm_raw,
        logger=None
    )[0]
    
    memory_records = memory.get_all()
    memory_records = eval(memory_records)
    
    output = tool.run({
        "user_task": "Bring me my favorite cup",
        "history_summary": "I have created a search instance for a cup. Now I need to gather information from my memory.",
        "current_task": "figure out what cups do we have based on my memory",
        "instance_description": "cup",
        "memory_records": memory_records
    })
    print("Tool output:\n")
    for item in output:
        print(item)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    memory = MilvusMemory("test", obs_savepth=None)
    memory.reset()
    load_toy_memory(memory)
    
    run_determine_unique_instances_tool(memory)
    run_recall_best_match_tool(memory)
    
    agent = Agent("full")
    agent.set_memory(memory)
    task_metadata = {
        "task_desc": "Find me a cup",
        "today_str": "2025-01-02"
    }
    agent.run(task_metadata)
    
    # run_recall_best_match_tool2(memory)
    # run_determine_search_intance_tool(memory)
    