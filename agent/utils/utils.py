import traceback, sys
from typing import Sequence
import json
from langchain_core.messages import ToolMessage

def file_to_string(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read().strip()
    
def try_except_continue(state, func):
    while True:
        try:
            ret = func(state)
            return ret
        except Exception as e:
            print("I crashed trying to run:", func)
            print("Here is my error")
            print(e)
            traceback.print_exception(*sys.exc_info())
            continue
        
def replace_messages(current: Sequence, new: Sequence):
    """Custom update strategy to replace the previous value with the new one."""
    return new

def filter_retrieved_record(messages: list):
    tool_responses = [msg.content for msg in filter(lambda x: isinstance(x, ToolMessage), messages)]
    records = []
    for response in tool_responses:
        for r in json.loads(response):
            records.append(r)
    unique_by_ids = {r["id"]: r for r in records}
    records = sorted(unique_by_ids.values(), key=lambda x: x["id"])
    return records
            
def parse_db_records_for_llm(messages: list):
    processed_records = [{"record_id": record["id"], "text": record["text"]} for record in messages]
    parsed_processed_records = [json.dumps(record) for record in processed_records]
    return parsed_processed_records