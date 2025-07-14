import os
from dataclasses import dataclass, asdict
from typing import List, Optional
import inspect 
from datetime import datetime
import re
import json

from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

@dataclass
class MemoryItem:
    caption: str
    text_embedding: List[float]
    time: float
    position: list
    theta: float
    vidpath: str
    start_frame: int
    end_frame: int

    @classmethod
    def from_dict(cls, dict_input):      
        return cls(**{
            k: v for k, v in dict_input.items() 
            if k in inspect.signature(cls).parameters
        })
    
    def __post_init__(self):
        # Not every method will use a caption, so we set it to none in those cases
        if self.caption is None:
            self.caption = ''
        if self.vidpath is None:
            self.vidpath = ''
            
class Memory:
    def get_last_id(self) -> int:
        raise NotImplementedError

    def insert(self, item: MemoryItem):
        raise NotImplementedError

    def memory_to_string(self, memory_list: list[MemoryItem]) -> str:
        raise NotImplementedError
    
class MilvusWrapper:

    def __init__(self, collection_name='test', ip_address='127.0.0.1', port=19530, drop_collection=False):
        self.collection_name = collection_name
        self.collection = self.connect_to_milvus_collection(collection_name, 1024, address=ip_address, port=port, drop_collection=drop_collection)

    def drop_collection(self):
        utility.drop_collection(self.collection_name)

    def connect_to_milvus_collection(self, collection_name, dim, address='127.0.0.1', port=19530, drop_collection=False):
        connections.connect(host=address, port=port)
        
        if drop_collection:
            utility.drop_collection(collection_name)
        
        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, description='ids', is_primary=True, auto_id=False),
            FieldSchema(name='text_embedding', dtype=DataType.FLOAT_VECTOR, description='embedding vectors', dim=dim),
            FieldSchema(name='position', dtype=DataType.FLOAT_VECTOR, description='position of robot', dim=3),
            FieldSchema(name='theta', dtype=DataType.FLOAT, description='rotation of robot', dim=1),
            FieldSchema(name='pose_embedding', dtype=DataType.FLOAT_VECTOR, description='position + theta', dim=4),
            FieldSchema(name='time', dtype=DataType.FLOAT_VECTOR, description='time', dim=2),
            FieldSchema(name='timestamp', dtype=DataType.DOUBLE, description='unix timestamp', dim=1),
            FieldSchema(name='caption', dtype=DataType.VARCHAR, description='caption string', max_length=3000),
            FieldSchema(name='vidpath', dtype=DataType.VARCHAR, description='video image path', max_length=200),
            FieldSchema(name='start_frame', dtype=DataType.INT64, description='video start frame', dim=1),
            FieldSchema(name='end_frame', dtype=DataType.INT64, description='end start frame', dim=1)
        ]
        schema = CollectionSchema(fields=fields, description='text image search')
        collection = Collection(name=collection_name, schema=schema)

        # create IVF_FLAT index for collection.
        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":1024}
        }
        collection.create_index(field_name="text_embedding", index_params=index_params)
        
        index_params = {
            'metric_type': 'L2',
            'index_type': "IVF_FLAT",
            'params': {"nlist": 1024}
        }
        collection.create_index(field_name="pose_embedding", index_params=index_params)


        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":2}
        }
        collection.create_index(field_name="position", index_params=index_params)

        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":2}
        }
        collection.create_index(field_name="time", index_params=index_params)

        return collection
    
    def insert(self, data_list):
        res = self.collection.insert(data_list)
        return res
    
    def reload(self):
        self.collection.load()
    
    def search_by_expr(self, expr, k:int):
        res = self.collection.query(
            limit=k, 
            expr=expr, 
            output_fields=["*"],
            consistency_level="Strong"
        )
        res = [res]
        return res
        
    def search(self, 
               embedding, 
               k:int, 
               expr:str="timestamp >= 0",
               anns_field="text_embedding"):
        param = {
                "metric_type": "L2",
                "params": {
                    "nprobe": 1024
                }
            }
        BATCH_SIZE = 2
        res = self.collection.search(
            data=[embedding],
            anns_field=anns_field,
            param=param,
            batch_size=BATCH_SIZE,
            limit=k,
            expr=expr,
            output_fields=["*"],
            consistency_level="Strong"
        )
        return res
        
        
class MilvusMemory(Memory):
    def __init__(self, 
                 db_collection_name: str, 
                 obs_savepth: str, 
                 db_ip='127.0.0.1', 
                 db_port=19530,
                 drop_collection: bool = True):
        self.last_id = 0
        self.last_seen_id = None
        
        self.obs_savepth = obs_savepth
        self.db_collection_name = db_collection_name
        self.db_ip = db_ip
        self.db_port = db_port

        # self.embedder = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')
        self.embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        self.working_memory = []
        
        self.reset(drop_collection=drop_collection)
        
    def reset(self, drop_collection:bool =True, delete_all_files:bool =False):
        if drop_collection:
            connections.connect(host=self.db_ip, port=self.db_port)
            utility.drop_collection(self.db_collection_name)
            # print("Resetting memory. We are dropping the current collection")
        if delete_all_files:
            confirm = input(f"Delete all files under {self.obs_savepth} (y/n)?  ")
            if confirm == 'y' and os.path.isdir(self.obs_savepth):
                import shutil
                shutil.rmtree(self.obs_savepth)
        self.milv_wrapper = MilvusWrapper(self.db_collection_name, self.db_ip, self.db_port, drop_collection=drop_collection)
        
        # This is mostly redundant due to milv_wrapper; Only use it to parse documents
        self.text_vector_db = Milvus(
            self.embedder,
            connection_args={"host": self.db_ip, "port": self.db_port},
            collection_name=self.db_collection_name,
            vector_field='text_embedding',
            text_field='caption',
        )
        
    def flush_and_reload(self):
        self.milv_wrapper.collection.flush()
        self.milv_wrapper.reload()
        
    def insert(self, item: MemoryItem, images=None):
        memory_dict = asdict(item)
        memory_dict['id'] = self.last_id # cannot use num_entities from db until we call `self.milv_wrapper.collection.load()`
        self.last_id += 1
        
        memory_dict['timestamp'] = memory_dict['time']
        memory_dict['time'] =  [memory_dict['time'], 0] # This is used for similarity search
        
        pose_vec = item.position + [item.theta]
        memory_dict["pose_embedding"] = pose_vec
        
        if 'text_embedding' not in memory_dict.keys():
            memory_dict["text_embedding"] = self.embedder.embed_query(memory_dict['caption'])
        
        self.milv_wrapper.insert([memory_dict])
        
        if images is not None:
            for fid, frame in zip(range(memory_dict["start_frame"], memory_dict["end_frame"]+1), images):
                savepath = os.path.join(self.obs_savepth, f"{fid:06d}.png")
                frame.save(savepath)
                
    def update(self, id: int, item: MemoryItem):
        """
        Update the memory entry with given id by deleting the old one and inserting the new item.
        NOTE: Milvus deletion is soft and asynchronous. Call `flush_and_reload()` afterwards to ensure consistency.
        """
        # Delete old record (soft delete)
        self.milv_wrapper.collection.delete(expr=f"id == {id}")
        
        # Prepare new record with the same id
        memory_dict = asdict(item)
        memory_dict['id'] = id  # Keep same ID
        memory_dict['timestamp'] = memory_dict['time']
        memory_dict['time'] = [memory_dict['time'], 0]

        if 'text_embedding' not in memory_dict or memory_dict['text_embedding'] is None:
            memory_dict['text_embedding'] = self.embedder.embed_query(memory_dict['caption'])

        # Insert new record
        self.milv_wrapper.insert([memory_dict])
                
    def search_by_text(self, query: str, k:int = 8) -> str:
        # self.milv_wrapper.reload()
        query_embedding = self.embedder.embed_query(f"Represent this sentence for searching relevant passages: {query}")
        results = self.milv_wrapper.search(query_embedding, k = k)
        docs = self._parse_query_results(results)
        docs = self._memory_to_json(docs)
        return docs
    
    def search_by_txt_and_time(self, 
                               query: str, 
                               start_time: str = None, 
                               end_time: str = None, 
                               k:int = 8) -> str:
        # self.milv_wrapper.reload()
        
        query_embedding = self.embedder.embed_query(f"Represent this sentence for searching relevant passages: {query}")
        expr_time = self._get_search_time_range(start_time, end_time)

        # TODO need to verify start_time and end_time str before calling this function
        # start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").timestamp()
        # end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").timestamp()
        # expr=f"timestamp >= {start_dt} and timestamp <= {end_dt}"
        
        results = self.milv_wrapper.search(
            embedding=query_embedding, 
            k=k, 
            expr=expr_time,
            anns_field="text_embedding")
        docs = self._parse_query_results(results)
        docs = self._memory_to_json(docs)
        return docs
    
    def search_by_position_and_time(self, 
                                    position: List[float], 
                                    start_time: str = None, 
                                    end_time: str = None, 
                                    k: int = 8) -> str:
        # Only use position for search (ignore theta)
        position_query = position
        expr_time = self._get_search_time_range(start_time, end_time)

        # start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").timestamp()
        # end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").timestamp()
        # expr = f"timestamp >= {start_dt} and timestamp <= {end_dt}"

        results = self.milv_wrapper.search(
            embedding=position_query, 
            k=k, 
            expr=expr_time,
            anns_field="position"  # <- the 3D vector field you already index
        )
        docs = self._parse_query_results(results)
        docs = self._memory_to_json(docs)
        return docs
    
    def search_by_time(self, time_str: str, k: int = 8) -> str:
        time_ts = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").timestamp()
        time_vector = [time_ts, 0.0]

        results = self.milv_wrapper.search(
            embedding=time_vector,
            k=k,
            anns_field="time",
            expr="timestamp >= 0"
        )
        docs = self._parse_query_results(results)
        return self._memory_to_json(docs)

    def search_last_k_by_text(self, is_first_time: bool, query: str, k: int = 10) -> str:
        if self.last_seen_id is not None and self.last_seen_id == 0:
            self.last_seen_id = None
            return ''
        
        query_embedding = self.embedder.embed_query(f"Represent this sentence for searching relevant passages: {query}")
        if is_first_time:
            self.milv_wrapper.reload()
            start_id, end_id = max(self.last_id-k, 0), self.last_id
        else:
            start_id, end_id = max(self.last_seen_id-k, 0), self.last_seen_id
        if end_id <= 1: # end of search
            return None
        n_retrieval = max((k // 4), 5)
        results = self.milv_wrapper.search(embedding=query_embedding, k=n_retrieval, expr=f"id >= {start_id} and id < {end_id}")
        docs = self._parse_query_results(results)
        docs = self._memory_to_json(docs)
        self.last_seen_id = start_id
        return docs
    
    def get_by_id(self, id) -> str:
        results = self.milv_wrapper.search_by_expr(expr=f"id == {id}", k=1)
        docs = self._parse_query_results(results)
        docs = self._memory_to_json(docs)
        return docs
    
    def search_all(self, query: str) -> str:
        self.milv_wrapper.reload()
        query_embedding = self.embedder.embed_query(f"Represent this sentence for searching relevant passages: {query}")
        results = self.milv_wrapper.search(embedding=query_embedding, k=100)
        docs = self._parse_query_results(results)
        docs = self._memory_to_json(docs)
        return docs
    
    def count_records_by_time(self, start_time: str = None, end_time: str = None) -> int:
        """
        Return the number of records in memory between start_time and end_time.
        """
        expr_time = self._get_search_time_range(start_time, end_time)
        
        results = self.milv_wrapper.collection.query(
            expr=expr_time,
            output_fields=["id"],  # Use minimal field for efficiency
            consistency_level="Strong"
        )
        return len(results)
    
    def get_all(self) -> str: 
        results = self.milv_wrapper.search_by_expr(expr=f"id >= 0", k=self.last_id)
        docs = self._parse_query_results(results)
        docs = self._memory_to_json(docs)
        return docs
        
    def _parse_query_results(self, results):
        ret = []
        output_fields = self.text_vector_db.fields[:]
        for result in results[0]:
            if type(result) is dict:
                data = {x: result[x] for x in output_fields}
            else:
                data = {x: result.entity.get(x) for x in output_fields}
            doc = self.text_vector_db._parse_document(data)
            if type(result) is dict:
                pair = (doc, 0.0)
            else:
                pair = (doc, result.score)
            ret.append(pair)
        return [doc for doc, _ in ret]
                
    def _memory_to_json(self, memory_list: list[MemoryItem]):
        rets = []
        for item in memory_list:
            ret = {"id": item.metadata["id"], 
                   "timestamp": float(item.metadata["timestamp"]),
                   "position": str(item.metadata["position"]), 
                   "theta": float(item.metadata["theta"]),
                   "vidpath": item.metadata["vidpath"], 
                   "start_frame": item.metadata["start_frame"], 
                   "end_frame": item.metadata["end_frame"], 
                   "text": item.page_content}
            rets.append(ret)
        return json.dumps(rets)
    
    def _get_search_time_range(self, start_time: Optional[str], end_time: Optional[str]) -> tuple[float, float]:
        """Return a Milvus expr string or None if no constraint is needed."""
        if start_time is None and end_time is None:
            return "timestamp >= 0"  # No time constraint

        stats = self.get_memory_stats()

        min_ts = stats["min_timestamp"] - 60
        max_ts = stats["max_timestamp"] + 60

        start_ts = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").timestamp() if start_time else min_ts
        end_ts = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").timestamp() if end_time else max_ts

        return f"timestamp >= {start_ts} and timestamp <= {end_ts}"
        
    def get_memory_stats(self):
        self.milv_wrapper.reload()
        
        self.milv_wrapper.collection.flush()
        # Get total number of records
        num_records = self.milv_wrapper.collection.num_entities

        # Pull all timestamps (cheap if low # of records)
        results = self.milv_wrapper.collection.query(
            expr="id >= 0",
            output_fields=["timestamp"],
            limit=num_records,  # Fetch all
            consistency_level="Strong"
        )

        timestamps = [float(r["timestamp"]) for r in results if "timestamp" in r]
        
        if not timestamps:
            return {
                "num_records": 0,
                "min_timestamp": None,
                "max_timestamp": None
            }

        return {
            "num_records": num_records,
            "min_timestamp": min(timestamps),
            "max_timestamp": max(timestamps)
        }
        
    def get_memory_stats_for_llm(self) -> str:
        stats = self.get_memory_stats()
        
        if stats["num_records"] == 0 or stats["min_timestamp"] is None or stats["max_timestamp"] is None:
            return "Memory is currently empty. No records have been stored."

        num = stats["num_records"]
        start_dt = datetime.utcfromtimestamp(stats["min_timestamp"])
        end_dt = datetime.utcfromtimestamp(stats["max_timestamp"])

        start_time_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        end_time_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
        end_date_str = end_dt.strftime("%Y-%m-%d")
        end_day_of_week = end_dt.strftime("%A")  # e.g., "Tuesday"

        return (
            f"Your memory currently contains {num} records. "
            f"The earliest memory was recorded at {start_time_str} UTC, "
            f"and the most recent memory was recorded at {end_time_str} UTC. "
            f"Today is {end_day_of_week}, {end_date_str}."
        )