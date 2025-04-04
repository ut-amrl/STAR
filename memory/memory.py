import os
from dataclasses import dataclass, asdict
import inspect 
from datetime import datetime
import re

from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

@dataclass
class MemoryItem:
    caption: str
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
        
    def search(self, query_embedding, k:int, expr:str="timestamp >= 0"):
        param = {
                "metric_type": "L2",
                "params": {
                    "nprobe": 1024
                }
            }
        BATCH_SIZE = 2
        res = self.collection.search(
            data=[query_embedding],
            anns_field="text_embedding",
            param=param,
            batch_size=BATCH_SIZE,
            limit=k,
            expr=expr,
            output_fields=["*"],
            consistency_level="Strong"
        )
        return res
        
        
class MilvusMemory(Memory):
    def __init__(self, db_collection_name: str, obs_savepth: str, db_ip='127.0.0.1', db_port=19530):
        self.last_id = 0
        self.last_seen_id = None
        
        self.obs_savepth = obs_savepth
        self.db_collection_name = db_collection_name
        self.db_ip = db_ip
        self.db_port = db_port

        self.embedder = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')
        self.working_memory = []
        
        self.text_vector_db = Milvus(
            self.embedder,
            connection_args={"host": self.db_ip, "port": self.db_port},
            collection_name=self.db_collection_name,
            vector_field='text_embedding',
            text_field='caption',
        )
        
        self.reset(drop_collection=False)
        
    def reset(self, drop_collection:bool =True, delete_all_files:bool =False):
        if drop_collection:
            utility.drop_collection(self.db_collection_name)
            print("Resetting memory. We are dropping the current collection")
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
        
    def insert(self, item: MemoryItem, images=None):
        memory_dict = asdict(item)
        memory_dict['id'] = self.last_id # cannot use num_entities from db until we call `self.milv_wrapper.collection.load()`
        self.last_id += 1
        
        memory_dict['timestamp'] = memory_dict['time']
        memory_dict['time'] =  [memory_dict['time'], 0] # This is used for similarity search
        
        if 'text_embedding' not in memory_dict.keys():
            memory_dict["text_embedding"] = self.embedder.embed_query(memory_dict['caption'])
        
        self.milv_wrapper.insert([memory_dict])
        
        if images is not None:
            for fid, frame in zip(range(memory_dict["start_frame"], memory_dict["end_frame"]+1), images):
                savepath = os.path.join(self.obs_savepth, f"{fid:06d}.png")
                frame.save(savepath)
                
    def search_by_text(self, query: str, k:int = 8) -> str:
        self.milv_wrapper.reload()
        query_embedding = self.embedder.embed_query(query)
        results = self.milv_wrapper.search(query_embedding, k = k)
        docs = self._parse_query_results(results)
        docs = self._memory_to_json(docs)
        return docs
    
    def search_by_txt_and_time(self, query: str, start_time: str, end_time: str, k:int = 8) -> str:
        self.milv_wrapper.reload()
        query_embedding = self.embedder.embed_query(query)
        # TODO need to verify start_time and end_time str before calling this function
        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").timestamp()
        end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").timestamp()
        expr=f"timestamp >= {start_dt} and timestamp <= {end_dt}"
        results = self.milv_wrapper.search(query_embedding=query_embedding, k=k, expr=expr)
        docs = self._parse_query_results(results)
        docs = self._memory_to_json(docs)
        return docs

    def search_last_k_by_text(self, is_first_time: bool, query: str, k: int = 10) -> str:
        if self.last_seen_id is not None and self.last_seen_id == 0:
            self.last_seen_id = None
            return ''
        
        query_embedding = self.embedder.embed_query(query)
        if is_first_time:
            self.milv_wrapper.reload()
            start_id, end_id = max(self.last_id-k, 0), self.last_id
        else:
            start_id, end_id = max(self.last_seen_id-k, 0), self.last_seen_id
        if end_id <= 1: # end of search
            return None
        n_retrieval = max((k // 4), 5)
        results = self.milv_wrapper.search(query_embedding=query_embedding, k=n_retrieval, expr=f"id >= {start_id} and id < {end_id}")
        docs = self._parse_query_results(results)
        docs = self._memory_to_json(docs)
        self.last_seen_id = start_id
        return docs
    
    def get_by_id(self, id) -> str:
        results = self.milv_wrapper.search_by_expr(expr=f"id == {id}", k=1)
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
                   "position": str(item.metadata["position"]), 
                   "vidpath": item.metadata["vidpath"], 
                   "start_frame": item.metadata["start_frame"], 
                   "end_frame": item.metadata["end_frame"], 
                   "text": item.page_content}
            rets.append(ret)
        import json
        return json.dumps(rets)