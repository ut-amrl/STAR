import os
from dataclasses import dataclass, asdict
import inspect 
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
    
class MilvusVideoWrapper:

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
        
        
class MilvusVideoMemory(Memory):
    def __init__(self, db_collection_name: str, db_ip='127.0.0.1', db_port=19530):
        self.last_id = 0
        self.db_collection_name = db_collection_name
        self.db_ip = db_ip
        self.db_port = db_port

        self.embedder = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')
        self.working_memory = []
        self.reset(drop_collection=False)
        
    def reset(self, drop_collection=True):
        if drop_collection:
            print("Resetting memory. We are dropping the current collection")
        self.milv_wrapper = MilvusVideoWrapper(self.db_collection_name, self.db_ip, self.db_port, drop_collection=drop_collection)
        
    def insert(self, item: MemoryItem):
        memory_dict = asdict(item)
        memory_dict['id'] = self.last_id # cannot use num_entities from db until we call `self.milv_wrapper.collection.load()`
        self.last_id += 1
        
        memory_dict['timestamp'] = memory_dict['time']
        memory_dict['time'] =  [memory_dict['time'], 0] # This is used for similarity search
        
        if 'text_embedding' not in memory_dict.keys():
            memory_dict["text_embedding"] = self.embedder.embed_query(memory_dict['caption'])
        
        self.milv_wrapper.insert([memory_dict])