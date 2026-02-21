from langchain.embeddings.base import Embeddings
from typing import List
import requests
import json
from transformers import BertTokenizer, BertModel
import torch
from transformers import AutoTokenizer, AutoModel
count=0
def normal(vector):
    ss=sum([s**2 for s in vector])**0.5
    return [round(s/ss,8) for s in vector]
 
class CustomEmbeddings(Embeddings):
    def __init__(self):
        pass
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
 
        """将文档转换为嵌入向量"""
        results=[ self.embed_query(text) for text in texts]
        return results
    
    def embed_query(self, text: str) -> List[float]:
        token=tokenizer([text], padding=True, truncation=True,return_tensors='pt').to(device)
        vector=model(**token)[1].tolist()[0]
        vector=normal(vector)
 
        return vector
device = torch.device("cuda:0")
#标准bert模型，最为向量模型
# model_path="bge_recall" 因为bert的768维度，会超出范围
# model_path = "BAAI/bge-large-zh-v1.5"
# tokenizer = BertTokenizer.from_pretrained(model_path)
# model = BertModel.from_pretrained(model_path).to(device)


model_path = "Qwen/Qwen3-Embedding-0.6B"  # 或 4B、8B
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)