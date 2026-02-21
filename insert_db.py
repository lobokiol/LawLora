
from langchain_community.vectorstores import FAISS
 
import pickle
import json
from langchain.docstore.document import Document
import random
from Embedding import  CustomEmbeddings
import predict
embeddings =CustomEmbeddings()
with open("data\\train_data.jsonl",encoding="utf-8") as f:
    lines1=[json.loads(s.strip()) for s in f.readlines()]
with open("data\\rest_data.jsonl",encoding="utf-8") as f:
    lines2=[json.loads(s.strip()) for s in f.readlines()]
lines=lines1+lines2
documents=[]
# {"input": "经审理查明：2017年5月12日，被告人王某驾驶车牌号为豫Ｃ×××××的白色小型越野车，携带捕兽夹到洛宁县底张乡庙沟村西侧沟内猎捕野生动物，共猎获果子狸2只（均为活体）、野猫2只（一只活体、一只死体）。2017年5月13日，被告人王某携带狩猎工具及猎获物驾车返回三门峡途经洛宁县长水镇高速公路入口时，被洛宁县公安局交通警察大队执勤民警当场查获。后洛宁县森林公安局民警将死亡的1只野猫予以掩埋，将另外1只野猫及2只果子狸予以放生。上述事实，被告人王某在审理中亦无异议，且有被告人王某的供述，证人耿某的证言，辨认笔录，现场勘验笔录、示意图及照片，提取痕迹、物证登记表，扣押决定书及清单，发还清单及销毁物品、文件清单，洛宁县森林公安局出具的情况说明，被告人王某户籍证明及无违法犯罪证明等证据，经当庭举证、质证证实，足以认定。", "output": {"是否死刑": false, "有期徒刑": 0, "是否无期": false, "罪名": ["非法狩猎"], "罚金": 10000, "犯罪嫌疑人": ["王某"]}}



for data in lines:
    text=data["input"]
    if len(text)>2000:
        continue
    result=predict.accusation_json2str(data["output"])
 
    #对应的罪名
    accusation_list=data["output"]["罪名"]
    #page_content:案情描述  建立向量
    #result：判决结果
    #category 罪名
    for accusation in accusation_list:
        documents.append(Document(
            page_content=text,
            #元数据，不参与检索
            metadata={"result": result, "category": accusation}
        ))
print ("开始建索引")
# 先归一化，将向量归一化
db = FAISS.from_documents(documents, embeddings)
with open("law_db","wb") as f:
    pickle.dump(db,f)

# with open("law_db","rb") as f:
#     db=pickle.load(f)

# # 现在可以进行相似性搜索
# query = "人工智能有哪些应用？"
# similar_docs = db.similarity_search(query, k=10,filter={"category": "诈骗"},)
# print (similar_docs)