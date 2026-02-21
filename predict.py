import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from json_output import parser
import torch
from tqdm import tqdm
import pickle
from langchain_openai import ChatOpenAI
from langchain.output_parsers import OutputFixingParser
import random
from predict_accusation import *
import os
def accusation_json2str(data):
    accusation_list=data["罪名"]
    result="判决结果："
    result+="罪名:"+"，".join(accusation_list)+"，"
    result+="罚金:"+str(data["罚金"])+"元，"
    result+="罪犯:"+",".join(data["犯罪嫌疑人"])+"，"
    if data["是否死刑"]:
        result+="刑期：死刑"
    elif data["是否无期"]:
        result+="刑期：无期徒刑"
    else:
        result+="刑期：有期徒刑"+str(data["有期徒刑"])+"个月"
    return result
llm = ChatOpenAI(
    model="qwen2.5-32b-instruct",
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv("api_key")
)

llm_think = ChatOpenAI(
    model="qwen3-30b-a3b",
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv("api_key")
)

import os
import dashscope

def think(prompt,stream=True):
    messages = [
        {'role': 'system', 'content': '你是一个资深的法律顾问'},
        {'role': 'user', 'content': prompt}
        ]
    response = dashscope.Generation.call(
        api_key=os.getenv("api_key"),
        model="qwen3-30b-a3b", 
        messages=messages,
        result_format='message',
        stream=stream,
        enable_thinking=False
    )
    return response



def parser_documents(documents):
    result=""
    for i,doc in enumerate(documents):
        text=doc.page_content+"判决结果："+doc.metadata["result"]+"\n\n"
        result+="案例"+str(i)+":"+text
    return result
def merge_result(all_result):
    death=False
    imprisonment=False
    for result in all_result:
        death=death or result["是否死刑"]
        imprisonment=imprisonment or result["是否无期"]
    month=[s["有期徒刑"] for s in all_result if not s["是否死刑"] and not s["是否无期"]]
    money=[s["罚金"] for s in all_result]
    if len(month)==0:
        month=0
    else:
        month=sum(month)/len(month)
    money=sum(money)/len(money)
    all_result[0]["有期徒刑"]=month
    all_result[0]["是否死刑"]=death
    all_result[0]["是否无期"]=imprisonment
    all_result[0]["罚金"]=money
    return all_result[0]

def predict(text,num=3):
    #text案情描述
    #罪名预测
    predict_label=predict_accusation(text)
    #检索卷宗
    similar_docs = db.similarity_search(text, k=10,filter={"category": predict_label})[0:3]
    #类似案件的案情描述
    case_list=parser_documents(similar_docs)
 
    all_result=[]   
    for _ in range(0,num):
        response=llm.invoke(case_list+"根据上述案例，对下面的案件做出判决"+text+parser.get_format_instructions())
        result=fix_parser.parse(response.content)
        result["罪名"]=[predict_label]
        all_result.append(result)
 
    #对刑期和罚金等数字 做平均
    result=merge_result(all_result)
    return result,case_list

with open("law_db","rb") as f:
    db=pickle.load(f)
fix_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

if __name__ == "__main__":
   

    text="公诉机关起诉书指控并审理经查明：2017年8月24日10时许，被告人刘某在本市丰台区木樨园长途客运站出站闸机处不配合检查工作，用随身携带的小推车冲撞民警、用雨伞击打民警头部，在民警及辅警对其进行控制时，对民警及辅警进行辱骂、抓挠、撕扯，造成民警张1某右前臂、双下肢多处挫伤，辅警龙某右上肢皮肤挫伤。经鉴定，张1某、龙某二人的损伤程度均为轻微伤。\r\n公诉机关建议判处被告人刘某××至一年。被告人刘某对指控事实、罪名及量刑建议没有异议且签字具结，在开庭审理过程中亦无异议。\r\n"
    result_json,case_list=predict(text)
    print ("相似案件",case_list)
    result=accusation_json2str(result_json)
    print ("判处结果",result)
    text2=text+result+"请根据判处结果，给出合理的司法解释"
    
    response=think(text2,stream=False)
    
    print ("司法解释")
    print (response["output"]["choices"][0]["message"]["content"])
 