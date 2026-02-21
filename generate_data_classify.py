import json
import tqdm
from transformers import AutoTokenizer,AutoConfig
base_path = os.path.dirname(__file__)
with open(base_path +"accusation_id",encoding="utf-8") as f:
    accusation_id=json.load(f)
# 加载模型，读取数据，进行数据转换，罪名映射，保存数据
def convert(line):
    "{'relevant_articles': [234], 'accusation': ['故意伤害'], 'punish_of_money': 0, 'criminals': ['段某'], 'term_of_imprisonment': {'death_penalty': False, 'imprisonment': 12, 'life_imprisonment': False}}"
    data={}
    results=[]
    text= line["input"] 
    input_ids = tokenizer.encode(text,truncation=True)
    for s in line["output"]["罪名"]:    
        data={"input":input_ids,"output":accusation_id[s]} 
        results.append(json.dumps(data,ensure_ascii=False))
    return results

#分词器一定要和模型配套
model_name = "Qwen/Qwen3-8B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
data_type="test"
with open(base_path + "data/{}_data.jsonl".format(data_type),encoding="utf-8") as f:
    lines=[ convert(json.loads(line.strip())) for line in tqdm.tqdm(f.readlines())]
lines=[ss for s in lines for ss in s]
with open(base_path + "data/{}_data_classify_token".format(data_type),"w",encoding="utf-8") as f:
    f.writelines("\n".join(lines))
 