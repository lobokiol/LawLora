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


def predict_accusation(text):
    tokens = tokenizer.encode(text, truncation=True, return_tensors="pt").to(
        model.device
    )
    result = model(tokens).logits
    p = torch.softmax(result, dim=-1)[0]
    index = int(torch.argmax(p))
    predict_label = id_accusation[index]
    return predict_label


with open("accusation_id", encoding="utf-8") as f:
    accusation_id = json.load(f)
id_accusation = dict([[int(s2), s1] for s1, s2 in accusation_id.items()])


model_name = "Qwen/Qwen3-8B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# 加载训练好的模型模型
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, trust_remote_code=True, num_labels=len(accusation_id)
).cuda()
# 训练好的lora挂在上去
model = PeftModel.from_pretrained(model, "qwen_lora_model")  # .half()
# 训练好的分类权重,替换原来的
score_weights = torch.load("score_weights.pt")
model.score.load_state_dict(score_weights)
model = model.eval() #评估模式，关闭 dropout、batchnorm 用统计参数 |
if __name__ == "__main__":
    text = "贺州市八步区人民检察院指控，2016年10月13日，被告人盘某到公安机关投案，主动交出其无证持有的砂枪一支。该枪支是其父亲生前留下，其曾用该枪到田里打过小鸟。经鉴定，该砂枪可利用火药气体发射金属弹丸，为《中华人民共和国枪支管理法》所规定的枪支。为证实指控的事实，公诉人宣读并出示了相应的证据，认为被告人盘某的行为已触犯《中华人民共和国刑法》××之规定，应以非法持有枪支罪追究其刑事责任。公诉人在发表公诉意见时认为，被告人盘某有自首情节，依法可以从轻处罚。"
    predict_label = predict_accusation(text)
    print(predict_label)
