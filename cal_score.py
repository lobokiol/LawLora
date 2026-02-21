import json
# 对比模型预测结果和真实结果，计算各项指标的准确率。
# 评估指标
# | 指标 | 计算方式 |
# |------|----------|
# | 罪名准确率 | 预测罪名在真实罪名中 |
# | 罪犯准确率 | 预测罪犯 == 真实罪犯 |
# | 罚金准确率 | 预测罚金在 ±25% 范围内 |
# | 刑期准确率 | 预测刑期在 ±25% 范围内 |
def get_month(data):
    if  data["是否死刑"]:
        return 12*100
    if data["是否无期"]:
        return 12*50
    return data["有期徒刑"]

with open("test_result",encoding="utf-8") as f:
    lines=[json.loads(s.strip()) for s in f.readlines()]
std=0.25
right_charge=0
right_person=0
right_fine=0
right_time=0
count=0
for data in lines:
    try:
        fact=data["fact"]
        predict=data["predict"]
        if fact["罪名"][0] in predict["罪名"]:
            right_charge+=1
        if str(fact["犯罪嫌疑人"])==str(predict["犯罪嫌疑人"]):
            right_person+=1 
        if predict["罚金"]>=(1-std)*fact["罚金"] and predict["罚金"]<=(1+std)*fact["罚金"]:
            right_fine+=1
        fact_time=get_month(fact)
        predict_time=get_month(predict)
    
        if predict_time>=(1-std)*fact_time and predict_time<=(1+std)*fact_time:
            right_time+=1

        count+=1
    except:
        continue
print ("罪名准确率",right_charge/count)
print ("罪犯准确率",right_person/count)
print ("罚金准确率",right_fine/count)
print ("坐牢时间准确率",right_time/count)