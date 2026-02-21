# 导入所需的库和模块

 
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoTokenizer,AutoConfig,AutoModelForSequenceClassification
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field

from json_output import parser 
import json
import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

#m每次只加载一小部分数据，放入模型训练
def data_collator(features) -> dict:
    '''
    batch = [
    {"input": [1,2,3], "output": 0},      # 长度3
    {"input": [4,5], "output": 1},        # 长度2
    {"input": [6,7,8,9], "output": 2},   # 长度4
]
               │
               ▼ padding后
input_ids = [
    [0, 0, 1, 2, 3],   # 补2个0
    [0, 0, 0, 4, 5],   # 补3个0
    [0, 6, 7, 8, 9],   # 补1个0
]
labels = [0, 1, 2]
对于自回归模型（如Qwen），通常使用pre-padding，因为：
1. 最后一个token包含了之前所有信息
2. attention mask可以正确区分有效token和padding
    '''
    max_seq_length=-1
    input_list = []
    labels_list = []
    # 找到最长的序列长度，以便进行padding
    max_seq_length=-1
    for feature in  features:
        ids= feature["input_ids"]["input"]
        if len(ids)>1000:
            continue
        max_seq_length=max(len(ids),max_seq_length)
    #max_seq_length   batch size 里面最长的数据
    #   填充到相同长度
    for feature in  features:
        ids= feature["input_ids"]["input"]
        if len(ids)>1000:
            continue
        label=feature["input_ids"]["output"]
        #数据长度的补齐
        ids = (max_seq_length-len(ids))*[0]+ids
        input_list.append(torch.LongTensor(ids))
        labels_list.append(torch.LongTensor([label]))
    # 堆叠成batch tesnsor 
    input_ids = torch.stack(input_list)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }
# 获取当前脚本所在的目录
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "accusation_id")
with open(file_path,encoding="utf-8") as f:
    accusation_id=json.load(f)
model_name = "Qwen/Qwen3-8B-Base"
#model_name="Qwen/qwen2.5-1.5B-Instruct"
#model_name="Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, load_in_8bit=False, trust_remote_code=True, device_map="auto",num_labels=len(accusation_id))
# 使用 4-bit 量化的模型加载配置
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    load_in_4bit=True,                    # 启用 4-bit 量化
    bnb_4bit_quant_type="nf4",           # 嵌套量化类型
    bnb_4bit_compute_dtype=torch.float16, # 计算数据类型
    bnb_4bit_use_double_quant=True,      # 使用双量化
    trust_remote_code=True, 
    device_map="auto",
    num_labels=len(accusation_id)
)


config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, device_map='auto')
 
# LoRA配置
lora_config = LoraConfig(
    r=16,                      # LoRA矩阵的秩
    lora_alpha=16,            # LoRA缩放因子
    target_modules=["q_proj", "v_proj"],  # 要应用LoRA的模块
    lora_dropout=0.1,         # Dropout概率
    bias="none",              # 是否训练偏置
)
 
# 训练参数配置
training_args = TrainingArguments(
    output_dir="./qwen_lora_output",        # 输出目录
    learning_rate=2e-4,                     # 学习率
    per_device_train_batch_size=4,          # 训练批次大小
    gradient_accumulation_steps=4,          # 梯度累积步数
    num_train_epochs=3,                     # 训练轮次
    weight_decay=0.01,                      # 权重衰减
    logging_dir="./logs",                   # 日志目录
    logging_steps=10,                       # 日志记录频率
    save_strategy="steps",                  # 保存策略
    save_steps=30,                        # 保存频率
    fp16=True,                            # 使用混合精度训练
    push_to_hub=False,                      # 是否推送到Hugging Face Hub
    report_to="tensorboard",
    save_total_limit=2,
)
 # 简单说：告诉模型哪些是"补齐的位置"，别把那些当内容学习。
model.config.pad_token_id=tokenizer.pad_token_id

file_path = os.path.join(base_path, "data/train_data_classify_token")
with open(file_path,encoding="utf-8")  as f:
    dataset=[ {"input_ids":json.loads(s)} for s in f.readlines()]
model = get_peft_model(model, lora_config) 
#挂在了lora的model
print (model)
# trainable params: 4,194,304
# all params: 8,033,830,720
# trainable%: 0.05%  ← 只训练0.05%的参数！
#使用lora的时候，它会默认把原模型的所有参数都冻结住
#把被误冻结的分类也就是score层，重新恢复
# 输入 → Qwen3-8B(冻结) + LoRA(微调) → 新表示(3584维)
#                                        ↓
#                                       必须通过 score层(线性变换) → 183类概率
for name, param in model.score.named_parameters():
    param.requires_grad=True
 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# 启动训练
trainer.train()

# # 保存LoRA模型
model.save_pretrained("./qwen_lora_model")
#保存分类层
torch.save(model.score.state_dict(), "score_weights.pt")