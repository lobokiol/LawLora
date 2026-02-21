from langchain_core.output_parsers import JsonOutputParser
 
from pydantic import BaseModel, Field
#我们需要的格式
#description写清楚
#  'output': "{'relevant_articles': [234], 'accusation': ['故意伤害'], 'punish_of_money': 0, 'criminals': ['段某'], 'term_of_imprisonment': {'death_penalty': False, 'imprisonment': 12, 'life_imprisonment': False}}"
 

class Crime(BaseModel):
    罪名: list[str] = Field(description="罪名")
    罚金:int=Field(description="罚款金额")
    犯罪嫌疑人:list[str]=Field(description="犯罪分子")
    是否死刑:bool=Field(description="是否死刑")
    有期徒刑:int=Field(description="刑期")
    是否无期:bool=Field(description="是否无期徒刑")
 
#主要是提供提示词
parser = JsonOutputParser(pydantic_object=Crime)
#print (parser.get_format_instructions())