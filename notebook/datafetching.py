import json
import pandas as pd 
import numpy as np 
import requests
import time


df = pd.read_csv("../doc/dialect_dataset.csv")
df.id=df.id.astype(str)
ids = df.id.array
ids= list(ids)

x = open("../data/output1.txt","w", encoding="UTF-8")

url = 'https://recruitment.aimtechnologies.co/ai-tasks'
for i in range(0,len(ids),1000):
    payload = ids[i:i+1]
    data=json.dumps(payload)
    time.sleep(2)  
    r = requests.post(url, data=data)
    if(r.ok): 
        x.write(r.text[1:-1])
        

x.close()
print("done")