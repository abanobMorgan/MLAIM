import json
import pandas as pd 
import numpy as np 
import requests


df = pd.read_csv("doc/dialect_dataset.csv")
df.id=df.id.astype(str)
ids = df.id.array
ids= list(ids)

x = open("data/output.txt","w", encoding="UTF-8")

url = 'https://recruitment.aimtechnologies.co/ai-tasks'
for i in range(0,len(ids),1000):
    payload = ids[:i]
    data=json.dumps(payload)
    r = requests.post(url, data=data)

    x.write(r.text)
x.close()