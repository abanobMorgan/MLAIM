import json
import pandas as pd
import requests
import time
import re

df = pd.read_csv("../doc/dialect_dataset.csv")
df.id=df.id.astype(str)
ids = df.id.array
ids= list(ids)

x = open("../data/finaloutput.txt","w", encoding="UTF-8")

url = 'https://recruitment.aimtechnologies.co/ai-tasks'
for i in range(0,len(ids),1000):
    payload = ids[i:i+1000]
    data=json.dumps(payload)
    time.sleep(2)  
    print(i)
    r = requests.post(url, data=data)
    if(r.ok): 
        data =eval(r.text) 
        for key, value in data.items(): 
            value = re.sub(r",",'',value)
            value = re.sub(r"\s+",' ',value)
            x.write('%s,%s\n' % (key, value))
    else : 
        print(f"error in id = : {i}")
                

x.close()
print("done")