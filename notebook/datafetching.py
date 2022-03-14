import json
import pandas as pd
import requests
import time
import re
def fetch(): 
    """
    
    input: 
        this function has no input. 
    outputs: 
        this function has no output. 

    this function get the data from aim technology website using post request convert the json string
    into dict then remove all commas ',' and spaces form the string. 
    lastly save the id and the text in the file. 
    then close the file.     
    
    """

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

if(__name__ == "__main__"): 
    fetch()