# Multi-dialect-Arabic
in this repo we need to classify 18 arabic dialect using Ml and DL model. 
* first we need to fetch the data form URL: using post request then convert it into dict
`
requests.post(url, data=data)
data =eval(r.text)
`
* second remove all ("," and " ") form the text and save it into a txt file.   
`  
for key, value in data.items():   
    value = re.sub(r",",'',value)  
    value = re.sub(r"\s+",' ',value)  
    x.write('%s,%s\n' % (key, value))  
`
* third starting cleaing the text from https, tags, english words, emoji, taskel, and anything. using re

* fourth starting to balace the data to perform better in the tarining and save it as final data 
* then create a Machine learning model such as Logistic regression witch not going to get a good accuray becase of the data size and features number is too big for machine learning witich lead to overfitting 


* then create a Deep learning model such as Bidirectional LSTM or LSTM. 
* lastly after training the models and save the weights. we can use this saved weights to predict the dialect with out trianing more models. 