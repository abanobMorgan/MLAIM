# Multi-dialect-Arabic
in this repo we need to classify 18 arabic dialect using Ml and DL model. 
* first we need to fetch the data form URL: using post request 
* second remove all ("," and " ") form the text and save it into a txt file. 
* third starting cleaing the text from https, tags, english words, emoji, taskel, and anything. 
* fourth starting to balace the data to perform better in the tarining and save it as final data 
* then create a Machine learning model such as Logistic regression witch not going to get a good accuray becase of the data size and features number is too big for machine learning witich lead to overfitting 
* then create a Deep learning model such as Bidirectional LSTM or LSTM. 
* lastly after training the models and save the weights. we can use this saved weights to predict the dialect with out trianing more models. 