import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from flask import Flask,render_template,request

app=Flask(__name__)
df=pd.read_csv('iphone.csv')
df.isna().sum()
X=df.drop(['Unnamed: 0','Purchase Iphone'],axis=1)
y=df['Purchase Iphone']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model=GaussianNB()
model.fit(X_train,y_train)


@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def pred():
    float_features=[float(g) for g in request.form.values()]
    features=[np.array(float_features)]
    prediction=model.predict(features)
    if prediction==0:
        x=f"The client will not buy IPHONE,because the value of prediction is {prediction}"
    else:
        x=f"The person will buy IPHONE,because the value of prediction is {prediction}"

    return render_template('output.html',predicted_purchase=x)


if __name__=="__main__":
    app.run(debug=True)
