from flask import Flask, render_template, request
# import jsonify
import requests
import pickle
import numpy as np
import pandas as pd
import sklearn
from xgboost import XGBRegressor

app = Flask("__name__")



@app.route('/',methods=['GET'])
def Home():
    return render_template('home.html')

@app.route("/predict", methods=['POST'])
def predict():
    model = pickle.load(open("house_price_model_xgboost.pkl", "rb"))

    if request.method == 'POST':
        CRIM = float(request.form['CRIM'])

        ZN=float(request.form['ZN'])

        INDUS=float(request.form['INDUS'])

        CHAS=float(request.form['CHAS']) 

        NOX=float(request.form['NOX'])

        RM=float(request.form['RM'])

        AGE=float(request.form['AGE'])

        DIS=float(request.form['DIS'])

        RAD=float(request.form['RAD'])

        TAX=float(request.form['TAX'])

        PTRATIO=float(request.form['PTRATIO'])

        B=float(request.form['B'])

        LSTAT=float(request.form['LSTAT'])

        # prediction=model.predict([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])

        data = [[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]]
        
        # Create the pandas DataFrame 
        new_df = pd.DataFrame(data, columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
        prediction = model.predict(new_df)
        output=round(prediction[0],2)

        return render_template('home.html', output1=output, CRIM=request.form['CRIM'], ZN=request.form['ZN'], INDUS=request.form['INDUS'], CHAS=request.form['CHAS'], NOX=request.form['NOX'], RM=request.form['RM'], AGE=request.form['AGE'], DIS=request.form['DIS'], RAD=request.form['RAD'], TAX=request.form['TAX'], PTRATIO=request.form['PTRATIO'], B=request.form['B'], LSTAT=request.form['LSTAT'])

    else:
        return render_template('home.html')
        

if __name__=="__main__":
    app.run(debug=True)

