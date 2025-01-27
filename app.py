from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app=Flask(__name__)

classifier=pickle.load(open("Liver-Disease-Prediction/liver_model_prediction.pkl","rb"))


@app.route('/')

@app.route('/liver')
def home():
    return render_template("liver.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method=='POST':
        Total_Bilirubin=float(request.form["TotalBilirubin"])
        Direct_Bilirubin=float(request.form["DirectBilirubin"])
        Alkaline_Phosphotase=int(request.form["AlkalinePhosphotase"])
        Alamine_Aminotransferase=int(request.form["AlamineAminotransferase"])
        Total_Protiens=float(request.form["TotalProtiens"])
        Albumin=float(request.form["Albumin"])
        Ratio=float(request.form["Ratio"])

        input_data=(Total_Bilirubin,Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase,Total_Protiens, Albumin, Ratio)
        input_data_as_numpy_array=np.asarray(input_data)
        input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)
        prediction=classifier.predict(input_data_reshaped)

        if(prediction[0]==1):
            result="Sorry, you have chances of getting the disease. Please consult the doctor immediately."
        else:
            result="No need to fear. You have no dangerous symptoms of the disease."

        return render_template("result.html", result=result)
    

if __name__=='__main__':
    app.run(debug=True)
