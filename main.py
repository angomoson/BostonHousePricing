import pickle
from flask import Flask, request, app, jsonify, url_for, render_template

import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("catboost_model.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api", methods = ['POST'])
def predict_ap():
    data = request.json['data']
    new_data = np.array(list(data.values())).reshape(1,12)
    n_data = pd.DataFrame(new_data, columns = ["area","bedrooms","bathrooms","stories","mainroad","guestroom","basement","hotwaterheating","airconditioning","parking","prefarea","furnishingstatus"])
    new_pred = model.predict(n_data)
    print(new_pred)
    return jsonify(new_pred[0])

if __name__ == "__main__":
    app.run(debug=True)
