from flask import Flask, render_template, request
from src.train import train
from src.predict import predict
import time


app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def train_model():
    if request.method == 'GET':
        return render_template('train.html')
    if request.method == 'POST':
        train(False)
        return render_template('train.html', training='~~Training Done~~')
    
        

@app.route("/predict", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('predict.html')
    if request.method == 'POST':
        result = predict()
        return render_template('predict.html', results = result[0])


if __name__=="__main__":
    app.run(port=5000, debug=True)
