import boto3
from flask import Flask, request, jsonify
import joblib


app= Flask(__name__)
s3 = boto3.resource('s3')
s3.Bucket('germancreditjdq').download_file('model.joblib', 'model.joblib')
model = joblib.load("model.joblib")

@app.route("/")
def index():
    return "Hi Flask"

@app.route("/predict", methods=["POST"])
def predict():
    request_data = request.get_json()
    age=request_data["age"]
    credit_amount= request_data["credit_amount"]
    duration= request_data["duration"]
    purpose= request_data["purpose"]
    housing= request_data["housing"]
    sex= request_data["sex"]
    prediction= model.predict([[age, credit_amount, duration, sex, purpose, housing]])
    probability= model.predict_proba([[age, credit_amount, duration, sex, purpose, housing]])
    return jsonify({"prediction": prediction.tolist(), "probabilidad": probability.tolist()})
    #f"{age}, {credit_amount}, {duration}, {purpose}, {housing}, {sex}"


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(host='0.0.0.0')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
