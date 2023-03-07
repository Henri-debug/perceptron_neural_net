import dill
import pandas as pd
import time
from flask import Flask, request, make_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = dill.load(open('perceptron_motor.pkl', 'rb'))


@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json()
    x1 = data['x1']
    x2 = data['x2']
    
    print('x1: '+ str(x1) + ' x2: '+str(x2))

    y = model.predict([x1,x2])

    return str(y)


if __name__ == '__main__':
    app.run(port=8080)
