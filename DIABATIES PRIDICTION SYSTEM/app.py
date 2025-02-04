from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

# Define np.float as float to avoid deprecation warning
np.float = float

app = Flask(__name__)

model = pickle.load(open("Diabetes.pkl", "rb"))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    text1 = request.form['1']
    text2 = request.form['2']
    text3 = request.form['3']
    text4 = request.form['4']
    text5 = request.form['5']
    text6 = request.form['6']
    text7 = request.form['7']
    text8 = request.form['8']

    row_df = pd.DataFrame([pd.Series([text1, text2, text3, text4, text5, text6, text7, text8])])
    print(row_df)

    prediction = model.predict_proba(row_df)
    output = '{0:.{1}f}'.format(prediction[0][1] * 100, 2)
    output_str = f'{output}%'

    # Replace '3.0%' with 3.0 for comparison
    if float(output_str.rstrip('%')) > 0.5:
        return render_template('result.html', pred=f'You have a chance of having diabetes.\nProbability of having Diabetes is {output_str}')
    else:
        return render_template('result.html', pred=f'You are safe.\n Probability of having diabetes is {output_str}')
if __name__ == '__main__':
    app.run(debug=True)

