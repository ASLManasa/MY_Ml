from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(name)

model = pickle.load(open("model.pkl","rb"))

@app.route('/')
def forest_fire():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict_data():
    values = [int(x) for x in request.form.values()]
    values = [np.array(values)]
    predict = model.predict(values)
    if predict[0] == 0:
        return render_template('index.html', pred="Your Forest will not catch fire")
    else:
        return render_template('index.html', pred="Your Forest will catch fire")


if name == 'main':
    app.run(debug=True)