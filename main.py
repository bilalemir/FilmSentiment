from flask import Flask,render_template,url_for,request
import numpy as np
import pickle
import flasgger
from flasgger import Swagger


app=Flask(__name__)
Swagger(app)

mnb = pickle.load(open('logistic_imdb.pkl','rb'))
countVect = pickle.load(open('tidf.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        Reviews = request.form['a']
        data = [Reviews]
        vect = countVect.transform(data).toarray()
        my_prediction = mnb.predict(vect)
    return render_template('after.html',prediction = my_prediction)



if __name__ == '__main__':
    app.run(debug=True)


    











