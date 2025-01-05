from flask import Flask,request,app,jsonify,url_for,render_template
import pickle
import pandas
import numpy

app=Flask(__name__)
model=pickle.load(open('lr_model.pkl','rb'))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.json['data']
    print(data)
    vect_data=vectorizer.transform(data)
    output=model.predict(vect_data)
    print(output[0])
    return jsonify(int(output[0]))

if __name__=="__main__":
    app.run(debug=True)
    