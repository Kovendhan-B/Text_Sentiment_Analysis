from flask import Flask,request,app,jsonify,url_for,render_template
import pickle
import pandas
import numpy

app=Flask(__name__)
model=pickle.load(open('lr_model.pkl','rb'))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    vect_data=vectorizer.transform(data)
    output=model.predict(vect_data)
    print(output[0])
    return jsonify(int(output[0]))

@app.route('/predict',methods=['POST'])
def predict():
    text = [request.form['text']]
    print(f"Received input: {text}")
    vect_data = vectorizer.transform(text)
    print(f"Vectorized input: {vect_data}")
    output = model.predict(vect_data)
    print(f"Model output: {output}")
    sentiment = "Negative" if output[0] == -1 else "Neutral" if output[0] == 0 else "Positive"
    return render_template("Home.html", prediction_text="Sentiment of text is {}".format(sentiment))

if __name__=="__main__":
    app.run(debug=True)
    