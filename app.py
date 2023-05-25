from flask import Flask, render_template, request
from helper_functions import tokenize_and_pad_tweet,classify_sentence,load_model,load_encoder
import joblib
import os 
import sys 
from io import StringIO
#return render_template('./templates/index.html')

app = Flask(__name__)

PROJECT_PATH = os.path.realpath(os.path.dirname(__file__))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    sentence = request.form['sentence']
    model_type = request.form['model']

    if model_type == 'ml':
        model = load_model("./Models/SVC.pkl")
        predicted_label, predicted_probabilities,output_list = classify_sentence(model, sentence)
        result = predicted_label
        return render_template('index.html', result=result)

    elif model_type == 'dl':
        max_seq_length = 61
        model = load_model("./Models/LSTM", 'DL')
        encoder = load_encoder('./Models/encoder.joblib')        
        tokenized = tokenize_and_pad_tweet(sentence, max_seq_length, './Data/tokenizer_model.pkl')
        predicted_label, predicted_probabilities, output_list = classify_sentence(model, sentence, 'DL', max_seq_length, 'onehot', encoder)
        result = predicted_label
        return render_template('index.html', result=result)

    

if __name__ == '__main__':
    app.run()
