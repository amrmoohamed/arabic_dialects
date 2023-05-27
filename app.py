from flask import Flask, render_template, request
from helper_functions import tokenize_and_pad_tweet,classify_sentence,load_model,load_encoder
import joblib
import os 
import sys 
from io import StringIO
import tarfile


# Specify the path to the tar.xz file
tar_file_path = './Models/SVC.pkl.tar.xz'
extract_path = './Models'
    
    # Open the tar file
with tarfile.open(tar_file_path) as tar:
        # Extract the contents to the specified directory
        tar.extractall(path=extract_path)

app = Flask(__name__)

PROJECT_PATH = os.path.realpath(os.path.dirname(__file__))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/classify', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form["Review"]
        selcted = request.form["model"]
        if selcted=="ML":
            model = load_model("./Models/SVC.pkl")
            predicted_label, predicted_probabilities,output_list = classify_sentence(model, review)
            result = predicted_label
            return render_template('result.html',prediction = result[0])
        else:
            max_seq_length = 72
            model = load_model("./Models/LSTM", 'DL')
            encoder = load_encoder('./Models/encoder.joblib')        
            tokenized = tokenize_and_pad_tweet(review, max_seq_length, './Models/tokenizer_model.pkl')
            predicted_label, predicted_probabilities, output_list = classify_sentence(model, review, 'DL', max_seq_length, 'onehot', encoder)
            result = predicted_label
            return render_template('result.html',prediction = result)
  

if __name__ == '__main__':
    app.run()
