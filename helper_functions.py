from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pickle
import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tkseem as tk
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.pipeline import Pipeline
import os
import pandas as pd 
import joblib

def split(df,target_column,labels_column):
    X = df[target_column].astype('U').values.tolist()
    y = df[labels_column].astype('U').values.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def vectorizer(train_data,max_rngram):
    vectorizer_tfidf = TfidfVectorizer(ngram_range=(1,max_rngram))
    vectorizer_tfidf.fit(train_data)
    return vectorizer_tfidf

def make_ml_pipeline(vectorizer,model,X_train,y_train):
    pipe = Pipeline([("vectorizer", vectorizer), ("classifier", model)])
    pipe.fit(X_train, y_train)
    return pipe 

def Encode_labels(labels,encode):
    if encode == 'onehot':
        encoder = OneHotEncoder(handle_unknown='ignore')
        labels = encoder.fit_transform(pd.DataFrame(labels)).toarray()
    elif encode == 'label':
        encoder = LabelEncoder()
        labels = encoder.fit_transform(labels)
        labels = to_categorical(labels)
    PROJECT_PATH = os.path.realpath(os.path.dirname(__file__))
    joblib.dump(encoder, os.path.join(PROJECT_PATH, './Models/encoder.joblib'))
    return labels

def load_encoder(path='./Models/encoder.joblib'):
    PROJECT_PATH = os.path.realpath(os.path.dirname(__file__))
    return joblib.load(os.path.join(PROJECT_PATH, path))

def save_model(model, filename, mode=None):
    """Saves a model to a .pkl file.

    Args:
        model: The model to save.
        filename: The name of the file to save the model to.

    """
    PROJECT_PATH = os.path.realpath(os.path.dirname(__file__))
    filepath = os.path.join(PROJECT_PATH, filename)

    if mode == 'DL':
            model.save(filepath)
    else:
        with open(filepath, "wb") as f:
            pickle.dump(model, f)

def load_model(filename,mode=None):
    """Loads a model from a .pkl file.

    Args:
        filename: The name of the file to load the model from.

    Returns:
        The loaded model.

    """
    PROJECT_PATH = os.path.realpath(os.path.dirname(__file__))
    filepath = os.path.join(PROJECT_PATH, filename)

    if mode == 'DL':
            model = keras.models.load_model(filepath)
    else:     
        with open(filepath, "rb") as f:
            model = pickle.load(f)

    return model

def evaluate_model(model, X_test, y_test,model_type=None):
    """ Evaluates a model on test data

    Args:
        model: The model to evaluate.
        test_data: The test data.

    Returns:
        The accuracy score and f1 macro score.

    """
    if model_type == 'DL':
            loss, accuracy = model.evaluate(X_test,y_test)
            probabilites = model.predict(X_test)
            predictions = probabilites.argmax(axis=-1)
            y_test = y_test.argmax(axis=-1)
    else:
        # Make predictions on the test data.
        predictions = model.predict(X_test)

        # Calculate the accuracy score.
        accuracy = accuracy_score(y_test, predictions)

    # Calculate the f1 macro score.
    f1_macro = f1_score(y_test, predictions, average="macro")

    # Print the results.
    print("Accuracy:", accuracy)
    print("F1 macro:", f1_macro)

    report = metrics.classification_report(y_test, predictions)
    print(report)

    return accuracy, f1_macro, report

def classify_sentence(model, sentence, model_type=None,max_seq_length=None,encoder_type=None,encoder=None):
    """Classifies a sentence using a model.

    Args:
        model: The model to use for classification.
        sentence: The sentence to classify.

    Returns:
        A tuple of the predicted label and the predicted probabilities for each class.

    """
    if model_type == 'DL' and max_seq_length != None:
            tokenized_tweet = tokenize_and_pad_tweet(sentence,max_seq_length)
            probabilities_array = model.predict(tokenized_tweet)
            predicted_probabilities = probabilities_array.tolist()[0]
            predicted_label = probabilities_array.argmax(axis=-1)
            one_hot_repr = (probabilities_array == probabilities_array.max(axis=1)[:,None]).astype(int)
            output_list = []
            if encoder_type == 'onehot':
                predicted_dialect = encoder.inverse_transform(one_hot_repr)[0]
                print(f"predicted Dialect is {predicted_dialect[0]} \n")
                for i in range(len(encoder.categories_[0])):
                    x = f"predicted Dialect {encoder.categories_[0][i]} with probability {predicted_probabilities[i]*100} \n"
                    print(x)
                    output_list.append(x)
            elif encoder_type == 'label':
                predicted_dialect = encoder.inverse_transform([predicted_label])
                print(f"predicted Dialect is {predicted_dialect[0]} \n")         
                for i in range(len(encoder.classes_)):
                    x = f"predicted Dialect {encoder.classes_[i]} with probability {predicted_probabilities[i]*100} \n"
                    print(x)
                    output_list.append(x)     
                        
    else:
        # Make a prediction.
        predicted_label = model.predict([sentence])
        print(f"predicted Dialect is {predicted_label[0]} \n")

        # Get the predicted probabilities for each class.
        predicted_probabilities = None
        output_list = []
        try:
            predicted_probabilities = model.predict_proba([sentence]).tolist()[0]
            for i in range(len(model.classes_)):
                x = f"predicted Dialect {model.classes_[i]} with probability {predicted_probabilities[i]*100} \n"
                print(x)
                output_list.append(x)

        except AttributeError:
            print("Model doesn't support predicting probabilites")

    return predicted_label, predicted_probabilities, output_list

def tokenize_and_pad_tweets(data,datatype= 'train',max_words=None,max_seq_len=None,model_path= './Models/tokenizer_model.pkl'):
    """Tokenizes tweets and pads the sequences to the length of the longest sequence in the dataset.

    Args:
        df_column (pandas.Series): A DataFrame column containing tweets.

    Returns:
        tuple:
            numpy.ndarray: An array of padded sequences.
            int: The vocabulary size.
            int: The maximum sequence length.
            Tokenizer: The tokenizer object used for the tokenization.
    """
    PROJECT_PATH = os.path.realpath(os.path.dirname(__file__))
    model_path = os.path.join(PROJECT_PATH, model_path)

    if max_words == None:
          tokenizer = tk.WordTokenizer()
    else:
          tokenizer = tk.WordTokenizer(vocab_size=max_words)
            
    if datatype == 'train':
        # Create tokenizer 
        path = os.path.join(PROJECT_PATH, './Data/tokenizer.txt')
        df = pd.DataFrame(data,columns=['tweet'])
        df.to_csv(path, sep='\n', header=False,index=False)

        tokenizer.train(path)
        
        sequences = [tokenizer.encode(sentence) for sentence in data]
        max_seq_len = max(len(seq) for seq in sequences)
        
        vocab_size = tokenizer.vocab_size
        sequences = pad_sequences(sequences, maxlen=max_seq_len,value = 0, padding='post')
         

        tokenizer.save_model(os.path.join(PROJECT_PATH,'./Models/tokenizer_model.pkl'))
    
    elif datatype == 'test' and max_seq_len != None:
        try:
            tokenizer.load_model(model_path)
            sequences = [tokenizer.encode(sentence) for sentence in data]
            vocab_size = tokenizer.vocab_size
            sequences = pad_sequences(sequences, maxlen=max_seq_len,value = 0, padding='post')
            
        except:
            print("please check if tokenizer model is passed correctly!")
    
    return sequences, vocab_size, max_seq_len, tokenizer

def tokenize_and_pad_tweet(tweet,max_seq_len=None,model_path= './Models/tokenizer_model.pkl',max_words=None):
    PROJECT_PATH = os.path.realpath(os.path.dirname(__file__))
    model_path = os.path.join(PROJECT_PATH, model_path)
    if max_words == None:
          tokenizer = tk.WordTokenizer()
    else:
          tokenizer = tk.WordTokenizer(vocab_size=max_words)

    if tweet != '' and max_seq_len != None:
        try:
            tokenizer.load_model(model_path)
            sequence = tokenizer.encode(tweet)
            vocab_size = tokenizer.vocab_size
            sequence = pad_sequences([sequence], maxlen=max_seq_len,value = 0, padding='post')[0]
            sequence = np.expand_dims(sequence, axis=0)
        except:
            print("please check your tweet and if tokenizer model is passed correctly!")
    return sequence
