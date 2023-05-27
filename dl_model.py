from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Bidirectional, LSTM
from helper_functions import evaluate_model
from helper_functions import tokenize_and_pad_tweets
from data_cleaning import transform_data
from helper_functions import split
from helper_functions import Encode_labels
from helper_functions import save_model,load_model
import nltk


def create_sequential_model(units,embedding_dim,vocab_size,max_seq_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length))
    model.add(Bidirectional(LSTM(units=units)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(units=5, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model 

def fit_evaluate_model(model,train_data,train_labels,test_data,test_labels,epochs,batch_size):
    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size,
                    validation_data=(test_data,test_labels))
    accuracy, f1_macro, report = evaluate_model(model, test_data, test_labels,'DL')
    save_model(model,'./Models/LSTM','DL')
    return history,accuracy,f1_macro,report

file_path = "./Data/dialects_database.db"
df = transform_data(file_path,"dl")
X_train, X_test, y_train, y_test = split(df,'clean_tweet','dialect')
train_padded_sequences, vocab_size, max_seq_length, tokenizer = tokenize_and_pad_tweets(X_train,'train')
test_padded_sequences, vocab_size, max_seq_length, tokenizer = tokenize_and_pad_tweets(X_test,'test',None,max_seq_length,'./Models/tokenizer_model.pkl')
train_labels = Encode_labels(y_train,'onehot')
test_labels = Encode_labels(y_test,'onehot')
model = create_sequential_model(64,64,vocab_size,max_seq_length)
history,accuracy,f1_macro,report = fit_evaluate_model(model,train_padded_sequences,train_labels,test_padded_sequences,test_labels,10,64)

