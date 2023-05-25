from sklearn.svm import LinearSVC
from data_cleaning import transform_data
from helper_functions import split
from helper_functions import vectorizer
from helper_functions import make_ml_pipeline
from helper_functions import evaluate_model
from helper_functions import save_model,load_model

file_path = "./Data/dialects_database.db"
df = transform_data(file_path)
X_train, X_test, y_train, y_test = split(df,'clean_tweet','dialect')
vectorizer = vectorizer(X_train,3)
SVC = LinearSVC()
pipeline = make_ml_pipeline(vectorizer,SVC,X_train,y_train)

accuracy, f1_macro, report = evaluate_model(pipeline, X_test, y_test)

save_model(pipeline,'./Models/SVC.pkl')

