# Arabic Dialect Classification

## _Arabic tweets dialect_


Many countries speak Arabic; however, each country has its own dialect, the aim of this Project is to
build a model that predicts the dialect given the text.



## ✨Dataset✨

addressed by Qatar Computing Research Institute, moreover, they published a paper, feel free to get more insights from it,
>   https://arxiv.org/pdf/2005.06557.pdf
## Repo Structure

- Data: This folder  contains the data.db used for the project.

- Models: This folder  contains trained machine learning / deep learning models used in the project.

- templates: This folder  contains HTML templates used for rendering web pages in the application.

- Data_Fetching.py: This file  contains code for fetching data from external sources.

- NLP_Final.ipynb: This file is  a Jupyter Notebook containing the final version of the NLP (natural language processing) code used in the project analysis.

- app.py: This file is  the main script for running the web application.

- data_cleaning.py: This file  contains code for cleaning and preprocessing the raw data before it is used in the models.

- dl_model.py: This file  contains code for defining and training a deep learning model for the project.

- helper_functions.py: This file  contains helper functions used throughout the project.

- ml_model.py: This file  contains code for defining and training a traditional machine learning model for the project.

- requirements.txt: This file  contains a list of all the dependencies required to run the project.



## Installation

To install the dependencies listed in a requirements.txt file, you can use the pip package manager in the command line. Here are the steps:

1. Open a terminal or command prompt.
2. Navigate to the directory where the requirements.txt file is located.
3. Run the following command to install the dependencies:

```sh
pip install -r requirements.txt
```

For run :

```sh
python3 scr/app.py
```

## Models Comparison

### We tried:-
> 2 ML Models
- Naive bayern
- Linear SVC
> 2 Dl Models
 - Vanilla RNN
 - LSTM

| Model | F1_macro_val | accuracy_val|
| :---         |     :---:      |          ---: |
| Naive bayern   | 51     | 67    |
| Linear SVC    | 80      | 84      |
| Vanilla RNN     | 50       | 52      |
| LSTM     | 75       | 80     |

## Team:-
1. Amr Mohamed Abd ALbadee
2. Amira Hesham Mo’men
3. Ibrahim Ayman Abu-Shara
4. Mostafa Mohammed ali
5. Ziad Mahmoud Mohammed

