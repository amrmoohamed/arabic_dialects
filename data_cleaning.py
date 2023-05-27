import nltk
from nltk.tokenize import TweetTokenizer
import re
import string
import emoji
from pyarabic.araby import strip_tashkeel
from pyarabic.araby import normalize_ligature
from tqdm import tqdm
from Data_Fetching import fetch
import numpy as np
import os

nltk.download("stopwords")
STOP_WORDS = set(nltk.corpus.stopwords.words("arabic"))

def tokenize_tweet(tweet):
    # create a TweetTokenizer object
    tknzr = TweetTokenizer()
    # tokenize the tweet
    tokens = tknzr.tokenize(tweet)
    return tokens

def remove_extra_spaces(words):
    """Removes extra whitespaces at the beginning and at the end of each word in a list"""
    cleaned_words = []
    for word in words:
        cleaned_word = ' '.join(word.split()).strip()
        cleaned_words.append(cleaned_word)
    return cleaned_words

def remove_urls(lst):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return [re.sub(pattern, '', item).strip() for item in lst if re.sub(pattern, '', item).strip() != '']

def remove_user_mentions(words):
    """Removes user mentions (@user) from a list of words"""
    cleaned_words = []
    for word in words:
        if not word.startswith('@'):
            cleaned_words.append(word)
    return cleaned_words

def remove_punctuation(lst):
    """Removes punctuation from a list of strings, including single punctuation characters"""
    translator = str.maketrans('', '', string.punctuation+'؟')
    result = []
    for item in lst:
        # Remove all punctuation characters
        item = item.translate(translator)
        # Remove any remaining single punctuation characters
        if item != '':
          result.append(item)
    return result

def remove_numbers(lst):
    """Removes numbers from a list of strings"""
    pattern = re.compile(r'\d+')
    return [re.sub(pattern, '', item) for item in lst if re.sub(pattern, '', item).strip() != '']

def remove_emojis(words):
    """Removes emojis from a list of words"""
    cleaned_words = []
    for word in words:
        cleaned_word = ''.join(c for c in word if c not in emoji.EMOJI_DATA)
        if cleaned_word != '':
            cleaned_words.append(cleaned_word)
    return cleaned_words

def remove_foreign_language(lst):
    pattern = re.compile(r'[^\u0600-\u06ff]+')
    return [re.sub(pattern, "", item) for item in lst if re.sub(pattern, "", item) != '']

def remove_tashkeel(lst):
    return [normalize_ligature(strip_tashkeel(word)) for word in lst]

def remove_repeated_chars(lst):
    pattern = re.compile(r"(\w)\1{2,}")
    return [re.sub(pattern, r"\1\1", item).strip() for item in lst if re.sub(pattern, '', item).strip() != '']

def remove_stop_words(lst):
    
    result = []
    for word in lst:
        if word not in STOP_WORDS:
            result.append(word)
    return result

def form_sentence(words):
    """Forms a sentence from a list of words"""
    sentence = ' '.join(words)
    return sentence

def clean_tweet(tweet,mode="ml"):
    """
    A function to clean a single tweet.
    """
    if mode=="ml":
        #tokenize tweet 
        words = tokenize_tweet(tweet)
        #remove extra white-spaces
        words = remove_extra_spaces(words)
        #remove urls 
        words = remove_urls(words)
        #remove user mentions 
        words = remove_user_mentions(words)
        #remove punctiation
        words = remove_punctuation(words)
        #remove numbers
        words = remove_numbers(words)
        #remove emojis
        words = remove_emojis(words)
        #remove non-arabic charachters
        words = remove_foreign_language(words)
        #remove tashkeel 
        words = remove_tashkeel(words)
        #remove repeated charachters
        words = remove_repeated_chars(words)
        #remove stop words 
        words = remove_stop_words(words)
        #form a new sentence
        sentence = form_sentence(words)
    else:
        words = tokenize_tweet(tweet)
        #remove extra white-spaces
        words = remove_extra_spaces(words)
        #remove urls 
        words = remove_urls(words)
        #remove user mentions 
        words = remove_user_mentions(words)
        #remove punctiation
        words = remove_punctuation(words)
        #remove numbers
        words = remove_numbers(words)
        #remove emojis
        words = remove_emojis(words)
        #remove non-arabic charachters
        words = remove_foreign_language(words)
        #form a new sentence
        sentence = form_sentence(words)
    return sentence 

def transform_data(database_path,mode="ml"):
    PROJECT_PATH = os.path.realpath(os.path.dirname(__file__))
    df = fetch(database_path)
    df['clean_tweet'] = tqdm(df['tweet'].apply(clean_tweet,args=(mode,)))
    df.clean_tweet = df.clean_tweet.replace('',np.NaN)
    df.dropna(inplace=True)
    df = df[['clean_tweet','dialect']]
    df.to_csv(os.path.join(PROJECT_PATH, "./Data/clean_df.csv"),index=False)
    return df

