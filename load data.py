
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import re
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import time


get_ipython().system('pip install emoji')


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def handle_negations(text):
    negation_patterns = ["n't", "not", "never", "no"]
    words = text.split()
    for i in range(len(words)):
        if words[i].lower() in negation_patterns:
            if i+1 < len(words):
                words[i+1] = "NOT_" + words[i+1]
    return ' '.join(words)


def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'@\w+', ' ', text)
        text = re.sub(r'#', ' ', text)
        text = re.sub(r'\w+:\/\/\S+', ' ', text)
        text = re.sub(r'\S*@\S*\s?', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'[^a-zA-Z]+', ' ', text)
        text = handle_negations(text)
        text = emoji.demojize(text)
        text = re.sub(r'\d+', ' ', text)
        text = text.lower()
    return text


df = pd.read_csv('/Sentiment_Data.csv', encoding='ISO-8859-1')
df.head()
