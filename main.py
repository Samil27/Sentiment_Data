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

df= pd.read_csv('/content/Sentiment_Data.csv', encoding="ISO-8859-1")
df.head()

import seaborn as sns

sentiment_map = {
    'Mild_Pos': 'Positive',
    'Strong_Pos': 'Positive',
    'Neutral': 'Neutral',
    'Strong_Neg': 'Negative',
    'Mild_Neg': 'Negative'
}
df['Sentiment'] = df['Sentiment'].map(sentiment_map)
df

plt.figure(figsize=(10, 6))
sns.countplot(x='Sentiment', data=df, palette='viridis')
plt.title('Distribution of Sentiments')
plt.show()

df['cleaned_Tweet'] = df['Tweet'].apply(clean_text)
df.head()

df['word_count'] = df['cleaned_Tweet'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(10, 6))
sns.histplot(df['word_count'], kde=True, bins=30, color='purple')
plt.title('Distribution of Word Counts in Tweets')
plt.show()

df = df.groupby('Sentiment').apply(lambda x: x.sample(n=76612, replace=True)).reset_index(drop=True)

texts = df['cleaned_Tweet'].astype(str).values
labels = df['Sentiment'].values

label_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
labels = np.array([label_mapping[label] for label in labels])

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
tokenizer = Tokenizer(num_words=5000)

def preprocess_text(tweet):
    tokens = word_tokenize(tweet)
    tokens = [word for word in tokens if word.lower() not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tweet = ' '.join(tokens)
    return tweet

X_train = [preprocess_text(tweet) for tweet in X_train]
X_test = [preprocess_text(tweet) for tweet in X_test]

tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=30)
X_test_pad = pad_sequences(X_test_seq, maxlen=30)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_pad)

plt.figure(figsize=(10, 7))
for i in np.unique(y_train):
    idxs = np.where(y_train == i)
    plt.scatter(X_train_pca[idxs, 0], X_train_pca[idxs, 1], label=f'Class {i}', alpha=0.5)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of Train Set Embeddings')
plt.legend()
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

model = Sequential([
    Embedding(input_dim=20000, output_dim=128, input_length=30),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train_pad, y_train, epochs=10, validation_split=0.1, batch_size=32)

def evaluate_model_performance(model, X_test_pad, y_test, history):
    y_test_pred = np.argmax(model.predict(X_test_pad), axis=-1)
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Negative', 'Neutral', 'Positive']))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

    y_pred = model.predict(X_test_pad)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)

    y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
    y_pred_binarized = label_binarize(y_pred_classes, classes=[0, 1, 2])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Neutral', 'Positive'])
    disp.plot(cmap=plt.cm.Blues, ax=axes[0])
    axes[0].set_title('Confusion Matrix')

    ax = axes[1]
    colors = ['blue', 'green', 'red']
    for i, color in zip(range(3), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2, label='Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title('ROC AUC for All Classes')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

# evaluate_model_performance(model, X_test_pad, y_test, history)

import lime
import lime.lime_text
from sklearn.pipeline import make_pipeline

def predict_fn(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=30)
    return model.predict(padded_sequences)

explainer = lime.lime_text.LimeTextExplainer(class_names=['Negative', 'Neutral', 'Positive'])

i = 5000
explanation = explainer.explain_instance(X_test[i], predict_fn, num_features=10)

explanation.show_in_notebook(text=True)

i = 8000
explanation = explainer.explain_instance(X_test[i], predict_fn, num_features=10)

explanation.show_in_notebook(text=True)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

dense_model = Sequential([
    Embedding(input_dim=20000, output_dim=128, input_length=30),
    Bidirectional(LSTM(256, return_sequences=True)),
    Bidirectional(LSTM(128, return_sequences=True)),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

dense_model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
dense_model.summary()

history = dense_model.fit(X_train_pad, y_train, epochs=10, validation_split=0.1, batch_size=32)

evaluate_model_performance(dense_model, X_test_pad, y_test, history)
