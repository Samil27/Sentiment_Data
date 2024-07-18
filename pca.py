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