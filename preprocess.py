
import seaborn as sns


sentiment_map = {
    'Mild_Pos': 'Positive',
    'Strong_Pos': 'Positive',
    'Neutral': 'Neutral',
    'Strong_Neg': 'Negative',
    'Mild_Neg': 'Negative'
}
df['Sentiment'] = df['Sentiment'].map(sentiment_map)


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
