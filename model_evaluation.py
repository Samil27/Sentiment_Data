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

evaluate_model_performance(model, X_test_pad, y_test, history)

get_ipython().system('pip install lime')

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
