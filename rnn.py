import pandas as pd
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pickle

def load_data():
    df = pd.read_csv('hate_speech_data.csv')
    text= df['tweet']
    label = df['class']
    return (text, label)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [stemmer.stem(w) for w in words]
    text = ' '.join(words)
    return text

def split_data(text, label):
    x_train, x_test, y_train, y_test = train_test_split(
        text,
        label,
        test_size = 0.2,
        random_state = 42, 
        stratify = label
    )
    return(x_train, x_test, y_train, y_test)

def set_data(x_train, x_test):
    vocab_size = 10000
    max_length = 100
    tokenizer = Tokenizer(num_words = vocab_size, oov_token = "<OOV>")
    tokenizer.fit_on_texts(x_train)
    x_train_tok = tokenizer.texts_to_sequences(x_train)
    x_test_tok = tokenizer.texts_to_sequences(x_test)
    x_train_pad = pad_sequences(x_train_tok, maxlen = max_length, padding = 'post', truncating = 'post')
    x_test_pad = pad_sequences(x_test_tok, maxlen = max_length, padding = 'post', truncating = 'post')
    return (x_train_pad, x_test_pad, tokenizer)

def set_model():
    vocab_size = 10000
    max_length = 100
    model = keras.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=100, input_length=max_length),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.5),
        layers.Bidirectional(layers.LSTM(128)),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy']
    )
    model.summary()
    return (model)

def main():
    text, label = load_data()
    text = text.apply(clean_text)
    x_train, x_test, y_train, y_test = split_data(text, label)
    x_train, x_test, tokenizer = set_data(x_train, x_test)
    with open('tokenizer.pkl', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    model = set_model()
    model.fit(
        x_train,
        y_train,
        validation_split = 0.2,
        epochs = 10,
        batch_size = 64
    )
    model.save("text_rnn.keras")

if __name__ == '__main__':
    main()