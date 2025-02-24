import pandas as pd
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import streamlit as st 
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np
import pickle

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

@st.cache_data
def load_data():
    df = pd.read_csv('hate_speech_data.csv')
    text = df['tweet']
    label = df['class']
    return text, label

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)

def split_data(text, label):
    x_train, x_test, y_train, y_test = train_test_split(
        text,
        label,
        test_size=0.2,
        random_state=42, 
        stratify=label
    )
    return (x_train, x_test, y_train, y_test)

def set_data(x_train, x_test):
    vocab_size = 10000
    max_length = 100
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train)
    x_train_tok = tokenizer.texts_to_sequences(x_train)
    x_test_tok = tokenizer.texts_to_sequences(x_test)
    x_train_pad = pad_sequences(x_train_tok, maxlen=max_length, padding='post', truncating='post')
    x_test_pad = pad_sequences(x_test_tok, maxlen=max_length, padding='post', truncating='post')
    return (x_train_pad, x_test_pad, tokenizer)

def show_classification_report(y_test, y_pred):
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    report = classification_report(y_test, y_pred, output_dict=True)
    report = pd.DataFrame(report)
    sns.heatmap(report.iloc[: -1, : -2], annot=True, fmt=".2f", cmap="Blues")
    ax1.set_title("Classification report")
    st.pyplot(fig1)
    plt.close(fig1)

def show_conf_matrix(y_test, y_pred):
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
    disp.plot(cmap='Blues', ax=ax2)
    ax2.set_title("Confusion Matrix")
    st.pyplot(fig2)
    plt.close(fig2)

def show_roc(y_test, y_pred_proba):
    class_names = ["hate speech", "offensive language", "neither"]
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    y_roc = label_binarize(y_test, classes=[0, 1, 2])
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_roc[:, i], y_pred_proba[:, i])
        auc = roc_auc_score(y_roc[:, i], y_pred_proba[:, i])
        ax3.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc:.2f})")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curve")
    ax3.legend()
    st.pyplot(fig3)
    plt.close(fig3)

@st.cache_resource
def load_model():
    model = keras.models.load_model('text_rnn.keras')
    return model

def comment_system(model, tokenizer):
    max_length = 100
    st.subheader("Main Post")
    post_text = "This is the main post that users can comment on. Write your comment below"
    st.write(post_text)
    if "comments" not in st.session_state:
        st.session_state.comments = []
    st.markdown("### Accepted comments:")
    if st.session_state.comments:
        for com in st.session_state.comments:
            st.write("- " + com)
    else:
        st.info("There are no comments yet.")
    st.markdown("---")
    st.subheader("Add your comment")
    comment = st.text_area("Write your comment here: ")
    if st.button("Post comment"):
        if not comment:
            st.error("Please enter a comment.")
        else:
            cleaned_comment = clean_text(comment)
            seq = tokenizer.texts_to_sequences([cleaned_comment])
            pad_seq = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
            pred = model.predict(pad_seq)[0]
            predicted_class = int(np.argmax(pred))
            classes = {0: "hate speech", 1: "offensive language", 2: "neither"}
            if classes[predicted_class] != "neither":
                st.write(f"**Comment Classification:** {classes[predicted_class]}")
            if predicted_class == 2:
                st.success("Your comment has been accepted")
                st.session_state.comments.append(comment)
                st.session_state
            else:
                st.error("You cannot post this comment because it contains unacceptable content.")