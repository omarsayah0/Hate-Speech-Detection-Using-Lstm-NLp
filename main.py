#This is how to run the program : python -m streamlit run main.py

import streamlit as st
import numpy as np
from utils import (
    load_data, 
    load_model, 
    set_data,
    clean_text,
    split_data,
    show_classification_report, 
    show_conf_matrix, 
    show_roc,
    comment_system,
    load_tokenizer
)

st.set_page_config(page_title="Text Classifier", layout="centered")

def main():
    st.title("Model Evaluation & Text Prediction")
    model = load_model() 
    tab1, tab2 = st.tabs(["Model Evaluation 📊", "Comment Section 💬"])
    with tab1:
        st.subheader("Model Performance")
        text, label = load_data()
        text_clean = text.apply(clean_text)
        x_train, x_test, y_train, y_test = split_data(text_clean, label)
        _, x_test_pad, _ = set_data(x_train, x_test)
        loss, acc = model.evaluate(x_test_pad, y_test, verbose=0)
        st.write(f"❌ loss : {loss:.2f}")
        st.write(f"✅ accuracy : {acc:.2f}")
        y_pred_proba = model.predict(x_test_pad)
        y_pred = np.argmax(y_pred_proba, axis=1)
        st.subheader("Classification Report")
        show_classification_report(y_test, y_pred)
        st.subheader("Confusion Matrix")
        show_conf_matrix(y_test, y_pred)
        st.subheader(" ROC Curve")
        show_roc(y_test, y_pred_proba)
    with tab2:
        st.subheader("A comment section")
        tokenizer = load_tokenizer()
        comment_system(model, tokenizer)

if __name__ == '__main__':
    main()