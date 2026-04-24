import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer

# Load trained ML model
model = joblib.load("models/model.pkl")   

# Load sentence transformer (same one used in training)
embedder = SentenceTransformer("models/sentence_transformer_model")

st.title("📰 Fake News Detector")

news_text = st.text_area("Enter News Text:", height=200)

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        embedding = embedder.encode([news_text])

        prediction = model.predict(embedding)[0]

        if prediction == 1:
            st.success("This news is REAL")
        else:
            st.error("This news is FAKE")