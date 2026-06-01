import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('best_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector")
st.markdown("Paste any news article or headline below to check if it's **Real** or **Fake**.")
st.divider()

user_input = st.text_area("Enter news text here:", height=200, placeholder="Paste your news article or headline...")

if st.button("🔍 Detect", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        vector = tfidf.transform([user_input])
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        st.divider()

        if prediction == 1:
            st.error("🚨 This news appears to be **FAKE**")
            st.metric("Confidence", f"{round(probability[1]*100, 1)}%")
        else:
            st.success("✅ This news appears to be **REAL**")
            st.metric("Confidence", f"{round(probability[0]*100, 1)}%")

st.divider()
st.caption("Built with Scikit-learn + Streamlit | BSCS ML Project")
