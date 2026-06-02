import streamlit as st
import pickle

model = pickle.load(open('best_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

st.markdown("""
<div style="
background: linear-gradient(90deg,#4F46E5,#7C3AED);
padding:25px;
border-radius:15px;
text-align:center;
color:white;
margin-bottom:20px;
">
<h1>📰 Fake News Detector</h1>
<p>AI Powered News Verification System</p>
</div>
""", unsafe_allow_html=True)

user_input = st.text_area(
    "💡 For best accuracy, paste a complete news article rather than a short headline."
)

with st.sidebar:
    st.header("📊 Project Details")
    st.write("Model: XGBoost")
    st.write("Vectorizer: TF-IDF")
    st.write("Dataset: WELFake")
    st.write("Type: NLP Classification")
    st.divider()
    st.write("Developer:")
    st.write("Ali Shehryar And Mohammad Umar")

if st.button("Verify"):
    vector = tfidf.transform([user_input])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]

    st.divider()

    if prediction == 1:
        confidence = probability[1] * 100
        st.error("🚨 FAKE NEWS DETECTED")
        st.progress(int(confidence))
        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )
    else:
        confidence = probability[0] * 100
        st.success("✅ REAL NEWS DETECTED")
        st.progress(int(confidence))
        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

st.divider()

st.caption(
    "Built using Streamlit, TF-IDF and XGBoost | BSCS Machine Learning Project"
)
