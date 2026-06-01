import streamlit as st
import pickle

# Load Model & Vectorizer
model = pickle.load(open('best_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Page Configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #1f77b4;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
    margin-bottom: 20px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">📰 Fake News Detector</div>',
            unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">AI Powered News Verification System</div>',
    unsafe_allow_html=True
)

st.divider()

# Input Section
user_input = st.text_area(
    "✍️ Enter News Article / Headline",
    height=220,
    placeholder="Paste any news article or headline here..."
)

# Detect Button
if st.button("🔍 Analyze News", use_container_width=True):

    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:

        with st.spinner("Analyzing news..."):

            vector = tfidf.transform([user_input])

            prediction = model.predict(vector)[0]
            probability = model.predict_proba(vector)[0]

        st.divider()

        if prediction == 1:

            confidence = round(probability[1] * 100, 2)

            st.error("🚨 FAKE NEWS DETECTED")

            st.progress(int(confidence))

            st.metric(
                label="Confidence Score",
                value=f"{confidence}%"
            )

        else:

            confidence = round(probability[0] * 100, 2)

            st.success("✅ REAL NEWS DETECTED")

            st.progress(int(confidence))

            st.metric(
                label="Confidence Score",
                value=f"{confidence}%"
            )

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        """
        This project uses:

        ✔ TF-IDF Vectorizer
        
        ✔ Machine Learning Model
        
        ✔ Streamlit Interface
        
        ✔ Fake News Classification
        """
    )

    st.info(
        "Designed for BSCS Machine Learning Project"
    )

# Footer
st.divider()

st.markdown(
    """
    <div class="footer">
        Built with ❤️ using Scikit-Learn & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
