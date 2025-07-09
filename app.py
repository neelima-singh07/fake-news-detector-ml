import streamlit as st
import pickle
import re


model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))


st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")


st.markdown("<h1 style='text-align: center; color: #3A6EA5;'>📰 Fake News Detector Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px; color:gray;'>Paste news below and view advanced analysis!</p>", unsafe_allow_html=True)

st.markdown("---")
news = st.text_area("📝 Paste your news article here", height=250, placeholder="e.g., The Prime Minister announced...")

# Abusive Word List
abusive_words = ["abuse", "kill", "hate", "terrorist", "fraud", "fake", "scam", "stupid", "idiot"]


if st.button("🔍 Analyze News"):

    if news.strip() == "":
        st.warning("Please enter some text.")
    else:
     
        data = vectorizer.transform([news])
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][prediction] * 100

       
        if prediction == 1:
            st.success("✅ This news is **REAL**")
        else:
            st.error("❌ This news is **FAKE**")

        st.markdown(f"🧠 **Confidence:** `{probability:.2f}%`")

       
        if abs(probability - (100 - probability)) < 10:
            st.info("⚠️ This is a borderline case. The model is not very confident.")

       
        st.markdown("## 📊 Text Insights")
        word_count = len(news.split())
        char_count = len(news)
        sentence_count = news.count(".") + news.count("!") + news.count("?")
        st.write(f"**🔤 Word Count:** {word_count}")
        st.write(f"**📝 Character Count:** {char_count}")
        st.write(f"**📚 Sentence Count:** {sentence_count}")

       
        lowered = news.lower()
        abusive_found = [word for word in abusive_words if word in lowered]
        st.warning(f"🚨 Abusive Words Detected: {', '.join(abusive_found) if abusive_found else 'None'}")

        
        st.markdown("## 🎯 Confidence Level")
        if probability > 85:
            st.success("🔵 High Confidence")
        elif probability > 60:
            st.info("🟡 Medium Confidence")
        else:
            st.warning("🔴 Low Confidence")

        st.markdown("---")
        st.markdown("<p style='text-align:center; font-size:13px; color:#999;'>Made  with ❤️ By Neelima!</p>", unsafe_allow_html=True)
