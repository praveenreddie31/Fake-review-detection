import streamlit as st
import joblib
import os
import re
import numpy as np

# ---------------------------
# 🔥 Hybrid Rule-Based Logic
# ---------------------------
def detect_fake_patterns(text):
    text = text.lower()

    words = text.split()
    if len(words) != len(set(words)):
        return True

    if text.count("!") > 3:
        return True

    keywords = ["best ever", "amazing", "must buy", "100% recommended"]
    if any(word in text for word in keywords):
        return True

    return False


# ---------------------------
# Load Model
# ---------------------------
base_dir = os.path.dirname(os.path.dirname(__file__))

model = joblib.load(os.path.join(base_dir, "model", "fake_review_model.pkl"))
vectorizer = joblib.load(os.path.join(base_dir, "model", "vectorizer.pkl"))

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Fake Review Detector", layout="centered")

st.title("🕵️ Fake Review Detection System")
st.write("Detect whether a review is **Fake or Genuine** using AI + Smart Rules")

# Input
user_input = st.text_area("Enter Review Text:")

# ---------------------------
# Prediction
# ---------------------------
if st.button("Analyze Review"):

    if user_input.strip() != "":

        input_vec = vectorizer.transform([user_input])

        # ML Prediction
        prediction = model.predict(input_vec)[0]

        # 🔥 Proper Confidence using sigmoid
        decision = model.decision_function(input_vec)[0]
        prob = 1 / (1 + np.exp(-decision))  # sigmoid

        if prediction == 1:
            confidence = round(prob * 100, 2)
        else:
            confidence = round((1 - prob) * 100, 2)

        # 🔥 Rule-Based Detection
        rule_flag = detect_fake_patterns(user_input)

        # Hybrid decision
        if prediction == 0 or rule_flag:
            result = "🚨 Fake Review"
            color = "red"
        else:
            result = "✅ Genuine Review"
            color = "green"

        # Display
        st.markdown(f"### Result: :{color}[{result}]")
        st.write(f"Confidence Score: {confidence}%")

        # Explanation
        if prediction == 0:
            st.warning("⚠️ ML model detected suspicious patterns.")

        if rule_flag:
            st.warning("⚠️ Rule-based system detected exaggerated or repetitive language.")

        if prediction == 1 and not rule_flag:
            st.success("✔️ This review appears natural and genuine.")

    else:
        st.warning("Please enter a review!")

# Footer
st.markdown("---")
st.caption("Built using SVM + NLP + Hybrid AI Rules")