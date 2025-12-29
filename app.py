import streamlit as st
import spacy
from spacy import displacy
import subprocess
import sys

st.set_page_config(page_title="NER App", layout="wide")
st.title("🧠 Named Entity Recognition (NER)")

# -------------------------------
# Load spaCy model safely
# -------------------------------
@st.cache_resource
def load_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # download model if not present
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
        )
        return spacy.load("en_core_web_sm")

nlp = load_model()

# -------------------------------
# User Input
# -------------------------------
text = st.text_area(
    "Enter any sentence:",
    placeholder="Apple was founded by Steve Jobs in California in 1976."
)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text")
    else:
        doc = nlp(text)

        # -------------------------------
        # Show detected entities
        # -------------------------------
        st.subheader("📌 Detected Entities")
        if doc.ents:
            for ent in doc.ents:
                st.write(f"**{ent.text}** → {ent.label_}")
        else:
            st.info("No entities detected.")

        # -------------------------------
        # Visualize entities
        # -------------------------------
        st.subheader("🖍 Entity Visualization")
        html = displacy.render(doc, style="ent", jupyter=False)
        st.components.v1.html(html, height=300, scrolling=True)
