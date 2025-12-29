import streamlit as st
import spacy
from spacy import displacy

# ---------------------------
# Load pretrained spaCy NER
# ---------------------------
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")  # pretrained NER model

nlp = load_model()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="NER App", layout="wide")
st.title("🧠 Named Entity Recognition (NER)")
st.write("Enter any sentence and the model will automatically detect entities.")

text = st.text_area(
    "Enter text:",
    placeholder="Apple was founded by Steve Jobs in California in 1976."
)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text")
    else:
        doc = nlp(text)

        # ---------------------------
        # Show detected entities
        # ---------------------------
        st.subheader("Detected Entities")
        if doc.ents:
            for ent in doc.ents:
                st.write(f"**{ent.text}** → `{ent.label_}`")
        else:
            st.info("No entities detected.")

        # ---------------------------
        # Visual NER output
        # ---------------------------
        st.subheader("Entity Visualization")
        html = displacy.render(doc, style="ent", jupyter=False)
        st.components.v1.html(html, height=350, scrolling=True)
