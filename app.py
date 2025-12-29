import streamlit as st
import spacy
from spacy import displacy

# Page config
st.set_page_config(page_title="NER App", layout="wide")
st.title("🧠 Named Entity Recognition (NER)")
st.write("Enter any sentence. The model will automatically detect entities.")

# Load pretrained spaCy NER model (NO predefined rules)
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

# User input
text = st.text_area(
    "Enter text:",
    placeholder="Apple was founded by Steve Jobs in California in 1976."
)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        doc = nlp(text)

        # Show entities
        st.subheader("Detected Entities")
        if doc.ents:
            for ent in doc.ents:
                st.write(f"**{ent.text}** → {ent.label_}")
        else:
            st.info("No entities found.")

        # Visual NER
        st.subheader("Entity Visualization")
        html = displacy.render(doc, style="ent", jupyter=False)
        st.components.v1.html(html, scrolling=True, height=300)
