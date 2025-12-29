import streamlit as st
import spacy
from spacy import displacy

# Load spaCy safely (NO model download error)
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except:
        nlp = spacy.blank("en")
        ner = nlp.add_pipe("ner")
        return nlp

nlp = load_nlp()

st.set_page_config(page_title="NER App", layout="wide")
st.title("🧠 Named Entity Recognition (NER)")

text = st.text_area(
    "Enter any sentence:",
    placeholder="Apple was founded by Steve Jobs in California in 1976."
)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter text")
    else:
        doc = nlp(text)

        if not doc.ents:
            st.info("No entities detected")
        else:
            st.subheader("Detected Entities")
            for ent in doc.ents:
                st.write(f"**{ent.text}** → {ent.label_}")

            st.subheader("Visualization")
            html = displacy.render(doc, style="ent", jupyter=False)
            st.components.v1.html(html, height=300, scrolling=True)
