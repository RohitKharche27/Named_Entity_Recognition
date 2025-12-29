import streamlit as st
import spacy
from spacy import displacy

# Load spaCy pipeline safely
@st.cache_resource
def load_nlp():
    try:
        # Use small English pipeline without downloading models
        nlp = spacy.blank("en")
        nlp.add_pipe("ner")
        return nlp
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

nlp = load_nlp()

st.set_page_config(page_title="NER App", layout="wide")
st.title("🧠 Named Entity Recognition (NER)")
st.write("Enter any sentence. The model will automatically detect entities.")

text = st.text_area(
    "Enter text:",
    placeholder="Apple was founded by Steve Jobs in California in 1976."
)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text")
    elif nlp is None:
        st.error("NLP model not loaded")
    else:
        doc = nlp(text)

        st.subheader("Detected Entities")
        if doc.ents:
            for ent in doc.ents:
                st.write(f"**{ent.text}** → {ent.label_}")
        else:
            st.info("No entities detected.")

        st.subheader("Entity Visualization")
        html = displacy.render(doc, style="ent", jupyter=False)
        st.components.v1.html(html, scrolling=True, height=300)
