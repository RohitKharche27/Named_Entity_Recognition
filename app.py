import streamlit as st
import spacy
from spacy.pipeline import EntityRuler

# -----------------------------
# Load blank English model
# -----------------------------
nlp = spacy.blank("en")

# -----------------------------
# Add EntityRuler
# -----------------------------
ruler = nlp.add_pipe("entity_ruler")

# -----------------------------
# spaCy default 18 entities (generic patterns)
# -----------------------------
default_patterns = [
    {"label": "PERSON", "pattern": [{"IS_TITLE": True}, {"IS_TITLE": True, "OP": "?"}]},
    {"label": "ORG", "pattern": [{"IS_TITLE": True, "OP": "+"}]},
    {"label": "GPE", "pattern": [{"IS_TITLE": True}]},
    {"label": "DATE", "pattern": [{"LIKE_NUM": True}]},
    {"label": "TIME", "pattern": [{"LIKE_NUM": True}, {"LOWER": "pm"}]},
    {"label": "MONEY", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["rs", "₹", "$"]}}]},
    {"label": "PERCENT", "pattern": [{"LIKE_NUM": True}, {"TEXT": "%"}]},
    {"label": "CARDINAL", "pattern": [{"LIKE_NUM": True}]},
    {"label": "ORDINAL", "pattern": [{"LOWER": {"IN": ["first", "second", "third"]}}]},
    {"label": "QUANTITY", "pattern": [{"LIKE_NUM": True}, {"IS_ALPHA": True}]},
    {"label": "LANGUAGE", "pattern": [{"LOWER": {"IN": ["english", "hindi", "marathi"]}}]},
    {"label": "EVENT", "pattern": [{"IS_TITLE": True}, {"LOWER": "day"}]},
    {"label": "LAW", "pattern": [{"IS_TITLE": True}, {"LOWER": "act"}]},
    {"label": "WORK_OF_ART", "pattern": [{"IS_TITLE": True}]},
    {"label": "PRODUCT", "pattern": [{"IS_TITLE": True}]},
    {"label": "LOC", "pattern": [{"IS_TITLE": True}]},
    {"label": "NORP", "pattern": [{"IS_TITLE": True}]},
    {"label": "FAC", "pattern": [{"IS_TITLE": True}]},
]

# -----------------------------
# Custom entity patterns
# -----------------------------
custom_patterns = [
    {"label": "BANK", "pattern": [{"LOWER": {"IN": ["sbi", "hdfc", "icici", "axis"]}}]},
    {"label": "ACCOUNT_NO", "pattern": [{"TEXT": {"REGEX": "\\d{9,18}"}}]},
    {"label": "AADHAR_NO", "pattern": [{"TEXT": {"REGEX": "\\d{4}\\s\\d{4}\\s\\d{4}"}}]},
    {"label": "PAN_NO", "pattern": [{"TEXT": {"REGEX": "[A-Z]{5}[0-9]{4}[A-Z]"}}]},
    {"label": "TRANSACTION_ID", "pattern": [{"TEXT": {"REGEX": "TXN[0-9A-Z]+"}}]},
    {"label": "DISEASE", "pattern": [{"LOWER": {"IN": ["diabetes", "cancer", "asthma"]}}]},
    {"label": "MEDICINE", "pattern": [{"LOWER": {"IN": ["paracetamol", "insulin", "ibuprofen"]}}]},
    {"label": "STOCK", "pattern": [{"IS_UPPER": True, "LENGTH": {">=": 3}}]},
]

# Add all patterns
ruler.add_patterns(default_patterns + custom_patterns)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Custom NER App", layout="centered")

st.title("🧠 Named Entity Recognition (NER)")
st.write("Detects **18 spaCy entities + custom banking, medical & finance entities**")

text = st.text_area(
    "Enter text",
    placeholder="Rohit transferred ₹5000 from SBI account 123456789012 using TXN8899 on 10 Jan 2025"
)

if st.button("Extract Entities"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        doc = nlp(text)
        if not doc.ents:
            st.info("No entities found")
        else:
            st.success("Entities Found")
            for ent in doc.ents:
                st.write(f"🔹 **{ent.text}** → `{ent.label_}`")
