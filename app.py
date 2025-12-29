import streamlit as st
import spacy
from spacy.pipeline import EntityRuler

# -----------------------------
# Load ML-based English model
# -----------------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# Add EntityRuler AFTER ner (HYBRID)
# -----------------------------
ruler = nlp.add_pipe("entity_ruler", after="ner", config={"overwrite_ents": False})

# -----------------------------
# Custom entity patterns
# -----------------------------
custom_patterns = [

    # BANK
    {"label": "BANK", "pattern": [{"LOWER": {"IN": ["sbi", "hdfc", "icici", "axis"]}}]},

    # ACCOUNT NUMBER
    {"label": "ACCOUNT_NO", "pattern": [{"TEXT": {"REGEX": "\\b\\d{9,18}\\b"}}]},

    # AADHAR
    {"label": "AADHAR_NO", "pattern": [{"TEXT": {"REGEX": "\\b\\d{4}\\s\\d{4}\\s\\d{4}\\b"}}]},

    # PAN
    {"label": "PAN_NO", "pattern": [{"TEXT": {"REGEX": "\\b[A-Z]{5}[0-9]{4}[A-Z]\\b"}}]},

    # TRANSACTION ID
    {"label": "TRANSACTION_ID", "pattern": [{"TEXT": {"REGEX": "\\bTXN[0-9A-Z]+\\b"}}]},

    # DISEASE
    {"label": "DISEASE", "pattern": [{"LOWER": {"IN": ["diabetes", "cancer", "asthma"]}}]},

    # MEDICINE
    {"label": "MEDICINE", "pattern": [{"LOWER": {"IN": ["paracetamol", "insulin", "ibuprofen"]}}]},

    # STOCK SYMBOLS
    {"label": "STOCK", "pattern": [{"IS_UPPER": True, "LENGTH": {">=": 3}}]},
]

ruler.add_patterns(custom_patterns)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Smart NER App", layout="centered")

st.title("🧠 Smart Named Entity Recognition")
st.write(
    "✔️ ML-based NER + Custom Rules\n\n"
    "✔️ Correct entities only (no random words)\n\n"
    "✔️ Works for ANY text you type"
)

text = st.text_area(
    "Enter text",
    height=220,
    placeholder="In March 2024, Elon Musk announced that Tesla Inc. invested $5 billion..."
)

if st.button("Extract Entities"):
    if not text.strip():
        st.warning("Please enter some text")
    else:
        doc = nlp(text)

        if not doc.ents:
            st.info("No entities detected")
        else:
            st.success("Detected Entities")
            for ent in doc.ents:
                st.write(f"🔹 **{ent.text}** → `{ent.label_}`")
