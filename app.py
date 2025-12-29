import streamlit as st
import spacy
from spacy.pipeline import EntityRuler

# -----------------------------
# Create NLP model HERE (no download)
# -----------------------------
nlp = spacy.blank("en")

# Add sentencizer (good practice)
nlp.add_pipe("sentencizer")

# Add EntityRuler
ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})

# -----------------------------
# DATE patterns
# -----------------------------
date_patterns = [
    {
        "label": "DATE",
        "pattern": [
            {"LOWER": {"IN": [
                "january","february","march","april","may","june",
                "july","august","september","october","november","december"
            ]}},
            {"LIKE_NUM": True}
        ]
    },
    {
        "label": "DATE",
        "pattern": [{"TEXT": {"REGEX": "\\d{4}"}}]
    }
]

# -----------------------------
# CORE ENTITY PATTERNS (STRICT)
# -----------------------------
core_patterns = [

    # PERSON (two capital words only)
    {"label": "PERSON", "pattern": [{"IS_TITLE": True}, {"IS_TITLE": True}]},

    # ORG
    {"label": "ORG", "pattern": [
        {"IS_TITLE": True, "OP": "+"},
        {"LOWER": {"IN": ["inc", "inc.", "ltd", "corp"]}}
    ]},

    # GPE
    {"label": "GPE", "pattern": [{"LOWER": {"IN": ["austin", "texas", "berlin", "pune", "mumbai"]}}]},

    # FAC
    {"label": "FAC", "pattern": [{"TEXT": "Giga"}, {"TEXT": "Texas"}]},

    # PRODUCT
    {"label": "PRODUCT", "pattern": [{"TEXT": "Model"}, {"LIKE_NUM": True}]},

    # MONEY
    {"label": "MONEY", "pattern": [
        {"TEXT": {"REGEX": "[$₹]"}},
        {"LIKE_NUM": True},
        {"LOWER": {"IN": ["million", "billion"]}}
    ]},

    # PERCENT
    {"label": "PERCENT", "pattern": [{"LIKE_NUM": True}, {"TEXT": "%"}]},

    # QUANTITY
    {"label": "QUANTITY", "pattern": [
        {"LIKE_NUM": True},
        {"LOWER": {"IN": ["kg", "kilograms", "cm", "centimeters"]}}
    ]},

    # WORK OF ART
    {"label": "WORK_OF_ART", "pattern": [{"TEXT": "Mona"}, {"TEXT": "Lisa"}]},

    # LAW
    {"label": "LAW", "pattern": [{"IS_TITLE": True, "OP": "+"}, {"LOWER": "act"}]},

    # EVENT
    {"label": "EVENT", "pattern": [{"IS_TITLE": True, "OP": "+"}, {"LOWER": "summit"}]},
]

# -----------------------------
# CUSTOM DOMAIN ENTITIES
# -----------------------------
custom_patterns = [
    {"label": "BANK", "pattern": [{"LOWER": {"IN": ["sbi", "hdfc", "icici", "axis"]}}]},
    {"label": "ACCOUNT_NO", "pattern": [{"TEXT": {"REGEX": "\\b\\d{9,18}\\b"}}]},
    {"label": "AADHAR_NO", "pattern": [{"TEXT": {"REGEX": "\\b\\d{4}\\s\\d{4}\\s\\d{4}\\b"}}]},
    {"label": "PAN_NO", "pattern": [{"TEXT": {"REGEX": "\\b[A-Z]{5}[0-9]{4}[A-Z]\\b"}}]},
    {"label": "TRANSACTION_ID", "pattern": [{"TEXT": {"REGEX": "\\bTXN[0-9A-Z]+\\b"}}]},
    {"label": "DISEASE", "pattern": [{"LOWER": {"IN": ["diabetes", "cancer", "asthma"]}}]},
    {"label": "MEDICINE", "pattern": [{"LOWER": {"IN": ["paracetamol", "insulin", "ibuprofen"]}}]},
    {"label": "STOCK", "pattern": [{"IS_UPPER": True, "LENGTH": {">=": 3}}]},
]

# -----------------------------
# Add all patterns (ORDER MATTERS)
# -----------------------------
ruler.add_patterns(date_patterns)
ruler.add_patterns(core_patterns)
ruler.add_patterns(custom_patterns)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="NER App (No Download)", layout="centered")

st.title("🧠 Named Entity Recognition")
st.write("✔️ Model created in code | ✔️ No spaCy model download | ✔️ Cloud-safe")

text = st.text_area(
    "Enter text",
    height=220,
    placeholder="In March 2024, Elon Musk announced at Giga Texas in Austin that Tesla Inc. invested $5 billion."
)

if st.button("Extract Entities"):
    if not text.strip():
        st.warning("Please enter text")
    else:
        doc = nlp(text)
        if not doc.ents:
            st.info("No entities found")
        else:
            for ent in doc.ents:
                st.write(f"🔹 **{ent.text}** → `{ent.label_}`")
