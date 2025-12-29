import streamlit as st
import spacy
from spacy.pipeline import EntityRuler

# -----------------------------
# Create blank English NLP
# -----------------------------
nlp = spacy.blank("en")

# -----------------------------
# Add EntityRuler (rule-based NER)
# -----------------------------
ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})

# -----------------------------
# DATE patterns (HIGH PRIORITY)
# -----------------------------
date_patterns = [
    # March 2024
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

    # 10 Jan 2025
    {
        "label": "DATE",
        "pattern": [
            {"LIKE_NUM": True},
            {"LOWER": {"IN": [
                "jan","january","feb","february","mar","march",
                "apr","april","may","jun","june","jul","july",
                "aug","august","sep","september",
                "oct","october","nov","november","dec","december"
            ]}},
            {"LIKE_NUM": True}
        ]
    },

    # 01/01/2025
    {
        "label": "DATE",
        "pattern": [{"TEXT": {"REGEX": "\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}"}}]
    },

    # 2025-01-10
    {
        "label": "DATE",
        "pattern": [{"TEXT": {"REGEX": "\\d{4}-\\d{2}-\\d{2}"}}]
    }
]

# -----------------------------
# STANDARD ENTITY RULES (SAFE)
# -----------------------------
standard_patterns = [

    # PERSON (two title-case words only)
    {
        "label": "PERSON",
        "pattern": [{"IS_TITLE": True}, {"IS_TITLE": True}]
    },

    # ORG (keywords)
    {
        "label": "ORG",
        "pattern": [
            {"IS_TITLE": True, "OP": "+"},
            {"LOWER": {"IN": ["inc", "inc.", "ltd", "corp", "corporation"]}}
        ]
    },

    # Known organizations
    {
        "label": "ORG",
        "pattern": [{"TEXT": "United"}, {"TEXT": "Nations"}]
    },

    # FAC
    {
        "label": "FAC",
        "pattern": [{"TEXT": "Giga"}, {"TEXT": "Texas"}]
    },

    # GPE
    {
        "label": "GPE",
        "pattern": [{"LOWER": {"IN": ["austin", "texas", "berlin"]}}]
    },

    # PRODUCT
    {
        "label": "PRODUCT",
        "pattern": [{"TEXT": "Model"}, {"LIKE_NUM": True}]
    },

    # LAW
    {
        "label": "LAW",
        "pattern": [{"IS_TITLE": True, "OP": "+"}, {"LOWER": "act"}]
    },

    # EVENT
    {
        "label": "EVENT",
        "pattern": [{"IS_TITLE": True, "OP": "+"}, {"LOWER": "summit"}]
    },

    # WORK_OF_ART
    {
        "label": "WORK_OF_ART",
        "pattern": [{"TEXT": "Mona"}, {"TEXT": "Lisa"}]
    },

    # LANGUAGE
    {
        "label": "LANGUAGE",
        "pattern": [{"LOWER": {"IN": ["english", "german", "french"]}}]
    },

    # NORP
    {
        "label": "NORP",
        "pattern": [{"LOWER": {"IN": ["german", "indian", "american"]}}]
    },

    # MONEY
    {
        "label": "MONEY",
        "pattern": [{"TEXT": {"REGEX": "[$₹]"}}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["million", "billion"]}}]
    },

    # PERCENT
    {
        "label": "PERCENT",
        "pattern": [{"LIKE_NUM": True}, {"TEXT": "%"}]
    },

    # TIME
    {
        "label": "TIME",
        "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["hour", "hours"]}}]
    },

    # QUANTITY
    {
        "label": "QUANTITY",
        "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["kilograms", "kg", "centimeters", "cm"]}}]
    }
]

# -----------------------------
# CUSTOM ENTITY RULES
# -----------------------------
custom_patterns = [
    {"label": "BANK", "pattern": [{"LOWER": {"IN": ["sbi", "hdfc", "icici", "axis"]}}]},
    {"label": "ACCOUNT_NO", "pattern": [{"TEXT": {"REGEX": "\\d{9,18}"}}]},
    {"label": "AADHAR_NO", "pattern": [{"TEXT": {"REGEX": "\\d{4}\\s\\d{4}\\s\\d{4}"}}]},
    {"label": "PAN_NO", "pattern": [{"TEXT": {"REGEX": "[A-Z]{5}[0-9]{4}[A-Z]"}}]},
    {"label": "TRANSACTION_ID", "pattern": [{"TEXT": {"REGEX": "TXN[0-9A-Z]+"}}]},
    {"label": "DISEASE", "pattern": [{"LOWER": {"IN": ["diabetes", "cancer", "asthma"]}}]},
    {"label": "MEDICINE", "pattern": [{"LOWER": {"IN": ["paracetamol", "insulin", "ibuprofen"]}}]},
    {"label": "STOCK", "pattern": [{"IS_UPPER": True, "LENGTH": {">=": 3}}]}
]

# -----------------------------
# Add rules in correct order
# -----------------------------
ruler.add_patterns(date_patterns)
ruler.add_patterns(standard_patterns)
ruler.add_patterns(custom_patterns)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="NER App", layout="centered")
st.title("🧠 Named Entity Recognition (Correct Rules)")

text = st.text_area(
    "Enter text",
    height=200,
    placeholder="In March 2024, Elon Musk announced at the Giga Texas factory..."
)

if st.button("Extract Entities"):
    doc = nlp(text)
    if not doc.ents:
        st.warning("No entities found")
    else:
        for ent in doc.ents:
            st.write(f"🔹 **{ent.text}** → `{ent.label_}`")
