Content is user-generated and unverified.
1
import streamlit as st
import spacy
from spacy import displacy
import pandas as pd

# Page config
st.set_page_config(page_title="Named Entity Recognition", page_icon="🔍", layout="wide")

# Title and description
st.title("🔍 Named Entity Recognition (NER)")
st.markdown("Extract and visualize named entities from text using spaCy")

# Load spaCy model
@st.cache_resource
def load_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.error("Model not found. Please install it using: python -m spacy download en_core_web_sm")
        return None

nlp = load_model()

if nlp:
    # Sidebar
    st.sidebar.header("Settings")
    show_visualization = st.sidebar.checkbox("Show Visual Highlighting", value=True)
    show_table = st.sidebar.checkbox("Show Entity Table", value=True)
    
    # Entity type filter
    st.sidebar.subheader("Filter Entity Types")
    entity_types = ["PERSON", "ORG", "GPE", "DATE", "MONEY", "TIME", "PERCENT", "LOC", "PRODUCT", "EVENT"]
    selected_entities = st.sidebar.multiselect(
        "Select entity types to display:",
        entity_types,
        default=entity_types
    )
    
    # Sample texts
    sample_texts = {
        "Custom Text": "",
        "Business News": "Apple Inc. announced a $100 billion investment in AI technology on January 15, 2024. CEO Tim Cook spoke at the event in San Francisco, stating that the company will hire 5,000 engineers by next year.",
        "Sports": "Lionel Messi scored two goals for Inter Miami in their 3-1 victory against New York Red Bulls on Saturday. The match took place at DRV PNK Stadium in Fort Lauderdale.",
        "Politics": "President Joe Biden met with Prime Minister Rishi Sunak at the White House on Monday to discuss trade relations between the United States and the United Kingdom."
    }
    
    # Text input
    st.subheader("Input Text")
    selected_sample = st.selectbox("Choose a sample or enter custom text:", list(sample_texts.keys()))
    
    if selected_sample == "Custom Text":
        text = st.text_area("Enter your text here:", height=150, placeholder="Type or paste your text here...")
    else:
        text = st.text_area("Enter your text here:", value=sample_texts[selected_sample], height=150)
    
    # Process button
    if st.button("🚀 Extract Entities", type="primary"):
        if text.strip():
            with st.spinner("Processing..."):
                # Process text
                doc = nlp(text)
                
                # Filter entities
                entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) 
                           for ent in doc.ents if ent.label_ in selected_entities]
                
                if entities:
                    st.success(f"Found {len(entities)} entities!")
                    
                    # Visual highlighting
                    if show_visualization:
                        st.subheader("📊 Visual Representation")
                        colors = {
                            "PERSON": "#aa9cfc",
                            "ORG": "#7aecec",
                            "GPE": "#feca74",
                            "DATE": "#ff9561",
                            "MONEY": "#9cc9cc",
                            "TIME": "#ffeb80",
                            "PERCENT": "#c887fb",
                            "LOC": "#bfe1d9",
                            "PRODUCT": "#bfeeb7",
                            "EVENT": "#e4e7d2"
                        }
                        
                        options = {"ents": selected_entities, "colors": colors}
                        html = displacy.render(doc, style="ent", options=options)
                        st.markdown(html, unsafe_allow_html=True)
                    
                    # Entity table
                    if show_table:
                        st.subheader("📋 Entity Details")
                        df = pd.DataFrame(entities, columns=["Entity", "Type", "Start", "End"])
                        st.dataframe(df, use_container_width=True)
                    
                    # Statistics
                    st.subheader("📈 Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Entities", len(entities))
                    
                    with col2:
                        entity_type_counts = pd.DataFrame(entities, columns=["Entity", "Type", "Start", "End"])["Type"].value_counts()
                        most_common = entity_type_counts.index[0] if len(entity_type_counts) > 0 else "N/A"
                        st.metric("Most Common Type", most_common)
                    
                    with col3:
                        unique_entities = len(set([e[0] for e in entities]))
                        st.metric("Unique Entities", unique_entities)
                    
                    # Entity type distribution
                    st.subheader("📊 Entity Type Distribution")
                    if len(entity_type_counts) > 0:
                        st.bar_chart(entity_type_counts)
                else:
                    st.warning("No entities found with the selected filters.")
        else:
            st.warning("Please enter some text to analyze.")
    
    # Information section
    with st.expander("ℹ️ About Entity Types"):
        st.markdown("""
        - **PERSON**: People, including fictional characters
        - **ORG**: Organizations, companies, agencies, institutions
        - **GPE**: Geopolitical entities (countries, cities, states)
        - **DATE**: Absolute or relative dates or periods
        - **MONEY**: Monetary values, including unit
        - **TIME**: Times smaller than a day
        - **PERCENT**: Percentage values
        - **LOC**: Non-GPE locations, mountain ranges, bodies of water
        - **PRODUCT**: Objects, vehicles, foods, etc.
        - **EVENT**: Named hurricanes, battles, wars, sports events
        """)

# Footer
st.markdown("---")
st.markdown("Built with spaCy and Streamlit | 🔍 NER Project")

