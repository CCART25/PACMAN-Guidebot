
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
import tiktoken
import uuid
import datetime
import json
from typing import List
from fpdf import FPDF
from collections import Counter
import os

# === Config ===
# Commented out for offline/demo use:
# openai.api_key = st.secrets["OPENAI_API_KEY"]
EMBEDDING_MODEL = "text-embedding-3-small"

# === Load clauses ===
@st.cache_data
def load_data():
    return pd.read_csv("pacman_clauses.csv")

data = load_data()

# === Embedding utilities ===
def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    np.random.seed(hash(text) % 2**32)
    return np.random.rand(128).tolist()

@st.cache_resource
def embed_all_clauses(data: pd.DataFrame):
    texts = data['clause_text'].tolist()
    ids = data['clause_number'].tolist()
    embeddings = [get_embedding(text) for text in texts]
    return embeddings, ids

@st.cache_resource
def build_faiss_index(embeddings: List[List[float]]) -> faiss.IndexFlatL2:
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index

# === Glossary ===
GLOSSARY = {
    "ARF": "Accompanied Resident Family – your family lives with you at your posting location.",
    "URF": "Unaccompanied Resident Family – your family lives elsewhere while you're posted.",
    "MBR": "Member Benefit Recipient – someone who qualifies you for benefits (e.g., spouse or dependent).",
    "DHOAS": "Defence Home Ownership Assistance Scheme – support for purchasing property.",
    "PSL": "Primary Service Location – the base or unit where you're posted."
}

def match_glossary_terms(text: str) -> dict:
    matches = {}
    for term, definition in GLOSSARY.items():
        if term in text:
            matches[term] = definition
    return matches

# === PDF Export ===
class PDFExporter(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "PACMAN GuideBot Result", ln=True, align="C")

    def add_clause(self, clause_id, title, text, glossary):
        self.set_font("Arial", "B", 10)
        self.cell(0, 10, f"Clause {clause_id}: {title}", ln=True)
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 10, text)
        if glossary:
            self.set_font("Arial", "I", 9)
            self.cell(0, 10, "Glossary:", ln=True)
            for term, defn in glossary.items():
                self.multi_cell(0, 10, f"- {term}: {defn}")
        self.ln(5)

# === App UI ===
st.set_page_config(page_title="PACMAN GuideBot", layout="wide")
st.title("PACMAN GuideBot")

query = st.text_area("Enter your question below:")

@st.cache_resource
def load_model():
    embeddings, ids = embed_all_clauses(data)
    index = build_faiss_index(embeddings)
    return index, ids, embeddings

def store_query_data(query, results, feedback=None):
    ref_id = f"PAC-{uuid.uuid4().hex[:8]}"
    timestamp = datetime.datetime.now().isoformat()
    log_data = {
        "reference_id": ref_id,
        "timestamp": timestamp,
        "query": query,
        "results": results,
        "feedback": feedback
    }
    with open("query_log.jsonl", "a") as f:
        f.write(json.dumps(log_data) + "\n")
    return ref_id

if query:
    st.write("Searching PACMAN for answers...")
    index, ids, embeddings = load_model()
    query_embedding = get_embedding(query)
    D, I = index.search(np.array([query_embedding]).astype("float32"), k=3)

    pdf = PDFExporter()
    pdf.add_page()
    result_log = []

    for i in I[0]:
        result = data.iloc[i]
        clause = result['clause_number']
        title = result['clause_title']
        text = result['clause_text']
        url = result['source_url']

        glossary_matches = match_glossary_terms(text)

        st.subheader(f"Clause {clause}: {title}")
        st.markdown("💬 **Why this might be relevant to your situation:**")
        st.write(
            f"This clause relates to your question because it addresses **{title.lower()}**, "
            "which could be a key consideration in the situation you've described."
        )

        st.markdown("🧾 **What PACMAN says:**")
        st.write(text)
        st.markdown(f"[🔗 View Full Clause]({url})")

        if glossary_matches:
            st.markdown("📘 **Glossary Terms in this Clause:**")
            for term, definition in glossary_matches.items():
                st.markdown(f"- **{term}**: {definition}")

        pdf.add_clause(clause, title, text, glossary_matches)

        result_log.append({
            "clause_number": clause,
            "clause_title": title,
            "clause_text": text,
            "source_url": url,
            "glossary": glossary_matches
        })

        st.markdown("---")

    feedback = st.radio("Did this answer help you?", ("Yes", "No"))
    ref_id = store_query_data(query, result_log, feedback)
    st.success(f"Reference ID: {ref_id}")

    if st.button("📄 Export to PDF"):
        filename = f"pacman_response_{ref_id}.pdf"
        pdf.output(filename)
        with open(filename, "rb") as f:
            st.download_button(
                label="Download PDF",
                data=f,
                file_name=filename,
                mime="application/pdf"
            )
