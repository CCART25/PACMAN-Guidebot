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
openai.api_key = st.secrets["OPENAI_API_KEY"]
EMBEDDING_MODEL = "text-embedding-3-small"

# === Load clauses ===
@st.cache_data
def load_data():
    return pd.read_csv("pacman_clauses.csv")

data = load_data()

# === Embedding utilities ===
def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

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

# === App Title ===
st.set_page_config(page_title="PACMAN GuideBot", layout="wide")
st.title("PACMAN GuideBot")

query = st.text_area("Enter your question below:")

# === Load vector model ===
@st.cache_resource
def load_model():
    embeddings, ids = embed_all_clauses(data)
    index = build_faiss_index(embeddings)
    return index, ids, embeddings

# === Log handling ===
def store_query_data(query, results, feedback=None):
    ref_id = f"PAC-{uuid.uuid4().hex[:8]}"
    timestamp = datetime.datetime.now().isoformat()
    log_data = {
        "reference_id": ref_id,
        "timestamp": timestamp,
        "query": query,
        "results": results,
        "feedba
