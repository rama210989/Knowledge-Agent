import streamlit as st

st.set_page_config(page_title="Support Ticket Search", layout="centered")

st.title("🔍 Support Ticket Similarity Search")
st.markdown("Type a query to find similar past tickets.")

query = st.text_input("Enter your query:")

if query:
    st.write("You searched for:", query)
    st.info("This is where we'll show the top matching tickets.")

import pandas as pd
import numpy as np
import faiss
import openai
from openai.embeddings_utils import get_embedding

# Set your OpenAI API key (or use st.secrets later for deployment)
openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else "sk-..."

# Load data
df = pd.read_csv("support_tickets.csv")

# Load or create embeddings
if 'embedding' not in df.columns:
    st.warning("Embeddings not found in CSV. Generating now...")
    df['embedding'] = df['description'].apply(lambda x: get_embedding(x, engine="text-embedding-ada-002"))
    df.to_csv("support_tickets.csv", index=False)  # Optional: save embeddings

# Convert embeddings to numpy array
embedding_matrix = np.array(df['embedding'].tolist())

# Build FAISS index
dimension = len(embedding_matrix[0])
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

# Process user query
if query:
    query_embedding = get_embedding(query, engine="text-embedding-ada-002")
    query_vector = np.array(query_embedding).astype("float32").reshape(1, -1)

    k = 5
    distances, indices = index.search(query_vector, k)

    st.subheader("🎯 Top Similar Tickets:")
    for idx, dist in zip(indices[0], distances[0]):
        st.markdown(f"**🎟️ Ticket #{idx}** (Distance: `{dist:.4f}`)")
        st.write(df.iloc[idx]['description'])
        st.markdown("---")

