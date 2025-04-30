import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai

st.set_page_config(page_title="Support Ticket Search", layout="centered")

st.title("🔍 Support Ticket Similarity Search")
st.markdown("Type a query to find similar past tickets.")

query = st.text_input("Enter your query:")

if query:
    st.write("You searched for:", query)
    st.info("This is where we'll show the top matching tickets.")

# Set your OpenAI API key (securely loaded from Streamlit Cloud secrets)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Helper function to get embedding directly from OpenAI API
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=text, model=model)
    return response["data"][0]["embedding"]

# Load data
df = pd.read_csv("support_tickets.csv")

# Load or create embeddings
if 'embedding' not in df.columns:
    st.warning("Embeddings not found in CSV. Generating now...")
    df['embedding'] = df['description'].apply(lambda x: get_embedding(x))
    df.to_csv("support_tickets.csv", index=False)

# Convert embeddings to numpy array
embedding_matrix = np.array(df['embedding'].tolist()).astype("float32")

# Build FAISS index
dimension = len(embedding_matrix[0])
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

# Process user query
if query:
    query_embedding = get_embedding(query)
    query_vector = np.array(query_embedding).astype("float32").reshape(1, -1)

    k = 5
    distances, indices = index.search(query_vector, k)

    st.subheader("🎯 Top Similar Tickets:")
    for idx, dist in zip(indices[0], distances[0]):
        st.markdown(f"**🎟️ Ticket #{idx}** (Distance: `{dist:.4f}`)")
        st.write(df.iloc[idx]['description'])
        st.markdown("---")
