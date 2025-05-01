import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
import ast  # 👈 Needed to safely parse stringified lists from CSV

st.set_page_config(page_title="Support Ticket Similarity Search", layout="centered")

st.title("🔍 Support Ticket Similarity Search")
st.markdown("Type a query to find similar past tickets or get a GPT-based solution.")

query = st.text_input("Enter your query:")

# Set your OpenAI API key (securely loaded from Streamlit Cloud secrets)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Helper function to get embedding directly from OpenAI API
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=text, model=model)
    return response["data"][0]["embedding"]

# Load data
df = pd.read_csv("support_tickets.csv")

# Ensure embeddings column exists and is in correct format
if 'embedding' not in df.columns:
    st.warning("Embeddings not found in CSV. Generating now...")
    df['embedding'] = df['description'].apply(lambda x: get_embedding(x))
    df.to_csv("support_tickets.csv", index=False)
elif isinstance(df['embedding'].iloc[0], str):
    df['embedding'] = df['embedding'].apply(ast.literal_eval)

# Convert embeddings to numpy array
embedding_matrix = np.array(df['embedding'].tolist()).astype("float32")

# Build FAISS index
dimension = len(embedding_matrix[0])
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

# Function to call GPT for fallback solution
def get_gpt_solution(query):
    prompt = f"""A user asked the following technical support question:

"{query}"

No similar issues were found in the support ticket database. Please provide a helpful, step-by-step solution or advice related to this issue."""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    
    return response["choices"][0]["message"]["content"]

# Distance threshold: below this is considered relevant
SIMILARITY_THRESHOLD = 0.4

# Process user query
if query:
    st.write("You searched for:", query)

    # Get embedding for user query
    query_embedding = get_embedding(query)
    query_vector = np.array(query_embedding).astype("float32").reshape(1, -1)

    # Search for top-k matches
    k = 5
    distances, indices = index.search(query_vector, k)

    # Filter matches below similarity threshold
    matches = [(idx, dist) for idx, dist in zip(indices[0], distances[0]) if dist < SIMILARITY_THRESHOLD]

    if matches:
        st.subheader("🎯 Top Similar Tickets:")
        for idx, dist in matches:
            st.markdown(f"**🎟️ Ticket #{df.iloc[idx]['ticket_id']}** (Distance: `{dist:.4f}`)")
            st.write(f"**Title:** {df.iloc[idx]['title']}")
            st.write(f"**Description:** {df.iloc[idx]['description']}")
            st.write(f"**Resolution:** {df.iloc[idx]['resolution']}")
            st.markdown("---")
    else:
        st.warning("No highly similar past tickets found.")
        st.subheader("💡 Suggested Resolution (via GPT)")
        gpt_response = get_gpt_solution(query)
        st.write(gpt_response)
