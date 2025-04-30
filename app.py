import streamlit as st

st.set_page_config(page_title="Support Ticket Search", layout="centered")

st.title("🔍 Support Ticket Similarity Search")
st.markdown("Type a query to find similar past tickets.")

query = st.text_input("Enter your query:")

if query:
    st.write("You searched for:", query)
    st.info("This is where we'll show the top matching tickets.")
