import streamlit as st
import requests

st.title("titanic_chatbot")

user_query = st.text_input("Ask a question about Titanic passengers:", key="user_query_input")

if user_query:
    response = requests.get("http://127.0.0.1:8000/query", params={"question": user_query})
    if response.status_code == 200:
        st.write(response.json()["response"])
    else:
        st.write("Error fetching data from the API. Ensure FastAPI is running.")
