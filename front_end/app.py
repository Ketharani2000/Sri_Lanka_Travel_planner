import streamlit as st
import requests

st.title("Sri Lanka Travel Planner")
st.write("Ask me anything about traveling in Sri Lanka!")

question = st.text_input("Your question:")

if st.button("Ask"):
    if question:
        response = requests.post(
            "http://localhost:8000/ask",
            json={"question": question}
        ).json()
        st.write("**Answer:**", response["answer"])
    else:
        st.warning("Please enter a question!")